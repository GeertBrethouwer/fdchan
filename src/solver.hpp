// FDCHAN: incompressible channel flow, Fourier in x, finite differences in y (compact or explicit) and z
// (explicit).
//
// Parallelisation: z-slabs (comm.hpp). Each rank (one GPU) holds nz = nzl local z-planes of the nzg global ones,
// for all x-modes and all y. z-stencils read p/2 halo planes from the neighbouring ranks (halo exchange); the
// pressure is solved after an all-to-all transpose to kx-slabs (all z, all y, nkl x-modes). x-FFTs and y-solves
// are local. With one rank the same code runs (the halos are periodic copies of the rank's own planes).
//
// Storage on a rank:
//   spectral fields     u(kx, z, y)  complex, index (j*nz + k)*nkx + i,  kx = 0..nx/2-1, k = 0..nz-1 (local)
//   padded work arrays  P[0..5]      one x-line holds mx/2+1 complex or mx (+2) reals, index (j*nz+k)*lp + i
//   halo buffers        [lower|upper][j][q][i], q = 0..h-1 (lower: k = q-h, upper: k = nz+q), width nkx
//   work arrays         T1 (kx,z,y); T2 for the pressure (z global, local kx, y) = (j*nkl + i)*nzg + k
//
// y derivatives are A f' = B f along each (kx, z) line (A tridiagonal, the identity for explicit schemes):
// one thread per line, a sweep down and a sweep up. Every line is local to a GPU in a z-slab decomposition.
//
// One RK3 stage:
//   1. T1 = Dy u, T2 = Dy w; P0..P5 = u, v, w, omega_x, omega_y, omega_z (padded)  -> 6 inverse x-FFTs
//   2. N = u x omega in P3..P5 (pointwise)                                          -> 3 forward x-FFTs
//   3. H = N + nu Lap(u);  r = u + dt (gam H + zet H_old)   (stored in P3..P5)
//   4. pressure: right-hand side (banded) -> T2 -> z-FFT -> banded LU with pivoting per (kx, +-kz) for
//      (phi, D1 phi) -> inverse z-FFT -> T1 = phi
//   5. u = r - grad phi at the interior points; constant flow rate correction
// The discrete divergence (Fourier x, D1z, D1y) is zero at every point, walls included.
#pragma once
#include <chrono>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <utility>
#include <string>
#include <vector>

#include "backend.hpp"
#include "fft.hpp"
#include "operators.hpp"
#include "params.hpp"
#include "psolve.hpp"
#include "comm.hpp"
#include "timer.hpp"

namespace fd {

constexpr int MAXW = 12;  // max stencil width stored per row
constexpr int MAXP2 = 4;  // max half width of the z stencils (order 8)
constexpr int MAXB = 24;  // max band width (kl + ku + 1) of the scalar y-systems

struct ZSt {  // periodic central stencils in z, a[l + h], l = -h..h, already divided by dz (dz^2)
  int h;
  real a1[2 * MAXP2 + 1], a2[2 * MAXP2 + 1];
};

// Compact z-derivative (z_scheme = compact): al f'_{k-1} + f'_k + al f'_{k+1} = sum_l c_l (f_{k+l} - f_{k-l}) (D1), or
// al f''_{k-1} + f''_k + al f''_{k+1} = sum_l c_l (f_{k+l} - 2 f_k + f_{k-l}) (D2), periodic in z. Distributed over the
// z-slabs as in x3d2 (DistD2-TDS): every rank solves its block of a line with the Thomas algorithm (y), its block
// then depends on the outside only through x_{-1} (last value of the previous rank) and x_n (first value of the
// next one): x = y + x_{-1} p + x_n q, with A p = -al e_0 and A q = -al e_{n-1} (decay vectors, |p_k|, |q_{n-1-k}| ~ r^k).
// The ranks exchange y_0 and y_{n-1} with their neighbours and solve a 2x2 system per block boundary, dropping the
// terms of order r^n (exact to round-off when n is large enough, checked at start-up). One rank: exact 2x2 solve.
struct ZCmp {
  int L;            // half width of the right-hand side (= halo width)
  real al;          // alpha of the tridiagonal left-hand side
  real c[MAXP2];    // c_l / dz (D1) or c_l / dz^2 (D2)
  const real *cp, *inv, *pv, *qv;  // Thomas factors and decay vectors of the local block (length nz)
  real p0o, qn1o, p0n, qn1p, il, ir;  // block-boundary coupling with the neighbours
  real e11, e12, e21, e22;            // one rank: inverse of the exact periodic 2x2 system
};

// A compact z-derivative as its kernels read it (fused second phase): the block solution y and, per line, the
// values x_{-1} and x_n of the neighbouring blocks; x_k = y_k + p_k x_{-1} + q_k x_n is formed where it is read,
// so the derivative is never written in final form (one pass less over the field to write, one to read).
struct ZDer {
  const cplx* y = nullptr;   // block solution, layout (j*nz + k)*nkx + i
  const cplx* xb = nullptr;  // x_{-1} at [ln], x_n at [nlines + ln], ln = j*nkx + i
  const real *pv = nullptr, *qv = nullptr;
  int nz = 0, nkx = 0;
  idx nlines = 0;
  HD bool valid() const { return y != nullptr; }
  HD cplx at(int j, int k, int i) const {
    const idx ln = idx(j) * nkx + i;
    return y[(idx(j) * nz + k) * nkx + i] + pv[k] * xb[ln] + qv[k] * xb[nlines + ln];
  }
};

// coefficients of the tridiagonal compact scheme of the given order for the d-th derivative (d = 1, 2) on a uniform
// grid, from the Taylor conditions: unknowns al, c_1..c_L with L = order/2 - 1
inline void compact_coeffs(int d, int order, double& al, std::vector<double>& c) {
  const int L = order / 2 - 1, n = L + 1;
  std::vector<double> M(size_t(n * n), 0.0), b(static_cast<size_t>(n), 0.0);
  auto fact = [](int m) { double f = 1; for (int i = 2; i <= m; ++i) f *= i; return f; };
  for (int e = 0; e < n; ++e) {
    const int m = (d == 1) ? 2 * e + 1 : 2 * e + 2;  // order of the derivative matched
    // sum_l c_l 2 l^m / m!  -  al * 2 / (m - d)!  =  [m == d]   (the LHS f^(d) of the row itself has coefficient 1)
    for (int l = 1; l <= L; ++l) M[size_t(e * n + l)] = 2.0 * std::pow(double(l), m) / fact(m);
    M[size_t(e * n + 0)] = -2.0 / fact(m - d);
    b[size_t(e)] = (m == d) ? 1.0 : 0.0;
  }
  // Gaussian elimination with partial pivoting
  for (int k = 0; k < n; ++k) {
    int pr = k;
    for (int r = k + 1; r < n; ++r)
      if (std::abs(M[size_t(r * n + k)]) > std::abs(M[size_t(pr * n + k)])) pr = r;
    for (int cc = 0; cc < n; ++cc) std::swap(M[size_t(k * n + cc)], M[size_t(pr * n + cc)]);
    std::swap(b[size_t(k)], b[size_t(pr)]);
    for (int r = k + 1; r < n; ++r) {
      const double f = M[size_t(r * n + k)] / M[size_t(k * n + k)];
      for (int cc = k; cc < n; ++cc) M[size_t(r * n + cc)] -= f * M[size_t(k * n + cc)];
      b[size_t(r)] -= f * b[size_t(k)];
    }
  }
  std::vector<double> x(static_cast<size_t>(n));
  for (int k = n - 1; k >= 0; --k) {
    double s = b[size_t(k)];
    for (int cc = k + 1; cc < n; ++cc) s -= M[size_t(k * n + cc)] * x[size_t(cc)];
    x[size_t(k)] = s / M[size_t(k * n + k)];
  }
  al = x[0];
  c.assign(x.begin() + 1, x.end());
}

// Thomas factors of tridiag(al, 1, al) of size n and the solution of A x = rhs (host, double)
struct TriLocal {
  std::vector<double> cp, inv;
  void init(int n, double al) {
    cp.assign(size_t(n), 0.0); inv.assign(size_t(n), 0.0);
    for (int k = 0; k < n; ++k) {
      const double den = (k == 0) ? 1.0 : 1.0 - al * cp[size_t(k - 1)];
      inv[size_t(k)] = 1.0 / den;
      cp[size_t(k)] = al / den;
    }
  }
  std::vector<double> solve(const std::vector<double>& r, double al) const {
    const size_t n = cp.size();
    std::vector<double> x(n);
    for (size_t k = 0; k < n; ++k) x[k] = (k == 0 ? r[k] : r[k] - al * x[k - 1]) * inv[k];
    for (size_t k = n - 1; k-- > 0;) x[k] -= cp[k] * x[k + 1];
    return x;
  }
};

// NF fields swept together along y: the inputs and the arrays that hold the forward-sweep values
template <int NF>
struct YLines {
  const cplx* in[NF];
  cplx* tmp[NF];
};

// value of f at local plane kk (may lie in the halo), mode i, row j; L = line length of f, W = halo width
HD inline cplx zv(const cplx* f, const cplx* hal, idx L, int i, int kk, int j, int nzl, int h, int ny, int W) {
  if (kk >= 0 && kk < nzl) return f[(idx(j) * nzl + kk) * L + i];
  if (kk < 0) return hal[(idx(j) * h + (kk + h)) * W + i];
  return hal[idx(ny) * h * W + (idx(j) * h + (kk - nzl)) * W + i];
}

// pointwise part of rhs(): H = N + nu Lap f, r = f + dtg H + dtz H_old at spectral point n (padded point o)
struct RhsPoint {
  const cplx *f, *hf;
  ZDer d2z;  // compact z: D2z f; explicit: not valid, the stencil of z is used
  cplx *h, *r;
  const real* kx;
  ZSt z;
  int nkx, nz, ny;
  real sc, nuu, dtg, dtz, force;
  int nb = 0;              // Boussinesq buoyancy: - Ri_n g_c s_n of the scalars (bco = -Ri_n g_c)
  const cplx* bs[NSMAX];
  real bco[NSMAX];
  HD void operator()(idx n, idx o, int i, int k, int j, cplx d2y) const {
    if (j == 0 || j == ny - 1) {
      r[o] = cplx(0, 0);
      h[n] = cplx(0, 0);
      return;
    }
    const cplx fc = f[n];
    cplx lap = (-kx[i] * kx[i]) * fc + d2y;
    if (d2z.valid()) lap += d2z.at(j, k, i);
    else
      for (int l = -z.h; l <= z.h; ++l) lap += z.a2[l + z.h] * zv(f, hf, nkx, i, k + l, j, nz, z.h, ny, nkx);
    cplx hh = sc * r[o] + nuu * lap;
    if (i == 0) hh.re += force;
    for (int q = 0; q < nb; ++q) hh += bco[q] * bs[q][n];
    r[o] = fc + dtg * hh + dtz * h[n];
    h[n] = hh;
  }
};

// pointwise part of nonlinear(): u, v, w and the vorticity into the padded arrays C0..C5 at padded point n
struct VortPoint {
  const cplx *U, *V, *W, *hU, *hV;
  ZDer dzU, dzV;  // compact z: D1z u, D1z v
  cplx *C0, *C1, *C2, *C3, *C4, *C5;
  const real* kx;
  ZSt z;
  int nkx, nz, ny;
  HD void operator()(idx n, idx s, int i, int k, int j, cplx dyu, cplx dyw) const {
    cplx dzu(0, 0), dzv(0, 0);
    if (dzU.valid()) {
      dzu = dzU.at(j, k, i); dzv = dzV.at(j, k, i);
    } else {
      for (int l = -z.h; l <= z.h; ++l) {
        if (l == 0) continue;
        dzu += z.a1[l + z.h] * zv(U, hU, nkx, i, k + l, j, nz, z.h, ny, nkx);
        dzv += z.a1[l + z.h] * zv(V, hV, nkx, i, k + l, j, nz, z.h, ny, nkx);
      }
    }
    cplx uu = U[s], vv = V[s], ww = W[s];
    cplx ox = dyw - dzv, oy = dzu - itimes(kx[i], ww), oz = itimes(kx[i], vv) - dyu;
    if (i == 0) { uu.im = 0; vv.im = 0; ww.im = 0; ox.im = 0; oy.im = 0; oz.im = 0; }
    C0[n] = uu; C1[n] = vv; C2[n] = ww; C3[n] = ox; C4[n] = oy; C5[n] = oz;
  }
  HD void zero(idx n) const {
    const cplx zz(0, 0);
    C0[n] = zz; C1[n] = zz; C2[n] = zz; C3[n] = zz; C4[n] = zz; C5[n] = zz;
  }
};

class Solver {
 public:
  Params P;
  int nx, ny, nz, nkx, mx, lp, ld, ntz, K = 0;
  // decomposition: nz = local z-planes (z0 = first global plane), nzg = global; kx0, nkl = local x-modes in the
  // pressure solve; zst/kxst = starts of all ranks
  int nzg = 0, z0 = 0, kx0 = 0, nkl = 0, hz_halo = 0, nrank = 1, myrank = 0;
  std::vector<int> zst, kxst;
  idx nsp, npad, ncols;
  double nu, hz;
  std::vector<double> y, q, dyc, kx_h;
  YDeriv D1, D2;
  double kap1 = 1, kap1y = 1, lam2y = 0, lam2z = 0, qint = 0;
  ZSt zs;
  bool zcomp = false;  // z_scheme = compact
  ZCmp zc[2];          // D1z, D2z
  double zc_al[2] = {0, 0}, zc_r[2] = {0, 0};
  int zc_nmin = 0;
  DArray<real> d_zcp[2], d_zinv[2], d_zpv[2], d_zqv[2];
  DArray<cplx> DZ1, DZ2, d_zx;  // compact z: block solutions y; first/last values of the lines for the exchange
  DArray<cplx> ZB1, ZB2;        // compact z: x_{-1}, x_n per line of DZ1, DZ2
  // scalars (P.nscal): fields, explicit right-hand sides of this and the previous stage, diffusivities
  int nsc = 0;
  DArray<cplx> Sf[NSMAX], NSf[NSMAX], NSo[NSMAX];
  DArray<cplx> hbS;  // halo of the scalar being processed
  double kap[NSMAX] = {0}, gvec[3] = {0, 0, 0};
  // scalar y-solve (theta-method): band of A2 and B2 per row (offsets -skl..sku), the wall rows of D1, and per kx the
  // banded LU (U rows, multipliers), the 4 homogeneous solutions and the inverse influence matrix
  int skl = 0, sku = 0, snb = 0, nc1 = 0, ncn = 0;
  DArray<real> d_a2b, d_b2b, d_c1, d_cn, d_sU, d_sL, d_sX, d_sE;
  // device arrays
  struct YDev {  // a y derivative on the device
    const int* s;
    const real *c, *lo, *m, *up, *id;
    int W, n;
    bool compact;
  };
  DArray<int> d_ys[2];
  DArray<real> d_yc[2], d_ylo[2], d_ym[2], d_yup[2], d_yid[2];
  DArray<real> d_kx, d_kzp2, d_G0, d_G1, d_q, d_dyc, d_su, d_sy;
  DArray<real> d_sl;            // LU multipliers of the pressure columns (factorised once, see factor_columns)
  DArray<unsigned char> d_sp;   // their pivot choices
  bool factored = false;
  DArray<cplx> u, v, w, hx, hy, hz_, T1, T2;
  DArray<real> Pb[10];  // 6 always, 3 more for the advective form or the scalars, 1 more for scalar_form divergence/skew
  DArray<double> d_prof, d_scal;
  DArray<cplx> hbA, hbB, hbC, hsend, sbuf, rbuf;  // halo buffers, transpose buffers
  DArray<cplx> hrecv;  // halo3: the received planes of u, v, w (hsend holds 3 fields)
  // overlap_scalars: the scalars run on stream s_scal with their own scratch arrays (swapped with the main ones while
  // their work is issued, see ScalarContext) and NCCL communicator (channel 1); s_comm: pipelined pressure step
  bool ovs = false;
  stream_t s_scal{}, s_comm{};
  DArray<cplx> T1s, T2s, hsend_s, DZ1s, DZ2s, ZB1s, ZB2s, d_zx_s;
  Event ev_rhs, ev_scal, ev_pack, ev_done;
  // p_chunks > 1: chunks of the local kx (pressure step); layouts in setup_pipe
  int pch = 1;
  std::vector<int> kc0;                                  // local kx start of chunk c (c = 0..pch)
  std::vector<std::vector<int>> c_sc, c_sd, c_rc, c_rd;  // per chunk: all-to-all counts and displacements (reals)
  DArray<int> d_kxch, d_kxco, d_ncc;                     // per global kx: chunk, offset in it; per (c, r): kx count
  DArray<idx> d_sdc, d_rdc;                              // per (c, r): send / receive block start (complex)
  std::vector<std::unique_ptr<ZFFT>> zfc;                // z-FFT plan per chunk
  std::vector<std::unique_ptr<Event>> evF, evB;
  DArray<int> d_kxown, d_kxoff, d_zown, d_zoff;
  std::vector<int> a2a_sc, a2a_sd, a2a_rc, a2a_rd;  // all-to-all counts and displacements (in reals)
  DArray<idx> d_sd, d_rd;
  DArray<int> d_cnt, d_nzr;
  double dpdx_acc = 0;
  XFFT xfft;
  ZFFT zfft;
  // state
  double time = 0, dt = 0, dpdx_step = 0;
  double pscale = 0;  // p = pscale * phi of the last stage (in T1 after advance(), until T1 is reused)
  long step = 0;
  int W1 = 0, W2 = 0;

  explicit Solver(const Params& prm) : P(prm) {
    nx = P.nx; ny = P.ny; nzg = P.nz;
    nkx = nx / 2;
    nrank = g_comm.size; myrank = g_comm.rank;
    zst.resize(size_t(nrank + 1));
    kxst.resize(size_t(nrank + 1));
    for (int r = 0; r <= nrank; ++r) { zst[size_t(r)] = block_start(nzg, nrank, r); kxst[size_t(r)] = block_start(nkx, nrank, r); }
    z0 = zst[size_t(myrank)];
    nz = zst[size_t(myrank + 1)] - z0;
    kx0 = kxst[size_t(myrank)];
    nkl = kxst[size_t(myrank + 1)] - kx0;
    hz_halo = (P.z_scheme == "compact") ? P.fd_order_z / 2 - 1 : P.fd_order_z / 2;
    if (nrank > 1 && nz < hz_halo) fatal("z-slab thinner than the stencil half width: use fewer ranks");
    if (nkl < 1) fatal("more ranks than x-modes");
    mx = P.dealias ? 3 * nx / 2 : nx;
    if (mx % 2) mx += 1;
    lp = mx / 2 + 1;
    ld = 2 * lp;
    ntz = nzg / 2 + 1;
    nsp = idx(nkx) * nz * ny;
    npad = idx(ld) * nz * ny;
    ncols = idx(nkl) * ntz;
    nu = 1.0 / P.re;
    hz = P.lz / nzg;
    if (P.nl_form != "rotational" && P.nl_form != "advective") fatal("FDCHAN: nl_form must be rotational or advective");
    if (P.rk_stages != 3 && P.rk_stages != 4) fatal("FDCHAN: rk_stages must be 3 or 4");
    setup_operators();
    nsc = P.nscal;
    {
      int npb = (P.nl_form == "advective" || nsc > 0) ? 9 : 6;
      if (nsc > 0 && P.scal_form != "advective") npb = 10;
      for (int c = 0; c < npb; ++c) Pb[c].alloc(npad);
    }
    u.alloc(nsp); v.alloc(nsp); w.alloc(nsp);
    hx.alloc(nsp); hy.alloc(nsp); hz_.alloc(nsp);
    T1.alloc(nsp);
    const idx nt2 = idx(ny) * nkl * nzg;
    T2.alloc(std::max(nsp, nt2));
    sbuf.alloc(std::max(nsp, nt2));
    rbuf.alloc(std::max(nsp, nt2));
    const idx nh = 2 * idx(hz_halo) * ny * nkx;
    hbA.alloc(nh); hbB.alloc(nh); hbC.alloc(nh); hsend.alloc(3 * nh); hrecv.alloc(3 * nh);
    if (nsc > 0) {
      hbS.alloc(nh);
      for (int n = 0; n < nsc; ++n) { Sf[n].alloc(nsp); NSf[n].alloc(nsp); NSo[n].alloc(nsp); kap[n] = nu / P.pr[n]; }
      const double gn = std::sqrt(P.gravity[0] * P.gravity[0] + P.gravity[1] * P.gravity[1] + P.gravity[2] * P.gravity[2]);
      for (int c = 0; c < 3; ++c) gvec[c] = P.gravity[c] / gn;
      setup_scalar_solve();
    }
    setup_transpose();
    d_su.alloc(idx(2 * K + 1) * 2 * ny * ncols);
    d_sy.alloc(idx(NRHS) * 2 * ny * ncols);
    d_sl.alloc(idx(K) * 2 * ny * ncols);
    d_sp.alloc(idx(2) * ny * ncols);
    if (zcomp) {
      DZ1.alloc(nsp); DZ2.alloc(nsp);
      d_zx.alloc(4 * idx(ny) * nkx);
      ZB1.alloc(2 * idx(ny) * nkx); ZB2.alloc(2 * idx(ny) * nkx);
    }
    d_prof.alloc(4 * idx(ny));
    d_scal.alloc(8);
    xfft.init(mx, idx(nz) * ny);
    zfft.init(nzg, idx(nkl) * ny);
    dt = P.dt;
    // overlap of the scalars with the pressure step
    ovs = P.overlap_scalars && nsc > 0;
    if (ovs) {
      s_scal = new_stream();
      g_comm.init_nccl2();
      T1s.alloc(nsp); T2s.alloc(nsp); hsend_s.alloc(3 * nh);
      if (zcomp) {
        DZ1s.alloc(nsp); DZ2s.alloc(nsp);
        d_zx_s.alloc(4 * idx(ny) * nkx);
        ZB1s.alloc(2 * idx(ny) * nkx); ZB2s.alloc(2 * idx(ny) * nkx);
      }
    }
    setup_pipe();
  }
  static void swap_arr(DArray<cplx>& a, DArray<cplx>& b) { std::swap(a.p, b.p); std::swap(a.n, b.n); }
  void swap_scalar_scratch() {
    swap_arr(T1, T1s); swap_arr(T2, T2s); swap_arr(hsend, hsend_s);
    if (zcomp) { swap_arr(DZ1, DZ1s); swap_arr(DZ2, DZ2s); swap_arr(ZB1, ZB1s); swap_arr(ZB2, ZB2s); swap_arr(d_zx, d_zx_s); }
  }
  // while it exists, the work issued goes to the scalar stream, with the scalar scratch arrays and NCCL channel 1;
  // kernels take their pointers when they are issued, so the arrays can be swapped back right after
  struct ScalarContext {
    Solver& S;
    OnStream os;
    explicit ScalarContext(Solver& s) : S(s), os(s.s_scal) {
      S.swap_scalar_scratch();
      g_comm.channel = 1;
      g_timer.paused = true;  // no timers (they would make the host wait for this stream)
    }
    ~ScalarContext() {
      g_timer.paused = false;
      g_comm.channel = 0;
      S.swap_scalar_scratch();
    }
  };

  // p_chunks > 1: the local kx of the pressure step in chunks. Send block of chunk c to rank r: (j, i'', k) with
  // i'' the kx of chunk c of r's kx-slab, k local; receive block of chunk c from rank s: (j, i'', k_s) with i'' in my
  // chunk c. Blocks are ordered chunk by chunk, so every chunk is one all-to-all with its own displacements. In the
  // kx-slab the pipelined step keeps T2 as (i, j, k global): a chunk is contiguous there.
  void setup_pipe() {
    pch = 1;
    if (P.p_chunks <= 1) return;
    int mincnt = nkx;
    for (int r = 0; r < nrank; ++r) mincnt = std::min(mincnt, kxst[size_t(r + 1)] - kxst[size_t(r)]);
    pch = std::max(1, std::min(P.p_chunks, mincnt));
    if (pch == 1) return;
    s_comm = new_stream();
    const int C = pch;
    auto cnt = [&](int r) { return kxst[size_t(r + 1)] - kxst[size_t(r)]; };
    auto cst = [&](int r, int c) { return block_start(cnt(r), C, c); };   // chunk start in r's kx-slab
    kc0.resize(size_t(C + 1));
    for (int c = 0; c <= C; ++c) kc0[size_t(c)] = cst(myrank, c);
    std::vector<idx> sdc(size_t(C) * nrank), rdc(size_t(C) * nrank);
    std::vector<int> ncc(size_t(C) * nrank);
    c_sc.assign(size_t(C), std::vector<int>(size_t(nrank), 0)); c_sd = c_sc; c_rc = c_sc; c_rd = c_sc;
    idx so = 0, ro = 0;
    for (int c = 0; c < C; ++c)
      for (int r = 0; r < nrank; ++r) {
        const int nr_c = cst(r, c + 1) - cst(r, c), nme = kc0[size_t(c + 1)] - kc0[size_t(c)];
        const int nzr = zst[size_t(r + 1)] - zst[size_t(r)];
        const idx ns = idx(ny) * nr_c * nz, nrv = idx(ny) * nme * nzr;
        ncc[size_t(c) * nrank + r] = nr_c;
        sdc[size_t(c) * nrank + r] = so; rdc[size_t(c) * nrank + r] = ro;
        c_sc[size_t(c)][size_t(r)] = int(2 * ns); c_sd[size_t(c)][size_t(r)] = int(2 * so);
        c_rc[size_t(c)][size_t(r)] = int(2 * nrv); c_rd[size_t(c)][size_t(r)] = int(2 * ro);
        so += ns; ro += nrv;
      }
    std::vector<int> kxch(static_cast<size_t>(nkx)), kxco(static_cast<size_t>(nkx));
    for (int r = 0; r < nrank; ++r)
      for (int c = 0; c < C; ++c)
        for (int o = cst(r, c); o < cst(r, c + 1); ++o) {
          const int i = kxst[size_t(r)] + o;
          kxch[size_t(i)] = c; kxco[size_t(i)] = o - cst(r, c);
        }
    d_kxch.alloc(nkx); d_kxch.upload(kxch);
    d_kxco.alloc(nkx); d_kxco.upload(kxco);
    d_ncc.alloc(idx(ncc.size())); d_ncc.upload(ncc);
    d_sdc.alloc(idx(sdc.size())); d_sdc.upload(sdc);
    d_rdc.alloc(idx(rdc.size())); d_rdc.upload(rdc);
    for (int c = 0; c < C; ++c) {
      zfc.emplace_back(new ZFFT());
      zfc.back()->init(nzg, idx(kc0[size_t(c + 1)] - kc0[size_t(c)]) * ny);
      evF.emplace_back(new Event());
      evB.emplace_back(new Event());
    }
  }

  // the pipelined middle of project(): t1 (kx, z local, y) -> kx-slabs -> z-FFT, banded solves, inverse z-FFT ->
  // back to t1, chunk by chunk. The all-to-alls run on s_comm in the order F_0 .. F_{C-1}, B_0 .. B_{C-1} (the
  // same on all ranks); chunk c is computed on the main stream when F_c has arrived, while the later chunks are
  // still in transit, and B_c is sent while chunk c + 1 is computed.
  void pressure_pipe(cplx* t1) {
    const int C = pch, nkx_ = nkx, nz_ = nz, ny_ = ny, nzg_ = nzg, nr = nrank;
    const int *own = d_kxown.p, *zown = d_zown.p, *zoff = d_zoff.p, *nzr = d_nzr.p;
    const int *kxch = d_kxch.p, *kxco = d_kxco.p, *ncc = d_ncc.p;
    const idx *sdc = d_sdc.p, *rdc = d_rdc.p;
    cplx *sb = sbuf.p, *rb = rbuf.p, *t2 = T2.p;
    real *sbr = reinterpret_cast<real*>(sb), *rbr = reinterpret_cast<real*>(rb);
    ptranspose(ny, nz_, nkx_,
        [=] HD(idx j, int k, int i) -> cplx { return t1[(j * nz_ + k) * nkx_ + i]; },
        [=] HD(idx j, int k, int i, cplx val) {
          const int r = own[i], c = kxch[i];
          const idx b = idx(c) * nr + r;
          sb[sdc[b] + (j * ncc[b] + kxco[i]) * nz_ + k] = val;
        });
    ev_pack.record();
    {
      OnStream os(s_comm);
      ev_pack.wait();
      for (int c = 0; c < C; ++c) {
        g_comm.alltoallv(sbr, c_sc[size_t(c)], c_sd[size_t(c)], rbr, c_rc[size_t(c)], c_rd[size_t(c)]);
        evF[size_t(c)]->record();
      }
    }
    for (int c = 0; c < C; ++c) {
      const int i0 = kc0[size_t(c)], ncl = kc0[size_t(c + 1)] - i0;
      const idx rbase = idx(c) * nr;
      evF[size_t(c)]->wait();
      // chunk c into T2 as (i, j, k global)
      pfor(idx(ncl) * ny_ * nzg_, [=] HD(idx m) {
        const int kg = int(m % nzg_);
        const idx r = m / nzg_;
        const int j = int(r % ny_), ic = int(r / ny_);
        const int s = zown[kg];
        t2[(idx(i0 + ic) * ny_ + j) * nzg_ + kg] = rb[rdc[rbase + s] + (idx(j) * ncl + ic) * nzr[s] + zoff[kg]];
      });
      cplx* tc = t2 + idx(i0) * ny_ * nzg_;
      zfc[size_t(c)]->exec_untimed(tc, -1);
      solve_columns(idx(i0) * ntz, idx(i0 + ncl) * ntz, true);
      zfc[size_t(c)]->exec_untimed(tc, +1);
      pfor(idx(ncl) * ny_ * nzg_, [=] HD(idx m) {
        const int kg = int(m % nzg_);
        const idx r = m / nzg_;
        const int j = int(r % ny_), ic = int(r / ny_);
        const int s = zown[kg];
        rb[rdc[rbase + s] + (idx(j) * ncl + ic) * nzr[s] + zoff[kg]] = t2[(idx(i0 + ic) * ny_ + j) * nzg_ + kg];
      });
      evB[size_t(c)]->record();
      OnStream os(s_comm);
      evB[size_t(c)]->wait();
      g_comm.alltoallv(rbr, c_rc[size_t(c)], c_rd[size_t(c)], sbr, c_sc[size_t(c)], c_sd[size_t(c)]);
      if (c == C - 1) ev_done.record();
    }
    ev_done.wait();
    ptranspose(ny, nkx_, nz_,
        [=] HD(idx j, int i, int k) -> cplx {
          const int r = own[i], c = kxch[i];
          const idx b = idx(c) * nr + r;
          return sb[sdc[b] + (j * ncc[b] + kxco[i]) * nz_ + k];
        },
        [=] HD(idx j, int i, int k, cplx val) { t1[(j * nz_ + k) * nkx_ + i] = val; });
  }

  // ---------------------------------------------------------------- set-up
  // ---------------------------------------------------------------- compact z (z_scheme = compact)
  void setup_compact_z(std::vector<double>& kzp2) {
    zcomp = true;
    const int pz = P.fd_order_z, L = pz / 2 - 1;
    zs.h = hz_halo;  // zv() reads the halos with this width; the explicit stencils are not used
    std::vector<double> c[2];
    for (int d = 0; d < 2; ++d) {
      compact_coeffs(d + 1, pz, zc_al[d], c[d]);
      const double al = zc_al[d];
      zc_r[d] = (1 - std::sqrt(1 - 4 * al * al)) / (2 * al);  // decay per point of the block coupling
    }
    // block coupling dropped between ranks is ~ r^n: n must make it negligible against the round-off of real
    const double eps = double(std::numeric_limits<real>::epsilon());
    zc_nmin = int(std::ceil(std::log(0.1 * eps) / std::log(std::max(zc_r[0], zc_r[1]))));
    if (nrank > 1)
      for (int r = 0; r < nrank; ++r)
        if (zst[size_t(r + 1)] - zst[size_t(r)] < zc_nmin) {
          char msg[256];
          std::snprintf(msg, sizeof(msg),
                        "compact z_scheme of order %d needs at least %d z-planes per rank (rank %d has %d): use fewer "
                        "ranks, more z-points or z_scheme = explicit",
                        pz, zc_nmin, r, zst[size_t(r + 1)] - zst[size_t(r)]);
          fatal(msg);
        }
    auto ends = [&](int n, double al, double& p0, double& qn1, std::vector<double>* pv, std::vector<double>* qv) {
      TriLocal T;
      T.init(n, al);
      std::vector<double> e(static_cast<size_t>(n), 0.0);
      e[0] = -al;
      std::vector<double> p = T.solve(e, al);
      e[0] = 0; e[size_t(n - 1)] = -al;
      std::vector<double> q = T.solve(e, al);
      p0 = p[0]; qn1 = q[size_t(n - 1)];
      if (pv) *pv = p;
      if (qv) *qv = q;
    };
    auto tor = [](const std::vector<double>& x) { return std::vector<real>(x.begin(), x.end()); };
    const int prv = (myrank - 1 + nrank) % nrank, nxt = (myrank + 1) % nrank;
    for (int d = 0; d < 2; ++d) {
      const double al = zc_al[d], scale = (d == 0) ? 1 / hz : 1 / (hz * hz);
      ZCmp& Z = zc[d];
      Z.L = L;
      Z.al = real(al);
      for (int l = 0; l < MAXP2; ++l) Z.c[l] = real(l < L ? c[d][size_t(l)] * scale : 0.0);
      TriLocal T;
      T.init(nz, al);
      std::vector<double> p, q;
      double p0o, qn1o, p0n, qn1n, p0p, qn1p;
      ends(nz, al, p0o, qn1o, &p, &q);
      ends(zst[size_t(nxt + 1)] - zst[size_t(nxt)], al, p0n, qn1n, nullptr, nullptr);
      ends(zst[size_t(prv + 1)] - zst[size_t(prv)], al, p0p, qn1p, nullptr, nullptr);
      d_zcp[d].alloc(nz); d_zcp[d].upload(tor(T.cp));
      d_zinv[d].alloc(nz); d_zinv[d].upload(tor(T.inv));
      d_zpv[d].alloc(nz); d_zpv[d].upload(tor(p));
      d_zqv[d].alloc(nz); d_zqv[d].upload(tor(q));
      Z.cp = d_zcp[d].p; Z.inv = d_zinv[d].p; Z.pv = d_zpv[d].p; Z.qv = d_zqv[d].p;
      Z.p0o = real(p0o); Z.qn1o = real(qn1o); Z.p0n = real(p0n); Z.qn1p = real(qn1p);
      Z.il = real(1 / (1 - p0o * qn1p));  // boundary with the previous rank
      Z.ir = real(1 / (1 - p0n * qn1o));  // boundary with the next rank
      // one rank: x_{-1} = x_{n-1}, x_n = x_0 exactly: [1 - p_{n-1}, -q_{n-1}; -p_0, 1 - q_0] (x_{n-1}, x_0) = (y_{n-1}, y_0)
      const double m11 = 1 - p[size_t(nz - 1)], m12 = -q[size_t(nz - 1)], m21 = -p[0], m22 = 1 - q[0];
      const double det = m11 * m22 - m12 * m21;
      Z.e11 = real(m22 / det); Z.e12 = real(-m12 / det); Z.e21 = real(-m21 / det); Z.e22 = real(m11 / det);
    }
    // modified wavenumbers: k' h = 2 sum c_l sin(l th) / (1 + 2 al cos th), k'' h^2 = 2 sum c_l (1 - cos(l th)) / (...)
    auto k1 = [&](double th) {
      double s = 0;
      for (int l = 1; l <= L; ++l) s += 2 * c[0][size_t(l - 1)] * std::sin(l * th);
      return s / (1 + 2 * zc_al[0] * std::cos(th));
    };
    auto k2 = [&](double th) {
      double s = 0;
      for (int l = 1; l <= L; ++l) s += 2 * c[1][size_t(l - 1)] * (1 - std::cos(l * th));
      return s / (1 + 2 * zc_al[1] * std::cos(th));
    };
    kap1 = 0; lam2z = 0;
    for (int t = 0; t < ntz; ++t) {
      const double th = 2 * M_PI * t / nzg;
      const double s1 = (2 * t == nzg) ? 0.0 : k1(th);  // exactly zero at the Nyquist wavenumber
      kzp2[size_t(t)] = (s1 / hz) * (s1 / hz);
      lam2z = std::max(lam2z, k2(th) / (hz * hz));
    }
    for (int t = 0; t <= 2000; ++t) kap1 = std::max(kap1, k1(M_PI * t / 2000));
  }

  // d-th z-derivative (0: D1z, 1: D2z) of f (line length L, halos in hal), modes i < nkx, all y; z_scheme = compact
  // only. One pass over the lines (right-hand side, Thomas sweep: the block solution y in out), one exchange of a
  // plane with each neighbour, and per line x_{-1}, x_n in xb (2*ny*nkx). The kernels read the derivative through
  // the returned ZDer, which forms x = y + p x_{-1} + q x_n as it is read (no pass to write x).
  ZDer zderiv(int d, const cplx* f, idx L, const cplx* hal, cplx* out, cplx* xb) {
    const ZCmp Z = zc[d];
    const int nz_ = nz, ny_ = ny, nkx_ = nkx, h = hz_halo;
    const idx nlines = idx(ny_) * nkx_, lstride = nkx_;
    cplx *sfirst = d_zx.p, *slast = d_zx.p + nlines, *rprev = d_zx.p + 2 * nlines, *rnext = d_zx.p + 3 * nlines;
    const bool D2 = (d == 1);
    // 1: right-hand side and the local Thomas solve y, one thread per (kx, y) line
    pfor_lines(nlines, [=] HD(idx ln) {
      const int i = int(ln % nkx_), j = int(ln / nkx_);
      auto F = [&](int kk) { return zv(f, hal, L, i, kk, j, nz_, h, ny_, nkx_); };
      cplx* o = out + idx(j) * nz_ * lstride + i;  // o[k * lstride]
      cplx prev(0, 0);
      for (int k = 0; k < nz_; ++k) {
        cplx r(0, 0);
        if (D2) {
          const cplx f0 = F(k);
          for (int l = 1; l <= Z.L; ++l) r += Z.c[l - 1] * (F(k + l) + F(k - l) - real(2) * f0);
        } else {
          for (int l = 1; l <= Z.L; ++l) r += Z.c[l - 1] * (F(k + l) - F(k - l));
        }
        prev = Z.inv[k] * (k == 0 ? r : r - Z.al * prev);
        o[idx(k) * lstride] = prev;
      }
      cplx y = prev;
      slast[ln] = y;
      for (int k = nz_ - 2; k >= 0; --k) {
        y = o[idx(k) * lstride] - Z.cp[k] * y;
        o[idx(k) * lstride] = y;
      }
      sfirst[ln] = y;
    });
    // 2: y_{n-1} to the next rank, y_0 to the previous one
    const bool single = (nrank == 1);
    if (!single) {
      g_timer.tic(tm_halo);
      const int up = (myrank + 1) % nrank, dn = (myrank - 1 + nrank) % nrank;
      g_comm.sendrecv2(reinterpret_cast<const real*>(slast), reinterpret_cast<real*>(rprev), up, dn,
                       reinterpret_cast<const real*>(sfirst), reinterpret_cast<real*>(rnext), dn, up, 2 * nlines);
      g_timer.toc(tm_halo);
    }
    // 3: x_{-1}, x_n from the 2x2 systems at the block boundaries (x = y + x_{-1} p + x_n q where it is read)
    pfor(nlines, [=] HD(idx ln) {
      cplx xm1, xn;
      if (single) {
        const cplx yl = slast[ln], yf = sfirst[ln];
        xm1 = Z.e11 * yl + Z.e12 * yf;  // x_{n-1}
        xn = Z.e21 * yl + Z.e22 * yf;   // x_0
      } else {
        const cplx ylp = rprev[ln], yfn = rnext[ln], y0 = sfirst[ln], yl = slast[ln];
        const cplx x0 = Z.il * (y0 + Z.p0o * ylp);  // first value of this block
        xm1 = ylp + Z.qn1p * x0;                     // last value of the previous block
        xn = Z.ir * (yfn + Z.p0n * yl);              // first value of the next block
      }
      xb[ln] = xm1;
      xb[nlines + ln] = xn;
    });
    ZDer r;
    r.y = out; r.xb = xb; r.pv = Z.pv; r.qv = Z.qv; r.nz = nz_; r.nkx = nkx_; r.nlines = nlines;
    return r;
  }
  // the same, written out in final form into out (tests)
  void zderiv_full(int d, const cplx* f, idx L, const cplx* hal, cplx* out, cplx* xb) {
    const ZDer r = zderiv(d, f, L, hal, out, xb);
    pfor(nsp, [=] HD(idx n) {
      const int i = int(n % r.nkx);
      const idx kj = n / r.nkx;
      const int k = int(kj % r.nz), j = int(kj / r.nz);
      out[n] = r.at(j, k, i);
    });
  }

  void setup_operators() {
    const int p = P.fd_order;
    const bool cmp = (P.y_scheme == "compact");
    y = make_ygrid(ny, P.grid_type, P.grid_alpha);
    q = make_quadrature(y, p);
    D1 = cmp ? make_compact(y, 1, p, P.y_closure) : make_explicit(y, 1, p);
    D2 = cmp ? make_compact(y, 2, p, P.y_closure) : make_explicit(y, 2, p);
    W1 = D1.B.W; W2 = D2.B.W;
    if (W2 > MAXW || W1 > MAXW) fatal("stencil too wide");
    qint = 0;
    for (int j = 1; j < ny - 1; ++j) qint += q[size_t(j)];
    dyc.resize(size_t(ny));
    for (int j = 0; j < ny; ++j) {
      const int a = std::max(j - 1, 0), b = std::min(j + 1, ny - 1);
      dyc[size_t(j)] = (y[size_t(b)] - y[size_t(a)]) / (b - a);
    }
    auto tor = [](const std::vector<double>& x) { return std::vector<real>(x.begin(), x.end()); };
    for (int d = 0; d < 2; ++d) {
      const YDeriv& D = d == 0 ? D1 : D2;
      d_ys[d].alloc(ny); d_ys[d].upload(D.B.s);
      d_yc[d].alloc(idx(D.B.c.size())); d_yc[d].upload(tor(D.B.c));
      d_ylo[d].alloc(ny); d_ylo[d].upload(tor(D.lo));
      d_ym[d].alloc(ny); d_ym[d].upload(tor(D.m));
      d_yup[d].alloc(ny); d_yup[d].upload(tor(D.up));
      d_yid[d].alloc(ny); d_yid[d].upload(tor(D.id));
    }
    // max modified wavenumber k'h of the interior y scheme (on a uniform grid), for the CFL condition
    {
      std::vector<double> yu(41);
      for (int j = 0; j < 41; ++j) yu[size_t(j)] = -1.0 + j * 0.05;
      YDeriv Du = cmp ? make_compact(yu, 1, p, P.y_closure) : make_explicit(yu, 1, p);
      kap1y = 0;
      for (int t = 0; t <= 2000; ++t) {
        const double th = M_PI * t / 2000;
        double sn = 0;
        for (int mm = 0; mm < Du.B.W; ++mm) sn += Du.B.c[size_t(20 * Du.B.W + mm)] * 0.05 * std::sin((Du.B.s[20] + mm - 20) * th);
        const double den = 1 + (Du.lo[20] + Du.up[20]) * std::cos(th);
        kap1y = std::max(kap1y, std::abs(sn / den));
      }
    }
    d_q.alloc(ny); d_q.upload(tor(q));
    d_dyc.alloc(ny); d_dyc.upload(tor(dyc));
    // z stencils and modified wavenumbers
    const int pz = P.fd_order_z;
    std::vector<double> kzp2(static_cast<size_t>(ntz));
    if (P.z_scheme == "compact") {
      setup_compact_z(kzp2);
    } else {
    std::vector<double> a1 = central_weights(1, pz), a2 = central_weights(2, pz);
    zs.h = pz / 2;
    for (int l = 0; l <= pz; ++l) {
      zs.a1[l] = real(a1[size_t(l)] / hz);
      zs.a2[l] = real(a2[size_t(l)] / (hz * hz));
    }
    kap1 = 0; lam2z = 0;
    for (int t = 0; t < ntz; ++t) {
      const double th = 2 * M_PI * t / nzg;
      double s1 = 0, s2 = a2[size_t(pz / 2)];
      for (int l = 1; l <= pz / 2; ++l) {
        s1 += 2 * a1[size_t(pz / 2 + l)] * std::sin(l * th);
        s2 += 2 * a2[size_t(pz / 2 + l)] * std::cos(l * th);
      }
      if (2 * t == nzg) s1 = 0;  // exactly zero at the Nyquist wavenumber
      kzp2[size_t(t)] = (s1 / hz) * (s1 / hz);
      lam2z = std::max(lam2z, -s2 / (hz * hz));
    }
    for (int t = 0; t <= 2000; ++t) {  // max k'h of the central D1 (for the CFL condition)
      double s1 = 0;
      for (int l = 1; l <= pz / 2; ++l) s1 += 2 * a1[size_t(pz / 2 + l)] * std::sin(l * M_PI * t / 2000);
      kap1 = std::max(kap1, s1);
    }
    }
    d_kzp2.alloc(ntz); d_kzp2.upload(tor(kzp2));
    kx_h.resize(size_t(nkx));
    for (int i = 0; i < nkx; ++i) kx_h[size_t(i)] = 2 * M_PI * i / P.lx;
    d_kx.alloc(nkx); d_kx.upload(tor(kx_h));
    // pressure system for (phi, D1 phi), padded to K sub/super diagonals
    BandMat G0, G1;
    make_pressure_system(D1, G0, G1);
    const int kb = std::max(G0.kl, G0.ku);
    for (int k : {4, 6, 8, 10, 12, 14, 16})
      if (kb <= k) { K = k; break; }
    if (K == 0) fatal("pressure band too wide");
    std::vector<real> g0(size_t(2 * ny) * (2 * K + 1), real(0)), g1 = g0;
    for (int r = 0; r < 2 * ny; ++r)
      for (int d = 0; d < G0.width(); ++d) {
        g0[size_t(r) * (2 * K + 1) + d - G0.kl + K] = real(G0.a[size_t(r * G0.width() + d)]);
        g1[size_t(r) * (2 * K + 1) + d - G0.kl + K] = real(G1.a[size_t(r * G1.width() + d)]);
      }
    d_G0.alloc(idx(g0.size())); d_G0.upload(g0);
    d_G1.alloc(idx(g1.size())); d_G1.upload(g1);
    // spectral radius of D2 (interior, Dirichlet walls) by power iteration
    std::vector<double> f(size_t(ny), 0.0);
    std::mt19937 g(7);
    for (int j = 1; j < ny - 1; ++j) f[size_t(j)] = std::uniform_real_distribution<double>(-1, 1)(g);
    for (int it = 0; it < 400; ++it) {
      std::vector<double> h = D2.apply(f);
      h[0] = h[size_t(ny - 1)] = 0;
      double nrm = 0;
      for (double x : h) nrm = std::max(nrm, std::abs(x));
      for (int j = 0; j < ny; ++j) f[size_t(j)] = h[size_t(j)] / nrm;
      lam2y = nrm;
    }
  }

  double kxmax() const { return 2 * M_PI * (nkx - 1) / P.lx; }

  // ---------------------------------------------------------------- communication
  // all-to-all layouts (complex numbers; MPI counts in reals):
  //   z-slab -> kx-slab, send block for rank r: ((j*nk_r + i')*nz + k), i' = i - kxst[r]
  //   receive block from rank s:               ((j*nkl + i')*nz_s + k), k local to s
  void setup_transpose() {
    a2a_sc.assign(size_t(nrank), 0); a2a_sd = a2a_sc; a2a_rc = a2a_sc; a2a_rd = a2a_sc;
    std::vector<idx> sd(static_cast<size_t>(nrank)), rd(static_cast<size_t>(nrank));
    std::vector<int> cnt(static_cast<size_t>(nrank)), nzr(static_cast<size_t>(nrank));
    idx so = 0, ro = 0;
    for (int r = 0; r < nrank; ++r) {
      cnt[size_t(r)] = kxst[size_t(r + 1)] - kxst[size_t(r)];
      nzr[size_t(r)] = zst[size_t(r + 1)] - zst[size_t(r)];
      const idx ns = idx(ny) * cnt[size_t(r)] * nz, nr = idx(ny) * nkl * nzr[size_t(r)];
      sd[size_t(r)] = so; rd[size_t(r)] = ro;
      a2a_sc[size_t(r)] = int(2 * ns); a2a_sd[size_t(r)] = int(2 * so);
      a2a_rc[size_t(r)] = int(2 * nr); a2a_rd[size_t(r)] = int(2 * ro);
      so += ns; ro += nr;
    }
    std::vector<int> kxown(static_cast<size_t>(nkx)), kxoff(static_cast<size_t>(nkx)), zown(static_cast<size_t>(nzg)),
        zoff(static_cast<size_t>(nzg));
    for (int r = 0; r < nrank; ++r) {
      for (int i = kxst[size_t(r)]; i < kxst[size_t(r + 1)]; ++i) { kxown[size_t(i)] = r; kxoff[size_t(i)] = i - kxst[size_t(r)]; }
      for (int k = zst[size_t(r)]; k < zst[size_t(r + 1)]; ++k) { zown[size_t(k)] = r; zoff[size_t(k)] = k - zst[size_t(r)]; }
    }
    d_kxown.alloc(nkx); d_kxown.upload(kxown);
    d_kxoff.alloc(nkx); d_kxoff.upload(kxoff);
    d_zown.alloc(nzg); d_zown.upload(zown);
    d_zoff.alloc(nzg); d_zoff.upload(zoff);
    d_sd.alloc(nrank); d_sd.upload(sd);
    d_rd.alloc(nrank); d_rd.upload(rd);
    d_cnt.alloc(nrank); d_cnt.upload(cnt);
    d_nzr.alloc(nrank); d_nzr.upload(nzr);
  }
  // (kx, z local, y) in t1 -> (z global, kx local, y) in t2
  void transpose_to_kx(const cplx* t1, cplx* t2) {
    const int nkx_ = nkx, nz_ = nz, nkl_ = nkl, nzg_ = nzg;
    const int *own = d_kxown.p, *off = d_kxoff.p, *cnt = d_cnt.p, *zown = d_zown.p, *zoff = d_zoff.p, *nzr = d_nzr.p;
    const idx *sd = d_sd.p, *rd = d_rd.p;
    cplx *sb = sbuf.p, *rb = rbuf.p;
    // per y-row a (z, kx) -> (kx, z) transpose: tiled, so that reads and writes are both coalesced
    g_timer.tic(tm_pack);
    ptranspose(ny, nz_, nkx_,
        [=] HD(idx j, int k, int i) -> cplx { return t1[(j * nz_ + k) * nkx_ + i]; },
        [=] HD(idx j, int k, int i, cplx val) { const int r = own[i]; sb[sd[r] + (j * cnt[r] + off[i]) * nz_ + k] = val; });
    g_timer.toc(tm_pack);
    g_timer.tic(tm_a2a);
    g_comm.alltoallv(reinterpret_cast<real*>(sbuf.p), a2a_sc, a2a_sd, reinterpret_cast<real*>(rbuf.p), a2a_rc, a2a_rd);
    g_timer.toc(tm_a2a);
    g_timer.tic(tm_pack);
    pfor(idx(ny) * nkl_ * nzg_, [=] HD(idx m) {
      const int kg = int(m % nzg_);
      const idx r = m / nzg_;
      const int s = zown[kg];
      t2[m] = rb[rd[s] + r * nzr[s] + zoff[kg]];
    });
    g_timer.toc(tm_pack);
  }
  // inverse of transpose_to_kx
  void transpose_to_z(const cplx* t2, cplx* t1) {
    const int nkx_ = nkx, nz_ = nz, nkl_ = nkl, nzg_ = nzg;
    const int *own = d_kxown.p, *off = d_kxoff.p, *cnt = d_cnt.p, *zown = d_zown.p, *zoff = d_zoff.p, *nzr = d_nzr.p;
    const idx *sd = d_sd.p, *rd = d_rd.p;
    cplx *sb = sbuf.p, *rb = rbuf.p;
    g_timer.tic(tm_pack);
    pfor(idx(ny) * nkl_ * nzg_, [=] HD(idx m) {
      const int kg = int(m % nzg_);
      const idx r = m / nzg_;
      const int s = zown[kg];
      rb[rd[s] + r * nzr[s] + zoff[kg]] = t2[m];
    });
    g_timer.toc(tm_pack);
    g_timer.tic(tm_a2a);
    g_comm.alltoallv(reinterpret_cast<real*>(rbuf.p), a2a_rc, a2a_rd, reinterpret_cast<real*>(sbuf.p), a2a_sc, a2a_sd);
    g_timer.toc(tm_a2a);
    g_timer.tic(tm_pack);
    ptranspose(ny, nkx_, nz_,
        [=] HD(idx j, int i, int k) -> cplx { const int r = own[i]; return sb[sd[r] + (j * cnt[r] + off[i]) * nz_ + k]; },
        [=] HD(idx j, int i, int k, cplx val) { t1[(j * nz_ + k) * nkx_ + i] = val; });
    g_timer.toc(tm_pack);
  }
  // fill the halo buffer of f (line length L, first W entries of each line) from the neighbouring ranks
  void halo(const cplx* f, idx L, cplx* hal) {
    const int h = hz_halo, nz_ = nz, W = nkx;
    const idx part = idx(ny) * h * W;
    cplx* sb = hsend.p;
    // sb[0..part): my last h planes (to the upper neighbour), sb[part..): my first h planes (to the lower one)
    pfor(2 * part, [=] HD(idx n) {
      const int side = int(n / part);
      const idx m = n - side * part;
      const int i = int(m % W);
      const idx r = m / W;
      const int q = int(r % h);
      const idx j = r / h;
      int k = side == 0 ? nz_ - h + q : q;
      k %= nz_; if (k < 0) k += nz_;
      sb[n] = f[(j * nz_ + k) * L + i];
    });
    const int up = (myrank + 1) % nrank, dn = (myrank - 1 + nrank) % nrank;
    g_timer.tic(tm_halo);
    // my upper planes -> upper neighbour's lower halo, receive my lower halo from the lower neighbour; and the
    // other way round; both in one exchange
    g_comm.sendrecv2(reinterpret_cast<const real*>(sb), reinterpret_cast<real*>(hal), up, dn,
                     reinterpret_cast<const real*>(sb + part), reinterpret_cast<real*>(hal + part), dn, up, 2 * part);
    g_timer.toc(tm_halo);
  }
  // the halos of three fields (line length nkx) in one exchange: hsend holds [side][field][planes]
  void halo3(const cplx* f0, const cplx* f1, const cplx* f2, cplx* h0, cplx* h1, cplx* h2) {
    const int h = hz_halo, nz_ = nz, W = nkx;
    const idx part = idx(ny) * h * W, L = nkx;
    cplx *sb = hsend.p, *rv = hrecv.p;
    pfor(6 * part, [=] HD(idx n) {
      const int side = int(n / (3 * part));
      const idx rem = n - side * 3 * part;
      const int fl = int(rem / part);
      const idx m = rem - fl * part;
      const int i = int(m % W);
      const idx r = m / W;
      const int q = int(r % h);
      const idx j = r / h;
      int k = side == 0 ? nz_ - h + q : q;
      k %= nz_; if (k < 0) k += nz_;
      const cplx* f = fl == 0 ? f0 : fl == 1 ? f1 : f2;
      sb[n] = f[(j * nz_ + k) * L + i];
    });
    const int up = (myrank + 1) % nrank, dn = (myrank - 1 + nrank) % nrank;
    g_timer.tic(tm_halo);
    g_comm.sendrecv2(reinterpret_cast<const real*>(sb), reinterpret_cast<real*>(rv), up, dn,
                     reinterpret_cast<const real*>(sb + 3 * part), reinterpret_cast<real*>(rv + 3 * part), dn, up, 6 * part);
    g_timer.toc(tm_halo);
    pfor(6 * part, [=] HD(idx n) {
      const int side = int(n / (3 * part));
      const idx rem = n - side * 3 * part;
      const int fl = int(rem / part);
      const idx m = rem - fl * part;
      cplx* hal = fl == 0 ? h0 : fl == 1 ? h1 : h2;
      hal[side * part + m] = rv[n];
    });
  }

  void print_setup() const {
    std::printf("modes     nkx = %d (padded x grid %d), physical z points %d, y points %d\n", nkx, mx, nzg, ny);
    std::printf("parallel  %d rank(s), z-slabs of %d..%d planes, %d..%d x-modes per rank in the pressure solve\n", nrank,
                zst[1] - zst[0], zst[size_t(nrank)] - zst[size_t(nrank - 1)], kxst[1] - kxst[0],
                kxst[size_t(nrank)] - kxst[size_t(nrank - 1)]);
    std::printf("y grid    dy_wall = %.3e, dy_centre = %.3e\n", y[1] - y[0], y[size_t(ny / 2 + 1)] - y[size_t(ny / 2)]);
    std::printf("y scheme  %s, order %d; z scheme %s, order %d\n", P.y_scheme.c_str(), P.fd_order, P.z_scheme.c_str(),
                P.fd_order_z);
    if (zcomp)
      std::printf("z compact alpha D1z %.6f, D2z %.6f; block coupling decays as %.3f^n, needs >= %d z-planes per rank "
                  "(here %d..%d)\n", zc_al[0], zc_al[1], std::max(zc_r[0], zc_r[1]), zc_nmin, zst[1] - zst[0],
                  zst[size_t(nrank)] - zst[size_t(nrank - 1)]);
    std::printf("pressure  band K = %d, %lld columns of length %d (phi and D1 phi)\n", K, ncols, 2 * ny);
    std::printf("viscous   spectral radii: kx^2 %.3e, D2z %.3e, D2y %.3e\n", kxmax() * kxmax(), lam2z, lam2y);
    const double mb = double(sizeof(real)) / 1048576.0;
    int npb = 0;
    for (const auto& b : Pb) npb += (b.n > 0);
    const double tot = (double(npb) * npad + 2.0 * (11 + (zcomp ? 2 : 0) + 3 * nsc) * nsp + (3.0 * K + 1.0 + NRHS) * 2 * ny * ncols) * mb +
                       2.0 * ny * ncols / 1048576.0;
    if (nsc > 0)
      std::printf("scalars   %d, %s form; implicit y-systems: band %d+%d, Neumann rows of D1 with %d/%d entries\n", nsc,
                  P.scal_form.c_str(), skl, sku, nc1, ncn);
    if (ovs || pch > 1) {
      double ex = 0;  // extra arrays of the overlap options
      if (ovs) ex += (2.0 * nsp + (zcomp ? 2.0 * nsp : 0.0)) * 2 * mb;
      std::printf("overlap   %s%s%s; +%.1f MB for the scalar arrays (plus the z-FFT plans of the chunks)\n",
                  ovs ? "scalars on their own stream" : "", ovs && pch > 1 ? ", " : "",
                  pch > 1 ? ("pressure step in " + std::to_string(pch) + " chunks").c_str() : "", ex);
    } else if (P.p_chunks > 1) {
      std::printf("overlap   p_chunks = %d: not used (fewer kx per rank than chunks would need; at least 2)\n", P.p_chunks);
    }
    std::printf("memory    %.1f MB of field arrays on rank 0 (%.0f bytes per local grid point)\n", tot,
                tot * 1048576.0 / (double(nx) * ny * nz));
  }

  // ---------------------------------------------------------------- scalars (as HiCHAN)
  // Transport  ds/dt + u.grad s = kappa Lap s + Q_c + Q_u u/ubulk,  kappa = nu/Pr, with Dirichlet (s = g) or Neumann
  // (ds/dy = g) walls (g only for the x-mean, kx = 0). Low-storage RK as the velocity; the diffusion in x and y with
  // the theta-method (scal_theta; HiCHAN: Eqs. (9)-(13), B.2), the diffusion in z explicit (fdchan is not Fourier in z;
  // its explicit z-diffusion limit is far above the y-limit). The implicit y-solve is HiCHAN's scalar_solve:
  // premultiplied by A2, H = (1 + c kx^2) A2 - c B2 with identity wall rows, and the unknown wall values tau_1, tau_N
  // of the operator (and the wall values of s for Neumann walls) from a small influence-matrix system. H depends on
  // kx only: its LU, the homogeneous solutions and the inverse influence matrix are computed once per kx and stage.
  void setup_scalar_solve() {
    // band of A2 (tridiagonal, unit diagonal) and B2 per row, offsets o = col - row in [-skl, sku]
    skl = 1; sku = 1;
    for (int j = 0; j < ny; ++j) {
      skl = std::max(skl, j - D2.B.s[size_t(j)]);
      sku = std::max(sku, D2.B.s[size_t(j)] + D2.B.W - 1 - j);
    }
    snb = skl + sku + 1;
    if (snb > MAXB) fatal("scalar y-system: band too wide");
    std::vector<double> a2b(size_t(ny) * snb, 0.0), b2b(size_t(ny) * snb, 0.0);
    for (int j = 0; j < ny; ++j) {
      a2b[size_t(j * snb + skl)] = 1.0;
      if (D2.compact) {
        if (j > 0) a2b[size_t(j * snb + skl - 1)] = D2.lo[size_t(j)];
        if (j < ny - 1) a2b[size_t(j * snb + skl + 1)] = D2.up[size_t(j)];
      }
      for (int m = 0; m < D2.B.W; ++m) {
        const int col = D2.B.s[size_t(j)] + m;
        b2b[size_t(j * snb + col - j + skl)] += D2.B.c[size_t(j * D2.B.W + m)];
      }
    }
    auto tor = [](const std::vector<double>& x) { return std::vector<real>(x.begin(), x.end()); };
    d_a2b.alloc(idx(a2b.size())); d_a2b.upload(tor(a2b));
    d_b2b.alloc(idx(b2b.size())); d_b2b.upload(tor(b2b));
    // first and last rows of D1 (dense for compact schemes): the Neumann conditions and the wall gradients
    std::vector<double> c1(static_cast<size_t>(ny)), cn(static_cast<size_t>(ny)), e(static_cast<size_t>(ny), 0.0);
    for (int r = 0; r < ny; ++r) {
      e[size_t(r)] = 1;
      std::vector<double> d = D1.apply(e);
      c1[size_t(r)] = d[0]; cn[size_t(r)] = d[size_t(ny - 1)];
      e[size_t(r)] = 0;
    }
    double m1 = 0, mn = 0;
    for (int r = 0; r < ny; ++r) { m1 = std::max(m1, std::abs(c1[size_t(r)])); mn = std::max(mn, std::abs(cn[size_t(r)])); }
    nc1 = 1; ncn = 1;
    for (int r = 0; r < ny; ++r) {
      if (std::abs(c1[size_t(r)]) > 1e-15 * m1) nc1 = r + 1;
      if (std::abs(cn[size_t(ny - 1 - r)]) > 1e-15 * mn) ncn = r + 1;
    }
    d_c1.alloc(ny); d_c1.upload(tor(c1));
    d_cn.alloc(ny); d_cn.upload(tor(cn));
    d_sU.alloc(idx(nkx) * ny * (sku + 1));
    d_sL.alloc(idx(nkx) * ny * std::max(skl, 1));
    d_sX.alloc(idx(nkx) * 4 * ny);
    d_sE.alloc(idx(nkx) * 16);
  }

  // per kx: LU of H = (1 + c kx^2) A2 - c B2 (wall rows identity), the homogeneous solutions X_m (m = 0: tau_1,
  // 1: tau_N, 2: s(-1) = 1, 3: s(+1) = 1) and the inverse of the influence matrix (2x2 Dirichlet, else 4x4)
  void scalar_factor(double c, bool isnb, bool isnt) {
    const int ny_ = ny, nkx_ = nkx, kl = skl, ku = sku, nb = snb, nc1_ = nc1, ncn_ = ncn;
    const real *a2b = d_a2b.p, *b2b = d_b2b.p, *kxv = d_kx.p, *c1 = d_c1.p, *cn = d_cn.p;
    real *U = d_sU.p, *Lm = d_sL.p, *X = d_sX.p, *E = d_sE.p;
    const real cr = real(c);
    const bool isd = !(isnb || isnt);
    pfor_lines(nkx_, [=] HD(idx ii) {
      const int i = int(ii);
      const real ca = real(1) + cr * kxv[i] * kxv[i], cb = -cr;
      auto Ui = [&](int r, int q) -> real& { return U[(idx(r) * (ku + 1) + q) * nkx_ + i]; };
      auto Li = [&](int r, int q) -> real& { return Lm[(idx(r) * (kl > 0 ? kl : 1) + (q - 1)) * nkx_ + i]; };
      auto Xi = [&](int m, int r) -> real& { return X[(idx(m) * ny_ + r) * nkx_ + i]; };
      auto hrow = [&](int r, int o) { return ca * a2b[r * nb + o + kl] + cb * b2b[r * nb + o + kl]; };
      real row[MAXB];
      for (int r = 0; r < ny_; ++r) {
        for (int o = -kl; o <= ku; ++o) row[o + kl] = (r == 0 || r == ny_ - 1) ? (o == 0 ? real(1) : real(0)) : hrow(r, o);
        for (int l = (r - kl > 0 ? r - kl : 0); l < r; ++l) {
          const int o = l - r;
          real f = 0;
          if (row[o + kl] != real(0)) {
            f = row[o + kl] / Ui(l, 0);
            for (int q = 1; q <= ku; ++q)
              if (o + q <= ku) row[o + q + kl] -= f * Ui(l, q);
          }
          Li(r, r - l) = f;
        }
        for (int q = 0; q <= ku; ++q) Ui(r, q) = row[q + kl];
      }
      real e[4][5] = {};
      for (int m = 0; m < 4; ++m) {
        for (int r = 0; r < ny_; ++r) Xi(m, r) = 0;
        if (m == 0) Xi(m, 1) = a2b[1 * nb - 1 + kl];
        if (m == 1) Xi(m, ny_ - 2) = a2b[(ny_ - 2) * nb + 1 + kl];
        if (m == 2) Xi(m, 0) = 1;
        if (m == 3) Xi(m, ny_ - 1) = 1;
        for (int r = 0; r < ny_; ++r) {
          real y = Xi(m, r);
          for (int q = 1; q <= kl && q <= r; ++q) y -= Li(r, q) * Xi(m, r - q);
          Xi(m, r) = y;
        }
        for (int r = ny_ - 1; r >= 0; --r) {
          real x = Xi(m, r);
          for (int q = 1; q <= ku && r + q < ny_; ++q) x -= Ui(r, q) * Xi(m, r + q);
          Xi(m, r) = x / Ui(r, 0);
        }
        real h0 = 0, hn = 0;
        for (int o = 0; o <= ku && o < ny_; ++o) h0 += hrow(0, o) * Xi(m, o);
        for (int o = -kl; o <= 0; ++o)
          if (ny_ - 1 + o >= 0) hn += hrow(ny_ - 1, o) * Xi(m, ny_ - 1 + o);
        e[0][m + 1] = h0 - (m == 0 ? real(1) : real(0));
        e[1][m + 1] = hn - (m == 1 ? real(1) : real(0));
        real f3 = 0, f4 = 0;
        for (int r = 0; r < nc1_; ++r) f3 += c1[r] * Xi(m, r);
        for (int r = ny_ - ncn_; r < ny_; ++r) f4 += cn[r] * Xi(m, r);
        e[2][m + 1] = isnb ? f3 : (m == 2 ? real(1) : real(0));
        e[3][m + 1] = isnt ? f4 : (m == 3 ? real(1) : real(0));
      }
      // inverse of the influence matrix (columns m = 0..3), stored row-major in E[q * 4 + l]
      real inv[4][4];
      for (int q = 0; q < 4; ++q)
        for (int l = 0; l < 4; ++l) inv[q][l] = 0;
      if (isd) {
        const real a = e[0][1], b = e[0][2], cc = e[1][1], d = e[1][2], det = a * d - b * cc;
        inv[0][0] = d / det; inv[0][1] = -b / det; inv[1][0] = -cc / det; inv[1][1] = a / det;
      } else {
        real Mx[4][8];
        for (int q = 0; q < 4; ++q)
          for (int l = 0; l < 4; ++l) { Mx[q][l] = e[q][l + 1]; Mx[q][l + 4] = (q == l) ? real(1) : real(0); }
        for (int q = 0; q < 4; ++q) {
          int p = q;
          for (int l = q + 1; l < 4; ++l)
            if ((Mx[l][q] < 0 ? -Mx[l][q] : Mx[l][q]) > (Mx[p][q] < 0 ? -Mx[p][q] : Mx[p][q])) p = l;
          if (p != q)
            for (int l = 0; l < 8; ++l) { const real t = Mx[q][l]; Mx[q][l] = Mx[p][l]; Mx[p][l] = t; }
          const real piv = Mx[q][q];
          for (int l = 0; l < 8; ++l) Mx[q][l] /= piv;
          for (int r = 0; r < 4; ++r)
            if (r != q) {
              const real f = Mx[r][q];
              for (int l = 0; l < 8; ++l) Mx[r][l] -= f * Mx[q][l];
            }
        }
        for (int q = 0; q < 4; ++q)
          for (int l = 0; l < 4; ++l) inv[q][l] = Mx[q][l + 4];
      }
      for (int q = 0; q < 4; ++q)
        for (int l = 0; l < 4; ++l) E[(idx(q) * 4 + l) * nkx_ + i] = inv[q][l];
    });
  }

  // the implicit solve of every (kx, z) line of f: on input r, on output s; gb, gt the wall values (Dirichlet) or wall
  // gradients (Neumann) of the x-mean
  void scalar_lines(cplx* f, double c, bool isnb, bool isnt, double gb_, double gt_) {
    const int ny_ = ny, nkx_ = nkx, nz_ = nz, kl = skl, ku = sku, nb = snb, nc1_ = nc1, ncn_ = ncn;
    const real *a2b = d_a2b.p, *b2b = d_b2b.p, *kxv = d_kx.p, *c1 = d_c1.p, *cn = d_cn.p;
    const real *U = d_sU.p, *Lm = d_sL.p, *X = d_sX.p, *E = d_sE.p;
    const real cr = real(c), gb = real(gb_), gt = real(gt_);
    const bool isd = !(isnb || isnt);
    const idx st = idx(nz_) * nkx_;
    pfor_lines(idx(nz_) * nkx_, [=] HD(idx ln) {
      const int i = int(ln % nkx_);
      cplx* F = f + ln;  // F[j * st]
      const real ca = real(1) + cr * kxv[i] * kxv[i], cb = -cr;
      auto hrow = [&](int r, int o) { return ca * a2b[r * nb + o + kl] + cb * b2b[r * nb + o + kl]; };
      auto Ui = [&](int r, int q) { return U[(idx(r) * (ku + 1) + q) * nkx_ + i]; };
      auto Li = [&](int r, int q) { return Lm[(idx(r) * (kl > 0 ? kl : 1) + (q - 1)) * nkx_ + i]; };
      auto Xi = [&](int m, int r) { return X[(idx(m) * ny_ + r) * nkx_ + i]; };
      const cplx zero(0, 0);
      // r~ = r with zero wall values; A2 r~ in the interior rows, the wall rows the Dirichlet values of the x-mean
      const cplx r1 = F[1 * st], rn = F[idx(ny_ - 2) * st];
      const cplx ra1 = a2b[0 * nb + 1 + kl] * r1, ran = a2b[(ny_ - 1) * nb - 1 + kl] * rn;
      cplx prev = zero, cur = zero;  // r~_{j-1}, r~_j (original values)
      for (int j = 0; j < ny_; ++j) {
        const cplx nxt = (j + 1 <= ny_ - 2) ? F[idx(j + 1) * st] : zero;
        if (j > 0 && j < ny_ - 1) cur = F[idx(j) * st];
        cplx sx;
        if (j == 0) sx = (i == 0 && !isnb) ? cplx(gb, 0) : zero;
        else if (j == ny_ - 1) sx = (i == 0 && !isnt) ? cplx(gt, 0) : zero;
        else sx = a2b[j * nb - 1 + kl] * prev + cur + a2b[j * nb + 1 + kl] * nxt;
        for (int q = 1; q <= kl && q <= j; ++q) sx -= Li(j, q) * F[idx(j - q) * st];
        F[idx(j) * st] = sx;
        prev = cur;
        cur = zero;
      }
      for (int j = ny_ - 1; j >= 0; --j) {
        cplx x = F[idx(j) * st];
        for (int q = 1; q <= ku && j + q < ny_; ++q) x -= Ui(j, q) * F[idx(j + q) * st];
        F[idx(j) * st] = (real(1) / Ui(j, 0)) * x;
      }
      // the conditions of the particular solution
      cplx e1[4] = {zero, zero, zero, zero};
      for (int o = 0; o <= ku && o < ny_; ++o) e1[0] += hrow(0, o) * F[idx(o) * st];
      for (int o = -kl; o <= 0; ++o)
        if (ny_ - 1 + o >= 0) e1[1] += hrow(ny_ - 1, o) * F[idx(ny_ - 1 + o) * st];
      e1[0] -= ra1;
      e1[1] -= ran;
      if (isnb) {
        for (int r = 0; r < nc1_; ++r) e1[2] += c1[r] * F[idx(r) * st];
        if (i == 0) e1[2].re -= gb;
      }
      if (isnt) {
        for (int r = ny_ - ncn_; r < ny_; ++r) e1[3] += cn[r] * F[idx(r) * st];
        if (i == 0) e1[3].re -= gt;
      }
      cplx zz[4] = {zero, zero, zero, zero};
      const int nq = isd ? 2 : 4;
      for (int q = 0; q < nq; ++q)
        for (int l = 0; l < nq; ++l) zz[q] -= E[(idx(q) * 4 + l) * nkx_ + i] * e1[l];
      for (int j = 0; j < ny_; ++j) {
        cplx x = F[idx(j) * st];
        for (int m = 0; m < nq; ++m) x += Xi(m, j) * zz[m];
        F[idx(j) * st] = x;
      }
    });
  }

  // explicit right-hand side of the scalars (called at the end of nonlinear(): u, v, w in physical space in P0..P2):
  // NS = -fa u.grad(s) - (1 - fa) div(u s) + Q_c + Q_u u/ubulk + kappa D2z s (+ kappa (D2y - kx^2) s if theta = 0)
  void scalar_nonlinear() {
    if (nsc == 0) return;
    const int nkx_ = nkx, lp_ = lp, ld_ = ld, mx_ = mx, nz_ = nz, ny_ = ny;
    const real sc = real(1.0 / mx);
    const real* kxv = d_kx.p;
    const ZSt z = zs;
    const real fa = (P.scal_form == "divergence") ? real(0) : (P.scal_form == "skew") ? real(0.5) : real(1);
    cplx *G0 = reinterpret_cast<cplx*>(Pb[6].p), *G1 = reinterpret_cast<cplx*>(Pb[7].p), *G2 = reinterpret_cast<cplx*>(Pb[8].p);
    const real *pu = Pb[0].p, *pv = Pb[1].p, *pw = Pb[2].p;
    for (int n = 0; n < nsc; ++n) {
      const cplx* S = Sf[n].p;
      cplx* NS = NSf[n].p;
      const real kp = real(kap[n]), qc = real(P.ssrc_const[n]), qu = real(P.ssrc_u[n] / P.ubulk);
      const bool expl = (P.scal_theta[n] == 0);
      halo(S, nkx_, hbS.p);
      const cplx* hS = hbS.p;
      ZDer dzs, d2zs;
      if (zcomp) {
        dzs = zderiv(0, S, nkx_, hS, DZ1.p, ZB1.p);
        d2zs = zderiv(1, S, nkx_, hS, DZ2.p, ZB2.p);
      }
      // advective part and the sources, in physical space
      if (fa > 0) ydiff(0, S, T1.p);
      const cplx* dys = T1.p;
      pfor(idx(lp_) * nz_ * ny_, [=] HD(idx m) {
        const int i = int(m % lp_);
        const idx kj = m / lp_;
        if (i >= nkx_ || fa == real(0)) { G0[m] = cplx(0, 0); G1[m] = cplx(0, 0); G2[m] = cplx(0, 0); return; }
        const int k = int(kj % nz_), j = int(kj / nz_);
        const idx s = kj * nkx_ + i;
        cplx dz(0, 0);
        if (dzs.valid()) dz = dzs.at(j, k, i);
        else
          for (int l = -z.h; l <= z.h; ++l) {
            if (l == 0) continue;
            dz += z.a1[l + z.h] * zv(S, hS, nkx_, i, k + l, j, nz_, z.h, ny_, nkx_);
          }
        cplx dx = itimes(kxv[i], S[s]), dy = dys[s];
        if (i == 0) { dx.im = 0; dy.im = 0; dz.im = 0; }
        G0[m] = dx; G1[m] = dy; G2[m] = dz;
      });
      if (fa > 0)
        for (int c = 6; c < 9; ++c) xfft.backward(Pb[c].p);
      {
        real *gx = Pb[6].p, *gy = Pb[7].p, *gz = Pb[8].p;
        pfor(idx(mx_) * nz_ * ny_, [=] HD(idx m) {
          const idx o = (m / mx_) * ld_ + (m % mx_);
          real a = qc + qu * pu[o];
          if (fa > real(0)) a -= fa * (pu[o] * gx[o] + pv[o] * gy[o] + pw[o] * gz[o]);
          gx[o] = a;
        });
      }
      xfft.forward(Pb[6].p);
      // NS = sc * (forward FFT) + explicit diffusion
      if (expl) ydiff(1, S, T2.p);
      const cplx* d2y = T2.p;
      pfor(nsp, [=] HD(idx s) {
        const int i = int(s % nkx_);
        const idx kj = s / nkx_;
        const int k = int(kj % nz_), j = int(kj / nz_);
        cplx d2z(0, 0);
        if (d2zs.valid()) d2z = d2zs.at(j, k, i);
        else
          for (int l = -z.h; l <= z.h; ++l) d2z += z.a2[l + z.h] * zv(S, hS, nkx_, i, k + l, j, nz_, z.h, ny_, nkx_);
        cplx h = sc * G0[kj * lp_ + i] + kp * d2z;
        if (expl) h += kp * (d2y[s] - (kxv[i] * kxv[i]) * S[s]);
        if (i == 0) h.im = 0;
        NS[s] = h;
      });
      // divergence part -(1 - fa) div(u s): s in physical space, the fluxes back, i kx F_x + D1y F_y + D1z F_z
      if (fa < real(1)) {
        cplx* SP = reinterpret_cast<cplx*>(Pb[9].p);
        pfor(idx(lp_) * nz_ * ny_, [=] HD(idx m) {
          const int i = int(m % lp_);
          if (i >= nkx_) { SP[m] = cplx(0, 0); return; }
          cplx a = S[(m / lp_) * nkx_ + i];
          if (i == 0) a.im = 0;
          SP[m] = a;
        });
        xfft.backward(Pb[9].p);
        {
          const real* ps = Pb[9].p;
          real *fx = Pb[6].p, *fy = Pb[7].p, *fz = Pb[8].p;
          pfor(idx(mx_) * nz_ * ny_, [=] HD(idx m) {
            const idx o = (m / mx_) * ld_ + (m % mx_);
            fx[o] = pu[o] * ps[o]; fy[o] = pv[o] * ps[o]; fz[o] = pw[o] * ps[o];
          });
        }
        for (int c = 6; c < 9; ++c) xfft.forward(Pb[c].p);
        cplx* t1 = T1.p;
        pfor(nsp, [=] HD(idx s) { t1[s] = sc * G1[(s / nkx_) * lp_ + (s % nkx_)]; });
        ydiff(0, T1.p, T2.p);
        halo(G2, lp_, hbS.p);
        ZDer dzf;
        if (zcomp) dzf = zderiv(0, G2, lp_, hbS.p, DZ1.p, ZB1.p);
        const cplx* dyf = T2.p;
        const cplx* hF = hbS.p;
        const real cf = fa - real(1);
        pfor(nsp, [=] HD(idx s) {
          const int i = int(s % nkx_);
          const idx kj = s / nkx_;
          const int k = int(kj % nz_), j = int(kj / nz_);
          cplx dz(0, 0);
          if (dzf.valid()) dz = dzf.at(j, k, i);
          else
            for (int l = -z.h; l <= z.h; ++l) {
              if (l == 0) continue;
              dz += z.a1[l + z.h] * zv(G2, hF, lp_, i, k + l, j, nz_, z.h, ny_, nkx_);
            }
          cplx d = itimes(kxv[i], sc * G0[kj * lp_ + i]) + dyf[s] + sc * dz;
          if (i == 0) d.im = 0;
          NS[s] += cf * d;
        });
      }
    }
  }

  // r = s + dt (gam NS + zet NS_old) + (1 - theta) beta dt kappa (D2y - kx^2) s, then the theta-method solve
  void scalar_update(double dtg_, double dtz_) {
    if (nsc == 0) return;
    g_timer.tic(tm_ssolve);
    const int nkx_ = nkx;
    const real* kxv = d_kx.p;
    for (int n = 0; n < nsc; ++n) {
      const double th = P.scal_theta[n], dtb = dtg_ + dtz_;
      const real dtg = real(dtg_), dtz = real(dtz_), ce = real((1 - th) * dtb * kap[n]);
      cplx *S = Sf[n].p, *NS = NSf[n].p, *NO = NSo[n].p;
      if (th > 0) ydiff(1, S, T2.p);
      const cplx* d2y = T2.p;
      const bool imp = th > 0;
      pfor(nsp, [=] HD(idx s) {
        const int i = int(s % nkx_);
        cplx r = S[s] + dtg * NS[s] + dtz * NO[s];
        if (imp) r += ce * (d2y[s] - (kxv[i] * kxv[i]) * S[s]);
        S[s] = r;
      });
      std::swap(NSf[n].p, NSo[n].p);  // NS of this stage becomes NS_old
      const bool isnb = P.sbc_bot[n] == "N", isnt = P.sbc_top[n] == "N";
      const double c = th * dtb * kap[n];
      scalar_factor(c, isnb, isnt);
      scalar_lines(S, c, isnb, isnt, P.sval_bot[n], P.sval_top[n]);
    }
    g_timer.toc(tm_ssolve);
  }

  // ---------------------------------------------------------------- RK3 step
  void advance() {
    // u^s = u^{s-1} + dt (gam_s H^{s-1} + zet_s H^{s-2}) - (gam_s + zet_s) dt grad p
    // 3 stages: Wray's RK3 (|lambda dt| < sqrt(3) on the imaginary axis, 2.51 on the real axis);
    // 4 stages: SIMSON user guide (Chevalier et al. 2007), Table 5.1, also 3rd order (sqrt(8), 2.79)
    static const double gam3[3] = {8.0 / 15, 5.0 / 12, 3.0 / 4};
    static const double zet3[3] = {0.0, -17.0 / 60, -5.0 / 12};
    static const double gam4[4] = {8.0 / 17, 17.0 / 60, 5.0 / 12, 3.0 / 4};
    static const double zet4[4] = {0.0, -15.0 / 68, -17.0 / 60, -5.0 / 12};
    const int ns = P.rk_stages;
    const double* gam = ns == 4 ? gam4 : gam3;
    const double* zet = ns == 4 ? zet4 : zet3;
    dpdx_step = 0;
    if (P.forcing == "cfr") reset_dpdx_accumulator();
    for (int s = 0; s < ns; ++s) {
      g_timer.tic(tm_nl);
      nonlinear();
      if (!ovs) {
        g_timer.tic(tm_snl);
        scalar_nonlinear();
        g_timer.toc(tm_snl);
      }
      g_timer.toc(tm_nl);
      if (s == 0) {
        g_timer.tic(tm_dt);
        choose_dt();
        g_timer.toc(tm_dt);
      }
      g_timer.tic(tm_rhs);
      rhs(gam[s] * dt, zet[s] * dt);
      g_timer.toc(tm_rhs);
      if (ovs) {
        // the scalar terms and update on their own stream, at the same time as the pressure step: they need the
        // physical velocities of nonlinear() (P0..P2, untouched until the next stage) and the scalars that rhs()
        // has read for the buoyancy
        ev_rhs.record();
        {
          ScalarContext ctx(*this);
          ev_rhs.wait();
          scalar_nonlinear();
          scalar_update(gam[s] * dt, zet[s] * dt);
          ev_scal.record();
        }
      }
      g_timer.tic(tm_p);
      project();
      g_timer.toc(tm_p);
      if (P.forcing == "cfr") {
        g_timer.tic(tm_cfr);
        cfr_correct();
        g_timer.toc(tm_cfr);
      }
      if (!ovs) scalar_update(gam[s] * dt, zet[s] * dt);
      else ev_scal.wait();  // the next stage overwrites P0..P2 and uses the scalars
    }
    // u = r - grad phi with r = u + dt (gam H + zet H_old): phi = (gam + zet) dt p for the last stage
    pscale = 1.0 / ((gam[ns - 1] + zet[ns - 1]) * dt);
    if (P.forcing == "cfr") dpdx_step = dpdx_acc / dt;
    else dpdx_step = P.dpdx;
    time += dt;
    ++step;
  }

  YDev ydev(int d) const {
    YDev D;
    D.s = d_ys[d].p; D.c = d_yc[d].p; D.lo = d_ylo[d].p; D.m = d_ym[d].p; D.up = d_yup[d].p; D.id = d_yid[d].p;
    D.W = d == 0 ? W1 : W2;
    D.n = ny;
    D.compact = (d == 0 ? D1 : D2).compact;
    return D;
  }
  // out = d-th y derivative of in, for the spectral layout (nz*nkx lines, stride nz*nkx); in != out
  void ydiff(int d, const cplx* in, cplx* out) {
    const YDev D = ydev(d);
    const idx nl = idx(nz) * nkx;
    YLines<1> L;
    L.in[0] = in; L.tmp[0] = out;
    pfor_lines(nl, [=] HD(idx l) {
      ysweep<1>(D, L, nl, l, [&](int j, const cplx* x) { out[idx(j) * nl + l] = x[0]; });
    });
  }
  // y derivative of NF fields along line l (stride nl), for use inside a kernel: emit(j, x) receives the
  // derivatives x[0..NF) at row j as soon as they are final, so the caller can use them without storing them
  // (x3d2's fused transport kernels). Compact schemes: rows come in decreasing j, and L.tmp holds the
  // forward-sweep values afterwards (L.tmp may equal the output array of emit). Explicit: increasing j, L.tmp
  // unused. The arithmetic is that of the separate derivative followed by a pointwise kernel.
  template <int NF, class E>
  static HD void ysweep(const YDev& D, const YLines<NF>& L, idx nl, idx l, E&& emit) {
    cplx prev[NF], g[NF];
    for (int f = 0; f < NF; ++f) prev[f] = cplx(0, 0);
    for (int j = 0; j < D.n; ++j) {
      const int s = D.s[j];
      for (int f = 0; f < NF; ++f) g[f] = cplx(0, 0);
      for (int m = 0; m < D.W; ++m) {
        const real c = D.c[j * D.W + m];
        const idx o = idx(s + m) * nl + l;
        for (int f = 0; f < NF; ++f) g[f] += c * L.in[f][o];
      }
      if (D.compact) {
        if (j > 0)
          for (int f = 0; f < NF; ++f) g[f] -= D.m[j] * prev[f];
        for (int f = 0; f < NF; ++f) { L.tmp[f][idx(j) * nl + l] = g[f]; prev[f] = g[f]; }
      } else {
        emit(j, static_cast<const cplx*>(g));
      }
    }
    if (!D.compact) return;
    for (int f = 0; f < NF; ++f) g[f] = D.id[D.n - 1] * prev[f];
    emit(D.n - 1, static_cast<const cplx*>(g));
    for (int j = D.n - 2; j >= 0; --j) {
      for (int f = 0; f < NF; ++f) g[f] = D.id[j] * (L.tmp[f][idx(j) * nl + l] - D.up[j] * g[f]);
      emit(j, static_cast<const cplx*>(g));
    }
  }

  // 1-2: nonlinear term u x omega in P3..P5 (spectral, padded, unnormalised)
  void nonlinear() {
    halo3(u.p, v.p, w.p, hbA.p, hbB.p, hbC.p);  // halos of u, v, w for this stage (z-derivatives here and in rhs())
    if (P.nl_form == "advective") { nonlinear_advective(); return; }
    const int nkx_ = nkx, lp_ = lp, ld_ = ld, mx_ = mx, nz_ = nz, ny_ = ny;
    VortPoint pt;
    pt.U = u.p; pt.V = v.p; pt.W = w.p; pt.hU = hbA.p; pt.hV = hbB.p;
    pt.C0 = reinterpret_cast<cplx*>(Pb[0].p); pt.C1 = reinterpret_cast<cplx*>(Pb[1].p);
    pt.C2 = reinterpret_cast<cplx*>(Pb[2].p); pt.C3 = reinterpret_cast<cplx*>(Pb[3].p);
    pt.C4 = reinterpret_cast<cplx*>(Pb[4].p); pt.C5 = reinterpret_cast<cplx*>(Pb[5].p);
    pt.kx = d_kx.p; pt.z = zs; pt.nkx = nkx_; pt.nz = nz_; pt.ny = ny_;
    if (zcomp) {  // compact z: D1z u and D1z v first
      pt.dzU = zderiv(0, u.p, nkx, hbA.p, DZ1.p, ZB1.p);
      pt.dzV = zderiv(0, v.p, nkx, hbB.p, DZ2.p, ZB2.p);
    }
    const idx nl = idx(nz_) * nkx_;
    if (P.fuse_y) {
      // Dy u and Dy w are formed in one y-sweep and used at once: one thread per padded (x, z) line
      const YDev D = ydev(0);
      YLines<2> L;
      L.in[0] = u.p; L.in[1] = w.p; L.tmp[0] = T1.p; L.tmp[1] = T2.p;
      pfor_lines(idx(lp_) * nz_, [=] HD(idx lq) {
        const int i = int(lq % lp_), k = int(lq / lp_);
        if (i >= nkx_) {
          for (int j = 0; j < ny_; ++j) pt.zero((idx(j) * nz_ + k) * lp_ + i);
          return;
        }
        const idx ln = idx(k) * nkx_ + i;
        ysweep<2>(D, L, nl, ln, [&](int j, const cplx* dy) {
          pt((idx(j) * nz_ + k) * lp_ + i, idx(j) * nl + ln, i, k, j, dy[0], dy[1]);
        });
      });
    } else {
      ydiff(0, u.p, T1.p);
      ydiff(0, w.p, T2.p);
      const cplx *dyu = T1.p, *dyw = T2.p;
      pfor(idx(lp_) * nz_ * ny_, [=] HD(idx n) {
        const int i = int(n % lp_);
        const idx kj = n / lp_;
        if (i >= nkx_) { pt.zero(n); return; }
        const idx s = kj * nkx_ + i;
        pt(n, s, i, int(kj % nz_), int(kj / nz_), dyu[s], dyw[s]);
      });
    }
    for (int c = 0; c < 6; ++c) xfft.backward(Pb[c].p);
    const real *pu = Pb[0].p, *pv = Pb[1].p, *pw = Pb[2].p;
    real *p3 = Pb[3].p, *p4 = Pb[4].p, *p5 = Pb[5].p;
    pfor(idx(mx_) * nz_ * ny_, [=] HD(idx n) {
      const idx o = (n / mx_) * ld_ + (n % mx_);
      const real uu = pu[o], vv = pv[o], ww = pw[o];
      const real ox = p3[o], oy = p4[o], oz = p5[o];
      p3[o] = vv * oz - ww * oy;
      p4[o] = ww * ox - uu * oz;
      p5[o] = uu * oy - vv * ox;
    });
    for (int c = 3; c < 6; ++c) xfft.forward(Pb[c].p);
  }

  // advective form N_i = -(u d/dx + v d/dy + w d/dz) u_i: u, v, w and the 9 gradients in physical space
  void nonlinear_advective() {
    const int nkx_ = nkx, lp_ = lp, ld_ = ld, mx_ = mx, nz_ = nz, ny_ = ny;
    const real* kx = d_kx.p;
    const ZSt z = zs;
    const cplx* F[3] = {u.p, v.p, w.p};
    // u, v, w -> P0..P2
    {
      cplx *C0 = reinterpret_cast<cplx*>(Pb[0].p), *C1 = reinterpret_cast<cplx*>(Pb[1].p),
           *C2 = reinterpret_cast<cplx*>(Pb[2].p);
      const cplx *U = u.p, *V = v.p, *Wv = w.p;
      pfor(idx(lp_) * nz_ * ny_, [=] HD(idx n) {
        const int i = int(n % lp_);
        const idx kj = n / lp_;
        if (i < nkx_) {
          const idx s = kj * nkx_ + i;
          cplx a = U[s], b = V[s], c = Wv[s];
          if (i == 0) { a.im = 0; b.im = 0; c.im = 0; }
          C0[n] = a; C1[n] = b; C2[n] = c;
        } else {
          C0[n] = cplx(0, 0); C1[n] = cplx(0, 0); C2[n] = cplx(0, 0);
        }
      });
      for (int c = 0; c < 3; ++c) xfft.backward(Pb[c].p);
    }
    const cplx* HB[3] = {hbA.p, hbB.p, hbC.p};
    for (int comp = 0; comp < 3; ++comp) {
      const cplx* f = F[comp];
      const cplx* hf = HB[comp];
      ydiff(0, f, T1.p);
      const cplx* dyf = T1.p;
      ZDer dzf;
      if (zcomp) dzf = zderiv(0, f, nkx_, hf, DZ1.p, ZB1.p);
      cplx *G0 = reinterpret_cast<cplx*>(Pb[6].p), *G1 = reinterpret_cast<cplx*>(Pb[7].p),
           *G2 = reinterpret_cast<cplx*>(Pb[8].p);
      pfor(idx(lp_) * nz_ * ny_, [=] HD(idx n) {
        const int i = int(n % lp_);
        const idx kj = n / lp_;
        if (i < nkx_) {
          const int k = int(kj % nz_), j = int(kj / nz_);
          const idx s = kj * nkx_ + i;
          cplx dz(0, 0);
          if (dzf.valid()) dz = dzf.at(j, k, i);
          else
            for (int l = -z.h; l <= z.h; ++l) {
              if (l == 0) continue;
              dz += z.a1[l + z.h] * zv(f, hf, nkx_, i, k + l, j, nz_, z.h, ny_, nkx_);
            }
          cplx dx = itimes(kx[i], f[s]), dy = dyf[s];
          if (i == 0) { dx.im = 0; dy.im = 0; dz.im = 0; }
          G0[n] = dx; G1[n] = dy; G2[n] = dz;
        } else {
          G0[n] = cplx(0, 0); G1[n] = cplx(0, 0); G2[n] = cplx(0, 0);
        }
      });
      for (int c = 6; c < 9; ++c) xfft.backward(Pb[c].p);
      const real *pu = Pb[0].p, *pv = Pb[1].p, *pw = Pb[2].p, *gx = Pb[6].p, *gy = Pb[7].p, *gz = Pb[8].p;
      real* out = Pb[3 + comp].p;
      pfor(idx(mx_) * nz_ * ny_, [=] HD(idx n) {
        const idx o = (n / mx_) * ld_ + (n % mx_);
        out[o] = -(pu[o] * gx[o] + pv[o] * gy[o] + pw[o] * gz[o]);
      });
    }
    for (int c = 3; c < 6; ++c) xfft.forward(Pb[c].p);
  }

  // time step from the CFL conditions (physical velocities are in P0..P2 after stage 1)
  void choose_dt() {
    if (P.fixed_dt) { dt = P.dt; return; }
    const int mx_ = mx, nz_ = nz, ld_ = ld;
    const real kxm = real(M_PI * nx / P.lx), kzm = real(kap1 / hz), ky = real(kap1y);
    const real *pu = Pb[0].p, *pv = Pb[1].p, *pw = Pb[2].p, *dy = d_dyc.p;
    const double cmax = reduce_max(idx(mx_) * nz_ * ny, [=] HD(idx n) {
      const int i = int(n % mx_);
      const idx kj = n / mx_;
      const int j = int(kj / nz_);
      const idx o = kj * ld_ + i;
      return fabs(pu[o]) * kxm + fabs(pw[o]) * kzm + fabs(pv[o]) * ky / dy[j];
    });
    const double dta = P.cfl / std::max(g_comm.allmax(cmax), 1e-12);
    const double kxm2 = kxmax();
    // scalars: the theta-method in x, y is unconditionally stable for theta >= 1/2, the explicit part has the
    // diffusivity (1 - 2 theta) kappa; the z-diffusion is always explicit
    double dm = nu * (kxm2 * kxm2 + lam2z + lam2y);
    for (int n = 0; n < nsc; ++n)
      dm = std::max(dm, kap[n] * (std::max(0.0, 1 - 2 * P.scal_theta[n]) * (kxm2 * kxm2 + lam2y) + lam2z));
    const double dtv = P.cfl_visc / dm;
    double d = std::min({dta, dtv, P.dt_max});
    // internal gravity waves (as HiCHAN): |lambda| <= N, N^2 = |Ri| max |dS/dy| of the mean profile
    for (int n = 0; n < nsc; ++n)
      if (P.ri[n] != 0) {
        const std::vector<double> g = D1.apply(mean_of(Sf[n].p));
        double gm = 0;
        for (double x : g) gm = std::max(gm, std::abs(x));
        const double N = std::sqrt(std::abs(P.ri[n]) * gm);
        if (N > 0) d = std::min(d, P.cfl / N);
      }
    if (step > 0) d = std::min(d, P.dt_growth * dt);
    dt = d;
  }

  // 3: r = u + dtg H + dtz H_old in P3..P5, H_old = H
  void rhs(double dtg_, double dtz_) {
    const int nkx_ = nkx, lp_ = lp, nz_ = nz, ny_ = ny;
    const real dtg = real(dtg_), dtz = real(dtz_), nuu = real(nu), sc = real(1.0 / mx);
    const real fx = (P.forcing == "cpg") ? real(P.dpdx) : real(0);
    const cplx* F[3] = {u.p, v.p, w.p};
    cplx* H[3] = {hx.p, hy.p, hz_.p};
    cplx* R[3];
    for (int c = 0; c < 3; ++c) R[c] = reinterpret_cast<cplx*>(Pb[3 + c].p);
    const real* kx = d_kx.p;
    const ZSt z = zs;
    const YDev D = ydev(1);
    const idx nl = idx(nz_) * nkx_;
    const cplx* HB[3] = {hbA.p, hbB.p, hbC.p};
    for (int c = 0; c < 3; ++c) {
      const cplx* f = F[c];
      const cplx* hf = HB[c];
      cplx* h = H[c];
      cplx* r = R[c];
      const real force = (c == 0) ? fx : real(0);
      RhsPoint pt;
      pt.f = f; pt.hf = hf; pt.h = h; pt.r = r; pt.kx = kx; pt.z = z;
      pt.nkx = nkx_; pt.nz = nz_; pt.ny = ny_;
      pt.sc = sc; pt.nuu = nuu; pt.dtg = dtg; pt.dtz = dtz; pt.force = force;
      for (int q = 0; q < nsc; ++q)
        if (P.ri[q] != 0 && gvec[c] != 0) { pt.bs[pt.nb] = Sf[q].p; pt.bco[pt.nb] = real(-P.ri[q] * gvec[c]); ++pt.nb; }
      if (zcomp) {  // compact z: D2z f first
        pt.d2z = zderiv(1, f, nkx_, hf, DZ1.p, ZB1.p);
      }
      if (P.fuse_y) {
        // D2y f is formed in the y-sweep and used at once (x3d2's fused transport kernels): one thread per
        // (kx, z) line, no D2y array written and read again
        YLines<1> L;
        L.in[0] = f; L.tmp[0] = T1.p;
        pfor_lines(nl, [=] HD(idx ln) {
          const int i = int(ln % nkx_), k = int(ln / nkx_);
          ysweep<1>(D, L, nl, ln, [&](int j, const cplx* d2y) {
            pt(idx(j) * nl + ln, (idx(j) * nz_ + k) * lp_ + i, i, k, j, d2y[0]);
          });
        });
      } else {
        ydiff(1, f, T1.p);
        const cplx* d2y = T1.p;
        pfor(nsp, [=] HD(idx n) {
          const int i = int(n % nkx_);
          const idx kj = n / nkx_;
          pt(n, kj * lp_ + i, i, int(kj % nz_), int(kj / nz_), d2y[n]);
        });
      }
    }
  }

  // 4-5: pressure projection of r (P3..P5) into u, v, w
  void project() {
    const int nkx_ = nkx, lp_ = lp, nz_ = nz, ny_ = ny;
    const cplx* Rx = reinterpret_cast<cplx*>(Pb[3].p);
    const cplx* Ry = reinterpret_cast<cplx*>(Pb[4].p);
    const cplx* Rz = reinterpret_cast<cplx*>(Pb[5].p);
    const real* kx = d_kx.p;
    const YDev D = ydev(0);
    const ZSt z = zs;
    cplx *t1 = T1.p, *t2 = T2.p;
    g_timer.tic(tm_p_rhs);
    halo(Rz, lp_, hbA.p);
    const cplx* hR = hbA.p;
    ZDer dzr;
    if (zcomp) dzr = zderiv(0, Rz, lp_, hR, DZ1.p, ZB1.p);
    // right-hand side  (B1 E r_y)_j - (A1 c)_j,  c = -(i kx r_x + D1z r_z) at interior points, 0 at the walls
    pfor(nsp, [=] HD(idx n) {
      const int i = int(n % nkx_);
      const idx kj = n / nkx_;
      const int k = int(kj % nz_), j = int(kj / nz_);
      auto cval = [&](int jj) -> cplx {
        if (jj <= 0 || jj >= ny_ - 1) return cplx(0, 0);
        cplx d = itimes(kx[i], Rx[(idx(jj) * nz_ + k) * lp_ + i]);
        if (dzr.valid()) d += dzr.at(jj, k, i);
        else
          for (int l = -z.h; l <= z.h; ++l) {
            if (l == 0) continue;
            d += z.a1[l + z.h] * zv(Rz, hR, lp_, i, k + l, jj, nz_, z.h, ny_, nkx_);
          }
        return -d;
      };
      cplx s(0, 0);
      const int s0 = D.s[j];
      for (int m = 0; m < D.W; ++m) {
        const int jj = s0 + m;
        if (jj == 0 || jj == ny_ - 1) continue;
        s += D.c[j * D.W + m] * Ry[(idx(jj) * nz_ + k) * lp_ + i];
      }
      s -= cval(j);
      if (D.compact) {
        if (j > 0) s -= D.lo[j] * cval(j - 1);
        if (j < ny_ - 1) s -= D.up[j] * cval(j + 1);
      }
      t1[n] = s;
    });
    g_timer.toc(tm_p_rhs);
    if (pch > 1) {
      pressure_pipe(t1);  // chunks; the parts are not timed separately (they overlap)
    } else {
      // (kx, z local, y) -> (z global, kx local, y): all-to-all to the kx-slabs
      transpose_to_kx(t1, t2);
      zfft.exec(T2.p, -1);
      g_timer.tic(tm_p_lu);
      solve_columns();
      g_timer.toc(tm_p_lu);
      zfft.exec(T2.p, +1);
      transpose_to_z(t2, t1);
    }
    g_timer.tic(tm_p_grad);
    halo(t1, nkx_, hbB.p);
    const cplx* hP = hbB.p;
    ZDer dzp;
    if (zcomp) dzp = zderiv(0, t1, nkx_, hP, DZ1.p, ZB1.p);
    ydiff(0, T1.p, T2.p);  // D1y phi
    // u = r - grad phi
    cplx *U = u.p, *V = v.p, *Wv = w.p;
    pfor(nsp, [=] HD(idx n) {
      const int i = int(n % nkx_);
      const idx kj = n / nkx_;
      const int k = int(kj % nz_), j = int(kj / nz_);
      if (j == 0 || j == ny_ - 1) {
        U[n] = cplx(0, 0); V[n] = cplx(0, 0); Wv[n] = cplx(0, 0);
        return;
      }
      const idx o = kj * lp_ + i;
      cplx gz(0, 0);
      if (dzp.valid()) gz = dzp.at(j, k, i);
      else
        for (int l = -z.h; l <= z.h; ++l) {
          if (l == 0) continue;
          gz += z.a1[l + z.h] * zv(t1, hP, nkx_, i, k + l, j, nz_, z.h, ny_, nkx_);
        }
      cplx uu = Rx[o] - itimes(kx[i], t1[n]);
      cplx vv = Ry[o] - t2[n];
      cplx ww = Rz[o] - gz;
      if (i == 0) { uu.im = 0; vv.im = 0; ww.im = 0; }
      U[n] = uu; V[n] = vv; Wv[n] = ww;
    });
    // kx = 0: the kz = 0 and kz = Nyquist parts of v vanish (k = 0, continuity gives D1 v = 0, v = 0 at walls)
    double* prof = d_prof.p;
    const bool nyq = (nzg % 2 == 0);
    const int z0_ = z0, nzg_ = nzg;
    rowsum(ny_, nz_, [=] HD(int j, int k) { return double(V[(idx(j) * nz_ + k) * nkx_].re); }, prof);
    rowsum(ny_, nz_, [=] HD(int j, int k) { return ((k + z0_) % 2 ? -1.0 : 1.0) * double(V[(idx(j) * nz_ + k) * nkx_].re); },
           prof + ny_);
    {
      std::vector<double> ps = fetch(prof, 2 * ny_);
      g_comm.allsum(ps);
      d_prof.upload(ps.data(), 2 * ny_);
    }
    pfor(idx(nz_) * ny_, [=] HD(idx n) {
      const int k = int(n % nz_), j = int(n / nz_);
      const double m0 = prof[j] / nzg_, mn = nyq ? prof[ny_ + j] / nzg_ : 0.0;
      V[(idx(j) * nz_ + k) * nkx_].re -= real(m0 + (((k + z0_) % 2) ? -mn : mn));
    });
    g_timer.toc(tm_p_grad);
  }

  // The pressure matrix of a column, G0 + k2 G1, depends only on its k2 = kx^2 + kz'^2, so it is factorised
  // once (banded LU with partial pivoting: U rows in su, multipliers in sl, pivots in sp) and every RK stage only
  // does the substitutions. The results are identical to factorising in every solve.
  template <int KK>
  void factor_impl() {
    const int nkx_ = nkl, nz_ = nzg, ny_ = ny, ntz_ = ntz, kx0_ = kx0;
    const idx nc = ncols;
    const real *G0 = d_G0.p, *G1 = d_G1.p;
    const real* kx = d_kx.p;
    const real* kzp2 = d_kzp2.p;
    real *su = d_su.p, *sl = d_sl.p;
    unsigned char* sp = d_sp.p;
    pfor_lines(nc, [=] HD(idx col) {
      const int t = int(col % ntz_), i = int(col / ntz_);
      if (i + kx0_ == 0 && (t == 0 || 2 * t == nz_)) return;  // solved separately (zero)
      const real k2 = kx[i + kx0_] * kx[i + kx0_] + kzp2[t];
      band_factor_piv<KK, KK>(
          2 * ny_, G0, G1, k2,
          [=](int r, int m) -> real& { return su[(idx(r) * (2 * KK + 1) + m) * nc + col]; },
          [=](int r, int q) -> real& { return sl[(idx(r) * KK + q) * nc + col]; },
          [=](int r) -> unsigned char& { return sp[idx(r) * nc + col]; });
    });
    (void)nkx_; (void)ny_;
  }
  // columns c0 <= col < c1; T2 as (j, i, k) (kmaj = false) or (i, j, k) (kmaj = true, pipelined step)
  template <int KK>
  void solve_impl(idx c0, idx c1, bool kmaj) {
    const int nkx_ = nkl, nz_ = nzg, ny_ = ny, ntz_ = ntz, kx0_ = kx0;
    const idx nc = ncols;
    const real *su = d_su.p, *sl = d_sl.p;
    const unsigned char* sp = d_sp.p;
    real* sy = d_sy.p;
    cplx* t2 = T2.p;
    const real inz = real(1.0 / nzg);
    pfor_lines(c1 - c0, [=] HD(idx cl) {
      const idx col = c0 + cl;
      const int t = int(col % ntz_), i = int(col / ntz_);
      const int t2i = (nz_ - t) % nz_;
      const bool paired = (t != 0) && (2 * t != nz_);
      auto base = [=](int j) -> idx { return kmaj ? (idx(i) * ny_ + j) * nz_ : (idx(j) * nkx_ + i) * nz_; };
      if (i + kx0_ == 0 && (t == 0 || 2 * t == nz_)) {
        for (int j = 0; j < ny_; ++j) t2[base(j) + t] = cplx(0, 0);
        return;
      }
      band_solve_factored<KK, KK>(
          2 * ny_,
          [=](int r, int c) -> real {
            if (r % 2 == 0) return real(0);
            const idx b = base(r / 2);
            if (c < 2) return c == 0 ? t2[b + t].re : t2[b + t].im;
            if (!paired) return real(0);
            return c == 2 ? t2[b + t2i].re : t2[b + t2i].im;
          },
          [=](int r, int c, real val) {
            if (r % 2) return;
            const idx b = base(r / 2);
            val *= inz;
            if (c == 0) t2[b + t].re = val;
            else if (c == 1) t2[b + t].im = val;
            else if (paired) {
              if (c == 2) t2[b + t2i].re = val;
              else t2[b + t2i].im = val;
            }
          },
          [=](int r, int m) -> real { return su[(idx(r) * (2 * KK + 1) + m) * nc + col]; },
          [=](int r, int q) -> real { return sl[(idx(r) * KK + q) * nc + col]; },
          [=](int r) -> int { return sp[idx(r) * nc + col]; },
          [=](int r, int c) -> real& { return sy[(idx(r) * NRHS + c) * nc + col]; });
    });
  }
  void factor_columns() {
    switch (K) {
      case 4: factor_impl<4>(); break;
      case 6: factor_impl<6>(); break;
      case 8: factor_impl<8>(); break;
      case 10: factor_impl<10>(); break;
      case 12: factor_impl<12>(); break;
      case 14: factor_impl<14>(); break;
      case 16: factor_impl<16>(); break;
      default: fatal("unsupported band width");
    }
    factored = true;
  }
  void solve_columns(idx c0 = 0, idx c1 = -1, bool kmaj = false) {
    if (c1 < 0) c1 = ncols;
    if (!factored) factor_columns();
    switch (K) {
      case 4: solve_impl<4>(c0, c1, kmaj); break;
      case 6: solve_impl<6>(c0, c1, kmaj); break;
      case 8: solve_impl<8>(c0, c1, kmaj); break;
      case 10: solve_impl<10>(c0, c1, kmaj); break;
      case 12: solve_impl<12>(c0, c1, kmaj); break;
      case 14: solve_impl<14>(c0, c1, kmaj); break;
      case 16: solve_impl<16>(c0, c1, kmaj); break;
      default: fatal("unsupported band width");
    }
  }

  // constant flow rate: add a uniform increment to the interior mean u so that the bulk velocity is ubulk
  void cfr_correct() {
    const int nkx_ = nkx, nz_ = nz, ny_ = ny;
    cplx* U = u.p;
    rowsum(ny_, nz_, [=] HD(int j, int k) { return double(U[(idx(j) * nz_ + k) * nkx_].re); }, d_prof.p);
    std::vector<double> pr = fetch(d_prof.p, ny_);
    g_comm.allsum(pr);
    double b = 0;
    for (int j = 0; j < ny_; ++j) b += q[size_t(j)] * pr[size_t(j)];
    b /= 2.0 * nzg;
    const double d = (P.ubulk - b) * 2.0 / qint;
    dpdx_acc += d;
    const real dr = real(d);
    pfor(idx(nz_) * (ny_ - 2), [=] HD(idx n) {
      const int k = int(n % nz_), j = int(n / nz_) + 1;
      U[(idx(j) * nz_ + k) * nkx_].re += dr;
    });
  }
  void reset_dpdx_accumulator() { dpdx_acc = 0; }

  // ---------------------------------------------------------------- initial conditions
  void set_fields(const std::vector<cplx>& hu, const std::vector<cplx>& hv, const std::vector<cplx>& hw) {
    u.upload(hu); v.upload(hv); w.upload(hw);
  }
  // project the current u, v, w (used after setting an initial field)
  void project_current() {
    const int nkx_ = nkx, lp_ = lp;
    const cplx *U = u.p, *V = v.p, *Wv = w.p;
    cplx* R[3];
    for (int c = 0; c < 3; ++c) R[c] = reinterpret_cast<cplx*>(Pb[3 + c].p);
    cplx *R0 = R[0], *R1 = R[1], *R2 = R[2];
    pfor(nsp, [=] HD(idx n) {
      const int i = int(n % nkx_);
      const idx o = (n / nkx_) * lp_ + i;
      R0[o] = U[n]; R1[o] = V[n]; R2[o] = Wv[n];
    });
    project();
  }
  void init_flow() {
    std::vector<cplx> hu(size_t(nsp), cplx(0, 0)), hv = hu, hw = hu;
    auto at = [&](int i, int k, int j) { return size_t((idx(j) * nz + k) * nkx + i); };
    const double ul = (P.forcing == "cfr") ? 1.5 * P.ubulk : 0.5 * P.re * P.dpdx;
    for (int j = 0; j < ny; ++j)
      for (int k = 0; k < nz; ++k) hu[at(0, k, j)] = cplx(real(ul * (1 - y[size_t(j)] * y[size_t(j)])), 0);
    if (P.init_type == "mode") {
      // a wave exp(i(alpha x + beta z)) with stream function (1-y^2)^2 in the direction of the wavevector
      const int i = P.mode_kx;
      if (i < 0 || i >= nkx) fatal("mode_kx out of range");
      const double al = kx_h[size_t(i)], be = 2 * M_PI * P.mode_kz / P.lz, kk = std::sqrt(al * al + be * be);
      // v(y): -i k psi with psi = (1-y^2)^2, or a profile from mode_file (e.g. an eigenfunction); then
      // u_par = (i/k) dv/dy from continuity in the direction of the wavevector
      std::vector<std::complex<double>> vp(static_cast<size_t>(ny)), up(static_cast<size_t>(ny));
      if (P.mode_file.empty()) {
        for (int j = 0; j < ny; ++j) {
          const double yy = y[size_t(j)];
          vp[size_t(j)] = std::complex<double>(0, -kk) * (1 - yy * yy) * (1 - yy * yy);
        }
      } else {
        std::ifstream mf(P.mode_file);
        if (!mf) fatal("cannot open mode_file");
        double yy, vr, vi;
        for (int j = 0; j < ny; ++j) {
          if (!(mf >> yy >> vr >> vi)) fatal("mode_file has fewer than ny lines");
          if (std::abs(yy - y[size_t(j)]) > 1e-8) fatal("mode_file: y does not match the grid");
          vp[size_t(j)] = {vr, vi};
        }
        double vm = 0;
        for (auto& c : vp) vm = std::max(vm, std::abs(c));
        for (auto& c : vp) c /= vm;
      }
      {
        std::vector<double> vr(static_cast<size_t>(ny)), vi(static_cast<size_t>(ny));
        for (int j = 0; j < ny; ++j) { vr[size_t(j)] = vp[size_t(j)].real(); vi[size_t(j)] = vp[size_t(j)].imag(); }
        std::vector<double> dr = D1.apply(vr), di = D1.apply(vi);
        for (int j = 0; j < ny; ++j)
          up[size_t(j)] = std::complex<double>(0, 1.0 / kk) * std::complex<double>(dr[size_t(j)], di[size_t(j)]);
      }
      for (int j = 0; j < ny; ++j) {
        for (int k = 0; k < nz; ++k) {
          const double ph = be * (k + z0) * hz;
          const std::complex<double> e = P.init_amp * std::polar(1.0, ph);
          const std::complex<double> uu = e * (al / kk) * up[size_t(j)], ww = e * (be / kk) * up[size_t(j)],
                                     vv = e * vp[size_t(j)];
          hu[at(i, k, j)] += cplx(real(uu.real()), real(uu.imag()));
          hv[at(i, k, j)] = cplx(real(vv.real()), real(vv.imag()));
          hw[at(i, k, j)] = cplx(real(ww.real()), real(ww.imag()));
        }
      }
    } else if (P.init_type == "noise") {
      // random large-scale modes: |kx index| <= 4, |kz index| <= 4, smooth in y
      // init_amp is the rms of the perturbation (average of the three components over the channel)
      std::mt19937 g(unsigned(P.init_seed));
      std::normal_distribution<double> N(0, 1);
      std::vector<std::complex<double>> pu(static_cast<size_t>(nsp), 0.0), pv = pu, pw = pu;
      for (int i = 0; i <= std::min(4, nkx - 1); ++i)
        for (int m = -4; m <= 4; ++m) {
          if (i == 0 && m <= 0) continue;
          const double al = kx_h[size_t(i)], be = 2 * M_PI * m / P.lz, kk = std::sqrt(al * al + be * be);
          std::complex<double> a[3], b[3];
          for (int c = 0; c < 3; ++c) { a[c] = {N(g), N(g)}; b[c] = {N(g), N(g)}; }
          for (int j = 0; j < ny; ++j) {
            const double yy = y[size_t(j)], e2 = (1 - yy * yy);
            // stream function (wall-normal velocity) and wall-normal vorticity parts
            const std::complex<double> psi = e2 * e2 * (a[0] + a[1] * yy + a[2] * yy * yy);
            const std::complex<double> dpsi = e2 * e2 * (a[1] + 2.0 * a[2] * yy) - 4 * yy * e2 * (a[0] + a[1] * yy + a[2] * yy * yy);
            const std::complex<double> chi = e2 * (b[0] + b[1] * yy + b[2] * yy * yy);
            for (int k = 0; k < nz; ++k) {
              const std::complex<double> e = std::polar(1.0, be * (k + z0) * hz);
              std::complex<double> uu = e * ((al / kk) * dpsi - (be / kk) * chi);
              std::complex<double> ww = e * ((be / kk) * dpsi + (al / kk) * chi);
              std::complex<double> vv = e * std::complex<double>(0, -kk) * psi;
              if (i == 0) {  // real field for kx = 0: add the complex conjugate wave
                uu = 2.0 * uu.real(); vv = 2.0 * vv.real(); ww = 2.0 * ww.real();
              }
              pu[at(i, k, j)] += uu;
              pv[at(i, k, j)] += vv;
              pw[at(i, k, j)] += ww;
            }
          }
        }
      double ms = 0;
      for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
          for (int i = 0; i < nkx; ++i) {
            const size_t a = at(i, k, j);
            ms += q[size_t(j)] * (i == 0 ? 1.0 : 2.0) * (std::norm(pu[a]) + std::norm(pv[a]) + std::norm(pw[a]));
          }
      ms = g_comm.allsum(ms);
      ms /= 3.0 * 2.0 * nzg;
      const double sc = ms > 0 ? P.init_amp / std::sqrt(ms) : 0.0;
      for (size_t a = 0; a < size_t(nsp); ++a) {
        hu[a] += cplx(real(sc * pu[a].real()), real(sc * pu[a].imag()));
        hv[a] = cplx(real(sc * pv[a].real()), real(sc * pv[a].imag()));
        hw[a] = cplx(real(sc * pw[a].real()), real(sc * pw[a].imag()));
      }
    } else if (P.init_type != "laminar") {
      fatal("init_type must be laminar, noise, mode or restart");
    }
    set_fields(hu, hv, hw);
    project_current();
    if (P.forcing == "cfr") cfr_correct();
    for (int n = 0; n < nsc; ++n) init_scalar(n);
  }

  // initial scalar n (sinit, as HiCHAN): linear (the conduction profile if both walls are Dirichlet, else zero),
  // zero, or noise (linear plus random large-scale modes |kx index| <= 4, |kz index| <= 4, (1 - y^2) times a
  // random quadratic in y, of rms init_amp)
  void init_scalar(int n) {
    std::vector<cplx> hs(size_t(nsp), cplx(0, 0));
    auto at = [&](int i, int k, int j) { return size_t((idx(j) * nz + k) * nkx + i); };
    if (P.sinit[n] != "zero" && P.sbc_bot[n] == "D" && P.sbc_top[n] == "D")
      for (int j = 0; j < ny; ++j) {
        const double yy = y[size_t(j)], sl = 0.5 * (P.sval_bot[n] * (1 - yy) + P.sval_top[n] * (1 + yy));
        for (int k = 0; k < nz; ++k) hs[at(0, k, j)] = cplx(real(sl), 0);
      }
    if (P.sinit[n] == "noise") {
      std::mt19937 g(unsigned(P.init_seed + 1000 * (n + 1)));
      std::normal_distribution<double> N(0, 1);
      std::vector<std::complex<double>> ps(static_cast<size_t>(nsp), 0.0);
      for (int i = 0; i <= std::min(4, nkx - 1); ++i)
        for (int m = -4; m <= 4; ++m) {
          if (i == 0 && m <= 0) continue;
          const double be = 2 * M_PI * m / P.lz;
          std::complex<double> a[3];
          for (int c = 0; c < 3; ++c) a[c] = {N(g), N(g)};
          for (int j = 1; j < ny - 1; ++j) {
            const double yy = y[size_t(j)];
            const std::complex<double> f = (1 - yy * yy) * (a[0] + a[1] * yy + a[2] * (2 * yy * yy - 1));
            for (int k = 0; k < nz; ++k) {
              std::complex<double> v = std::polar(1.0, be * (k + z0) * hz) * f;
              if (i == 0) v = 2.0 * v.real();
              ps[at(i, k, j)] += v;
            }
          }
        }
      double ms = 0;
      for (int j = 0; j < ny; ++j)
        for (int k = 0; k < nz; ++k)
          for (int i = 0; i < nkx; ++i) ms += q[size_t(j)] * (i == 0 ? 1.0 : 2.0) * std::norm(ps[at(i, k, j)]);
      ms = g_comm.allsum(ms) / (2.0 * nzg);
      const double sc = ms > 0 ? P.init_amp / std::sqrt(ms) : 0.0;
      for (size_t a = 0; a < size_t(nsp); ++a)
        hs[a] += cplx(real(sc * ps[a].real()), real(sc * ps[a].imag()));
    }
    Sf[n].upload(hs);
  }

  // ---------------------------------------------------------------- diagnostics
  // per-y sums over (z, kx) with the Parseval weights (kx > 0 counted twice), divided by nz
  std::vector<double> profile(int a, int b, bool cross) {
    const cplx* F[3] = {u.p, v.p, w.p};
    (void)cross;
    return profile_of(F[a], F[b]);
  }
  // <f g> over (x, z) per y, for any two spectral arrays (layout of u)
  std::vector<double> profile_of(const cplx* fa, const cplx* fb) {
    const int nkx_ = nkx, nz_ = nz;
    rowsum(ny, nz_ * nkx_, [=] HD(int j, int c) {
      const int i = c % nkx_;
      const idx n = idx(j) * nz_ * nkx_ + c;
      const double wgt = (i == 0) ? 1.0 : 2.0;
      return wgt * double(fa[n].re * fb[n].re + fa[n].im * fb[n].im);
    }, d_prof.p);
    std::vector<double> p = fetch(d_prof.p, ny);
    g_comm.allsum(p);
    for (double& x : p) x /= nzg;
    return p;
  }
  std::vector<double> mean_profile(int c) {
    const cplx* F[3] = {u.p, v.p, w.p};
    return mean_of(F[c]);
  }
  std::vector<double> mean_of(const cplx* f) {
    const int nkx_ = nkx, nz_ = nz;
    rowsum(ny, nz_, [=] HD(int j, int k) { return double(f[(idx(j) * nz_ + k) * nkx_].re); }, d_prof.p);
    std::vector<double> p = fetch(d_prof.p, ny);
    g_comm.allsum(p);
    for (double& x : p) x /= nzg;
    return p;
  }
  double integrate(const std::vector<double>& f) const {
    double s = 0;
    for (int j = 0; j < ny; ++j) s += q[size_t(j)] * f[size_t(j)];
    return s;
  }
  // energy of one Fourier mode as in HiCHAN: integral over y of |u|^2 + |v|^2 + |w|^2 of its coefficient
  // f(kx, kz, y) = (1/nz) sum_z f(kx, z, y) exp(-i kz z); kx >= 0, kz may be negative; -1 (kx < 0): not logged (0)
  double mode_energy(int ikx, int ikz) {
    if (ikx < 0 || ikx >= nkx) return 0;
    const int nkx_ = nkx, nz_ = nz, z0_ = z0, nzg_ = nzg;
    const double kz2pi = 2 * M_PI * double(ikz) / nzg_;
    std::vector<double> cre[3], cim[3];
    const cplx* F[3] = {u.p, v.p, w.p};
    for (int c = 0; c < 3; ++c) {
      const cplx* f = F[c];
      rowsum(ny, nz_, [=] HD(int j, int k) {
        const cplx a = f[(idx(j) * nz_ + k) * nkx_ + ikx];
        const double ph = kz2pi * double(z0_ + k), cs = cos(ph), sn = sin(ph);
        return double(a.re) * cs + double(a.im) * sn;  // Re(a exp(-i ph))
      }, d_prof.p);
      cre[c] = fetch(d_prof.p, ny);
      rowsum(ny, nz_, [=] HD(int j, int k) {
        const cplx a = f[(idx(j) * nz_ + k) * nkx_ + ikx];
        const double ph = kz2pi * double(z0_ + k), cs = cos(ph), sn = sin(ph);
        return double(a.im) * cs - double(a.re) * sn;  // Im(a exp(-i ph))
      }, d_prof.p);
      cim[c] = fetch(d_prof.p, ny);
      g_comm.allsum(cre[c]);
      g_comm.allsum(cim[c]);
    }
    std::vector<double> e(static_cast<size_t>(ny), 0.0);
    for (int c = 0; c < 3; ++c)
      for (int j = 0; j < ny; ++j) {
        const double re = cre[c][size_t(j)] / nzg_, im = cim[c][size_t(j)] / nzg_;
        e[size_t(j)] += re * re + im * im;
      }
    return integrate(e);
  }
  double max_divergence() {
    const int nkx_ = nkx, nz_ = nz, ny_ = ny;
    ydiff(0, v.p, T1.p);
    halo(w.p, nkx, hbC.p);
    const cplx* hW = hbC.p;
    ZDer dzw;
    if (zcomp) dzw = zderiv(0, w.p, nkx, hW, DZ1.p, ZB1.p);
    const cplx *U = u.p, *Wv = w.p, *dyv = T1.p;
    const real* kx = d_kx.p;
    const ZSt z = zs;
    return g_comm.allmax(reduce_max(nsp, [=] HD(idx n) {
      const int i = int(n % nkx_);
      const idx kj = n / nkx_;
      const int k = int(kj % nz_), j = int(kj / nz_);
      cplx d = itimes(kx[i], U[n]) + dyv[n];
      if (dzw.valid()) d += dzw.at(j, k, i);
      else
        for (int l = -z.h; l <= z.h; ++l) {
          if (l == 0) continue;
          d += z.a1[l + z.h] * zv(Wv, hW, nkx_, i, k + l, j, nz_, z.h, ny_, nkx_);
        }
      return sqrt(abs2(d));
    }));
  }
  struct Log {
    double ub, tau_b, tau_t, tau_w, retau, div, ef, emode;
  };
  Log diagnostics() {
    Log L;
    std::vector<double> U = mean_profile(0);
    L.ub = integrate(U) / 2;
    std::vector<double> dU = D1.apply(U);
    const double g0 = dU[0], g1 = dU[size_t(ny - 1)];
    L.tau_b = nu * g0;
    L.tau_t = -nu * g1;
    L.tau_w = 0.5 * (std::abs(L.tau_b) + std::abs(L.tau_t));  // as HiCHAN: mean of both walls
    L.retau = std::sqrt(L.tau_w) * P.re;
    double e = 0;
    for (int c = 0; c < 3; ++c) e += integrate(profile(c, c, false));
    double em = 0;
    {
      std::vector<double> U2(U.size());
      for (size_t j = 0; j < U.size(); ++j) U2[j] = U[j] * U[j];
      em = integrate(U2);
    }
    L.ef = 0.25 * (e - em);  // (1/2) <u'.u'> averaged over the channel height 2
    L.div = max_divergence();
    L.emode = mode_energy(P.mode_log_kx, P.mode_log_kz);
    return L;
  }
};

}  // namespace fd
