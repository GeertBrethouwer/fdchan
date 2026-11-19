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
#include <random>
#include <string>
#include <vector>

#include "backend.hpp"
#include "fft.hpp"
#include "operators.hpp"
#include "params.hpp"
#include "psolve.hpp"
#include "comm.hpp"

namespace fd {

constexpr int MAXW = 12;  // max stencil width stored per row
constexpr int MAXP2 = 4;  // max half width of the z stencils (order 8)

struct ZSt {  // periodic central stencils in z, a[l + h], l = -h..h, already divided by dz (dz^2)
  int h;
  real a1[2 * MAXP2 + 1], a2[2 * MAXP2 + 1];
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
  cplx *h, *r;
  const real* kx;
  ZSt z;
  int nkx, nz, ny;
  real sc, nuu, dtg, dtz, force;
  HD void operator()(idx n, idx o, int i, int k, int j, cplx d2y) const {
    if (j == 0 || j == ny - 1) {
      r[o] = cplx(0, 0);
      h[n] = cplx(0, 0);
      return;
    }
    const cplx fc = f[n];
    cplx lap = (-kx[i] * kx[i]) * fc + d2y;
    for (int l = -z.h; l <= z.h; ++l) lap += z.a2[l + z.h] * zv(f, hf, nkx, i, k + l, j, nz, z.h, ny, nkx);
    cplx hh = sc * r[o] + nuu * lap;
    if (i == 0) hh.re += force;
    r[o] = fc + dtg * hh + dtz * h[n];
    h[n] = hh;
  }
};

// pointwise part of nonlinear(): u, v, w and the vorticity into the padded arrays C0..C5 at padded point n
struct VortPoint {
  const cplx *U, *V, *W, *hU, *hV;
  cplx *C0, *C1, *C2, *C3, *C4, *C5;
  const real* kx;
  ZSt z;
  int nkx, nz, ny;
  HD void operator()(idx n, idx s, int i, int k, int j, cplx dyu, cplx dyw) const {
    cplx dzu(0, 0), dzv(0, 0);
    for (int l = -z.h; l <= z.h; ++l) {
      if (l == 0) continue;
      dzu += z.a1[l + z.h] * zv(U, hU, nkx, i, k + l, j, nz, z.h, ny, nkx);
      dzv += z.a1[l + z.h] * zv(V, hV, nkx, i, k + l, j, nz, z.h, ny, nkx);
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
  DArray<cplx> u, v, w, hx, hy, hz_, T1, T2;
  DArray<real> Pb[9];  // 6 always, 3 more for the advective form
  DArray<double> d_prof, d_scal;
  DArray<cplx> hbA, hbB, hbC, hsend, sbuf, rbuf;  // halo buffers, transpose buffers
  DArray<int> d_kxown, d_kxoff, d_zown, d_zoff;
  std::vector<int> a2a_sc, a2a_sd, a2a_rc, a2a_rd;  // all-to-all counts and displacements (in reals)
  DArray<idx> d_sd, d_rd;
  DArray<int> d_cnt, d_nzr;
  double t_comm = 0, dpdx_acc = 0;
  XFFT xfft;
  ZFFT zfft;
  // state
  double time = 0, dt = 0, dpdx_step = 0;
  long step = 0;
  int W1 = 0, W2 = 0;
  // timers (seconds)
  double t_nl = 0, t_rhs = 0, t_p = 0, t_other = 0;

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
    hz_halo = P.fd_order_z / 2;
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
    setup_operators();
    for (int c = 0; c < (P.nl_form == "advective" ? 9 : 6); ++c) Pb[c].alloc(npad);
    u.alloc(nsp); v.alloc(nsp); w.alloc(nsp);
    hx.alloc(nsp); hy.alloc(nsp); hz_.alloc(nsp);
    T1.alloc(nsp);
    const idx nt2 = idx(ny) * nkl * nzg;
    T2.alloc(std::max(nsp, nt2));
    sbuf.alloc(std::max(nsp, nt2));
    rbuf.alloc(std::max(nsp, nt2));
    const idx nh = 2 * idx(hz_halo) * ny * nkx;
    hbA.alloc(nh); hbB.alloc(nh); hbC.alloc(nh); hsend.alloc(nh);
    setup_transpose();
    d_su.alloc(idx(2 * K + 1) * 2 * ny * ncols);
    d_sy.alloc(idx(NRHS) * 2 * ny * ncols);
    d_prof.alloc(4 * idx(ny));
    d_scal.alloc(8);
    xfft.init(mx, idx(nz) * ny);
    zfft.init(nzg, idx(nkl) * ny);
    dt = P.dt;
  }

  // ---------------------------------------------------------------- set-up
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
    std::vector<double> a1 = central_weights(1, pz), a2 = central_weights(2, pz);
    zs.h = pz / 2;
    for (int l = 0; l <= pz; ++l) {
      zs.a1[l] = real(a1[size_t(l)] / hz);
      zs.a2[l] = real(a2[size_t(l)] / (hz * hz));
    }
    std::vector<double> kzp2(static_cast<size_t>(ntz));
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
    ptranspose(ny, nz_, nkx_,
        [=] HD(idx j, int k, int i) -> cplx { return t1[(j * nz_ + k) * nkx_ + i]; },
        [=] HD(idx j, int k, int i, cplx val) { const int r = own[i]; sb[sd[r] + (j * cnt[r] + off[i]) * nz_ + k] = val; });
    auto t0 = tick();
    g_comm.alltoallv(reinterpret_cast<real*>(sbuf.p), a2a_sc, a2a_sd, reinterpret_cast<real*>(rbuf.p), a2a_rc, a2a_rd);
    t_comm += tick() - t0;
    pfor(idx(ny) * nkl_ * nzg_, [=] HD(idx m) {
      const int kg = int(m % nzg_);
      const idx r = m / nzg_;
      const int s = zown[kg];
      t2[m] = rb[rd[s] + r * nzr[s] + zoff[kg]];
    });
  }
  // inverse of transpose_to_kx
  void transpose_to_z(const cplx* t2, cplx* t1) {
    const int nkx_ = nkx, nz_ = nz, nkl_ = nkl, nzg_ = nzg;
    const int *own = d_kxown.p, *off = d_kxoff.p, *cnt = d_cnt.p, *zown = d_zown.p, *zoff = d_zoff.p, *nzr = d_nzr.p;
    const idx *sd = d_sd.p, *rd = d_rd.p;
    cplx *sb = sbuf.p, *rb = rbuf.p;
    pfor(idx(ny) * nkl_ * nzg_, [=] HD(idx m) {
      const int kg = int(m % nzg_);
      const idx r = m / nzg_;
      const int s = zown[kg];
      rb[rd[s] + r * nzr[s] + zoff[kg]] = t2[m];
    });
    auto t0 = tick();
    g_comm.alltoallv(reinterpret_cast<real*>(rbuf.p), a2a_rc, a2a_rd, reinterpret_cast<real*>(sbuf.p), a2a_sc, a2a_sd);
    t_comm += tick() - t0;
    ptranspose(ny, nkx_, nz_,
        [=] HD(idx j, int i, int k) -> cplx { const int r = own[i]; return sb[sd[r] + (j * cnt[r] + off[i]) * nz_ + k]; },
        [=] HD(idx j, int i, int k, cplx val) { t1[(j * nz_ + k) * nkx_ + i] = val; });
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
    auto t0 = tick();
    // my upper planes -> upper neighbour's lower halo; receive my lower halo from the lower neighbour
    g_comm.sendrecv(reinterpret_cast<const real*>(sb), 2 * part, up, reinterpret_cast<real*>(hal), dn, 11);
    g_comm.sendrecv(reinterpret_cast<const real*>(sb + part), 2 * part, dn, reinterpret_cast<real*>(hal + part), up, 12);
    t_comm += tick() - t0;
  }

  void print_setup() const {
    std::printf("modes     nkx = %d (padded x grid %d), physical z points %d, y points %d\n", nkx, mx, nzg, ny);
    std::printf("parallel  %d rank(s), z-slabs of %d..%d planes, %d..%d x-modes per rank in the pressure solve\n", nrank,
                zst[1] - zst[0], zst[size_t(nrank)] - zst[size_t(nrank - 1)], kxst[1] - kxst[0],
                kxst[size_t(nrank)] - kxst[size_t(nrank - 1)]);
    std::printf("y grid    dy_wall = %.3e, dy_centre = %.3e\n", y[1] - y[0], y[size_t(ny / 2 + 1)] - y[size_t(ny / 2)]);
    std::printf("y scheme  %s, order %d; z order %d\n", P.y_scheme.c_str(), P.fd_order, P.fd_order_z);
    std::printf("pressure  band K = %d, %lld columns of length %d (phi and D1 phi)\n", K, ncols, 2 * ny);
    std::printf("viscous   spectral radii: kx^2 %.3e, D2z %.3e, D2y %.3e\n", kxmax() * kxmax(), lam2z, lam2y);
    const double mb = double(sizeof(real)) / 1048576.0;
    const double tot = (6.0 * npad + 2.0 * 11 * nsp + (2.0 * K + 1.0 + NRHS) * 2 * ny * ncols) * mb;
    std::printf("memory    %.1f MB of field arrays on rank 0 (%.0f bytes per local grid point)\n", tot,
                tot * 1048576.0 / (double(nx) * ny * nz));
  }

  // ---------------------------------------------------------------- RK3 step
  void advance() {
    static const double gam[3] = {8.0 / 15, 5.0 / 12, 3.0 / 4};
    static const double zet[3] = {0.0, -17.0 / 60, -5.0 / 12};
    dpdx_step = 0;
    if (P.forcing == "cfr") reset_dpdx_accumulator();
    for (int s = 0; s < 3; ++s) {
      auto t0 = tick();
      nonlinear();
      if (s == 0) choose_dt();
      auto t1 = tick();
      t_nl += t1 - t0;
      rhs(gam[s] * dt, zet[s] * dt);
      auto t2 = tick();
      t_rhs += t2 - t1;
      project();
      auto t3 = tick();
      t_p += t3 - t2;
      if (P.forcing == "cfr") cfr_correct();
      t_other += tick() - t3;
    }
    if (P.forcing == "cfr") dpdx_step = dpdx_acc / dt;
    else dpdx_step = P.dpdx;
    time += dt;
    ++step;
  }

  double tick() {
    if (P.timing) sync();
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
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
    halo(u.p, nkx, hbA.p);  // halos of u, v, w for this stage (z-derivatives here and in rhs())
    halo(v.p, nkx, hbB.p);
    halo(w.p, nkx, hbC.p);
    if (P.nl_form == "advective") { nonlinear_advective(); return; }
    const int nkx_ = nkx, lp_ = lp, ld_ = ld, mx_ = mx, nz_ = nz, ny_ = ny;
    VortPoint pt;
    pt.U = u.p; pt.V = v.p; pt.W = w.p; pt.hU = hbA.p; pt.hV = hbB.p;
    pt.C0 = reinterpret_cast<cplx*>(Pb[0].p); pt.C1 = reinterpret_cast<cplx*>(Pb[1].p);
    pt.C2 = reinterpret_cast<cplx*>(Pb[2].p); pt.C3 = reinterpret_cast<cplx*>(Pb[3].p);
    pt.C4 = reinterpret_cast<cplx*>(Pb[4].p); pt.C5 = reinterpret_cast<cplx*>(Pb[5].p);
    pt.kx = d_kx.p; pt.z = zs; pt.nkx = nkx_; pt.nz = nz_; pt.ny = ny_;
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
      cplx *G0 = reinterpret_cast<cplx*>(Pb[6].p), *G1 = reinterpret_cast<cplx*>(Pb[7].p),
           *G2 = reinterpret_cast<cplx*>(Pb[8].p);
      pfor(idx(lp_) * nz_ * ny_, [=] HD(idx n) {
        const int i = int(n % lp_);
        const idx kj = n / lp_;
        if (i < nkx_) {
          const int k = int(kj % nz_), j = int(kj / nz_);
          const idx s = kj * nkx_ + i;
          cplx dz(0, 0);
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
    const double dtv = P.cfl_visc / (nu * (kxm2 * kxm2 + lam2z + lam2y));
    double d = std::min({dta, dtv, P.dt_max});
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
    halo(Rz, lp_, hbA.p);
    const cplx* hR = hbA.p;
    // right-hand side  (B1 E r_y)_j - (A1 c)_j,  c = -(i kx r_x + D1z r_z) at interior points, 0 at the walls
    pfor(nsp, [=] HD(idx n) {
      const int i = int(n % nkx_);
      const idx kj = n / nkx_;
      const int k = int(kj % nz_), j = int(kj / nz_);
      auto cval = [&](int jj) -> cplx {
        if (jj <= 0 || jj >= ny_ - 1) return cplx(0, 0);
        cplx d = itimes(kx[i], Rx[(idx(jj) * nz_ + k) * lp_ + i]);
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
    // (kx, z local, y) -> (z global, kx local, y): all-to-all to the kx-slabs
    transpose_to_kx(t1, t2);
    zfft.exec(T2.p, -1);
    solve_columns();
    zfft.exec(T2.p, +1);
    transpose_to_z(t2, t1);
    halo(t1, nkx_, hbB.p);
    const cplx* hP = hbB.p;
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
  }

  template <int KK>
  void solve_impl() {
    const int nkx_ = nkl, nz_ = nzg, ny_ = ny, ntz_ = ntz, kx0_ = kx0;
    const idx nc = ncols;
    const real *G0 = d_G0.p, *G1 = d_G1.p;
    const real* kx = d_kx.p;
    const real* kzp2 = d_kzp2.p;
    real *su = d_su.p, *sy = d_sy.p;
    cplx* t2 = T2.p;
    const real inz = real(1.0 / nzg);
    pfor_lines(nc, [=] HD(idx col) {
      const int t = int(col % ntz_), i = int(col / ntz_);
      const int t2i = (nz_ - t) % nz_;
      const bool paired = (t != 0) && (2 * t != nz_);
      if (i + kx0_ == 0 && (t == 0 || 2 * t == nz_)) {
        for (int j = 0; j < ny_; ++j) t2[(idx(j) * nkx_ + i) * nz_ + t] = cplx(0, 0);
        return;
      }
      const real k2 = kx[i + kx0_] * kx[i + kx0_] + kzp2[t];
      band_solve_piv<KK, KK>(
          2 * ny_, G0, G1, k2,
          [=](int r, int c) -> real {
            if (r % 2 == 0) return real(0);
            const idx b = (idx(r / 2) * nkx_ + i) * nz_;
            if (c < 2) return c == 0 ? t2[b + t].re : t2[b + t].im;
            if (!paired) return real(0);
            return c == 2 ? t2[b + t2i].re : t2[b + t2i].im;
          },
          [=](int r, int c, real val) {
            if (r % 2) return;
            const idx b = (idx(r / 2) * nkx_ + i) * nz_;
            val *= inz;
            if (c == 0) t2[b + t].re = val;
            else if (c == 1) t2[b + t].im = val;
            else if (paired) {
              if (c == 2) t2[b + t2i].re = val;
              else t2[b + t2i].im = val;
            }
          },
          [=](int r, int m) -> real& { return su[(idx(r) * (2 * KK + 1) + m) * nc + col]; },
          [=](int r, int c) -> real& { return sy[(idx(r) * NRHS + c) * nc + col]; });
    });
  }
  void solve_columns() {
    switch (K) {
      case 4: solve_impl<4>(); break;
      case 6: solve_impl<6>(); break;
      case 8: solve_impl<8>(); break;
      case 10: solve_impl<10>(); break;
      case 12: solve_impl<12>(); break;
      case 14: solve_impl<14>(); break;
      case 16: solve_impl<16>(); break;
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
  }

  // ---------------------------------------------------------------- diagnostics
  // per-y sums over (z, kx) with the Parseval weights (kx > 0 counted twice), divided by nz
  std::vector<double> profile(int a, int b, bool cross) {
    const int nkx_ = nkx, nz_ = nz;
    const cplx* F[3] = {u.p, v.p, w.p};
    const cplx *fa = F[a], *fb = F[b];
    rowsum(ny, nz_ * nkx_, [=] HD(int j, int c) {
      const int i = c % nkx_;
      const idx n = idx(j) * nz_ * nkx_ + c;
      const double wgt = (i == 0) ? 1.0 : 2.0;
      return wgt * double(fa[n].re * fb[n].re + fa[n].im * fb[n].im);
    }, d_prof.p);
    (void)cross;
    std::vector<double> p = fetch(d_prof.p, ny);
    g_comm.allsum(p);
    for (double& x : p) x /= nzg;
    return p;
  }
  std::vector<double> mean_profile(int c) {
    const int nkx_ = nkx, nz_ = nz;
    const cplx* F[3] = {u.p, v.p, w.p};
    const cplx* f = F[c];
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
  double mode_energy_v(int i) {
    if (i < 0 || i >= nkx) return 0;
    const int nkx_ = nkx, nz_ = nz;
    const cplx* f = v.p;
    rowsum(ny, nz_, [=] HD(int j, int k) { return double(abs2(f[(idx(j) * nz_ + k) * nkx_ + i])); }, d_prof.p);
    std::vector<double> p = fetch(d_prof.p, ny);
    g_comm.allsum(p);
    for (double& x : p) x /= nzg;
    return 0.5 * integrate(p);
  }
  double max_divergence() {
    const int nkx_ = nkx, nz_ = nz, ny_ = ny;
    ydiff(0, v.p, T1.p);
    halo(w.p, nkx, hbC.p);
    const cplx* hW = hbC.p;
    const cplx *U = u.p, *Wv = w.p, *dyv = T1.p;
    const real* kx = d_kx.p;
    const ZSt z = zs;
    return g_comm.allmax(reduce_max(nsp, [=] HD(idx n) {
      const int i = int(n % nkx_);
      const idx kj = n / nkx_;
      const int k = int(kj % nz_), j = int(kj / nz_);
      cplx d = itimes(kx[i], U[n]) + dyv[n];
      for (int l = -z.h; l <= z.h; ++l) {
        if (l == 0) continue;
        d += z.a1[l + z.h] * zv(Wv, hW, nkx_, i, k + l, j, nz_, z.h, ny_, nkx_);
      }
      return sqrt(abs2(d));
    }));
  }
  struct Log {
    double ub, tau_b, tau_t, retau, div, ef, emode;
  };
  Log diagnostics() {
    Log L;
    std::vector<double> U = mean_profile(0);
    L.ub = integrate(U) / 2;
    std::vector<double> dU = D1.apply(U);
    const double g0 = dU[0], g1 = dU[size_t(ny - 1)];
    L.tau_b = nu * g0;
    L.tau_t = -nu * g1;
    L.retau = std::sqrt(std::max(0.0, 0.5 * (L.tau_b + L.tau_t))) * P.re;
    double e = 0;
    for (int c = 0; c < 3; ++c) e += integrate(profile(c, c, false));
    double em = 0;
    for (double x : U) (void)x;
    {
      std::vector<double> U2(U.size());
      for (size_t j = 0; j < U.size(); ++j) U2[j] = U[j] * U[j];
      em = integrate(U2);
    }
    L.ef = 0.25 * (e - em);  // (1/2) <u'.u'> averaged over the channel height 2
    L.div = max_divergence();
    L.emode = mode_energy_v(P.mode_log_kx);
    return L;
  }
};

}  // namespace fd
