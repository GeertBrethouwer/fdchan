// Tests of the scalars (nscal > 0), u = v = w = 0 unless stated:
//   1. decay of single modes (pure diffusion: theta-method in x, y, explicit in z) against the exact decay of the
//      discrete modes for theta = 0, 0.5, 1, with Dirichlet and Neumann walls and the mean and non-mean modes;
//   2. steady state with a Neumann bottom wall, a Dirichlet top wall and a source: the quadratic profile (exact
//      for the difference schemes), also a Neumann value on both walls with a balancing source;
//   3. hydrostatic balance: Ri = 2, gravity -y, linear stratification, the velocity stays zero;
//   4. the three forms of the advection term (advective, divergence, skew) agree on a
//      divergence-free flow to the truncation error.
//
//   make CXX=g++ build/test_scalars && build/test_scalars                    (one rank)
//   make MPI=1 MPICXX=mpicxx build/test_scalars && mpirun -n 3 build/test_scalars
#include <cstdio>

#include "../src/solver.hpp"

namespace fd {
#ifdef FD_GPU
gpuStream_t g_stream;
#endif
Comm g_comm;
}  // namespace fd

using namespace fd;

static Params base() {
  Params P;
  P.nx = 8; P.ny = 41; P.nz = 24; P.lx = 2 * M_PI; P.lz = 2 * M_PI;
  P.grid_type = "tanh"; P.grid_alpha = 1.5;
  P.fd_order = 6; P.fd_order_z = 6; P.y_scheme = "compact";  // (fd_order_z = 0 is resolved by Params::read only)
  P.re = 10; P.forcing = "cpg"; P.dpdx = 0;
  return P;
}

static void zero_velocity(Solver& S) {
  std::vector<cplx> z(size_t(S.nsp), cplx(0, 0));
  S.set_fields(z, z, z);
}

// max over the domain of |s - ex|, ex given per (i, k, j) as a complex mode value, relative to max |ex|
template <class F>
static double maxerr(Solver& S, int n, F ex) {
  std::vector<cplx> h = S.Sf[n].host();
  double e = 0, m = 0;
  for (int j = 0; j < S.ny; ++j)
    for (int k = 0; k < S.nz; ++k)
      for (int i = 0; i < S.nkx; ++i) {
        const std::complex<double> a = ex(i, k, j);
        const cplx b = h[size_t((idx(j) * S.nz + k) * S.nkx + i)];
        const double d = std::abs(std::complex<double>(b.re, b.im) - a);
        e = std::isfinite(d) ? std::max(e, d) : 1e300;
        m = std::max(m, std::abs(a));
      }
  return g_comm.allmax(e) / std::max(g_comm.allmax(m), 1e-300);
}

int main(int argc, char** argv) {
  g_comm.init(&argc, &argv);
#ifdef FD_GPU
  gpu_check(gpuStreamCreate(&g_stream), "stream");
#else
  Pool::get().set_threads(1);
#endif
  const bool root = g_comm.root();
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  if (root) std::printf("scalars on %d rank(s)\n", g_comm.size);
  bool ok = true;
  // single precision: the errors stop at the round-off of the solves, amplified by the conditioning of the
  // y-systems on the stretched grid (a few 1e-4 for Neumann walls); the tolerances get a floor of 5000 eps
  const double eps = double(std::numeric_limits<real>::epsilon()), floor_ = sizeof(real) == 4 ? 5000 * eps : 0.0;
  auto report = [&](const char* name, double e, double tol) {
    tol = std::max(tol, floor_);
    const bool good = e < tol;
    ok = ok && good;
    if (root) std::printf("  %-66s %.2e (< %.0e) %s\n", name, e, tol, good ? "ok" : "WRONG");
  };

  // ---- 1. decay of modes: the discrete eigenvalues of D2y (Dirichlet / Neumann) from the y operators
  double eneu[2] = {0, 0};
  for (int run = 0; run < 4; ++run) {
    const double th = run == 0 ? 0.0 : run == 2 ? 1.0 : 0.5;
    Params P = base();
    if (run == 3) P.ny = 81;  // grid convergence of the Neumann case
    P.nscal = 2;
    for (int n = 0; n < 2; ++n) { P.pr[n] = 0.7; P.scal_theta[n] = th; P.sval_bot[n] = 0; P.sval_top[n] = 0; P.sinit[n] = "zero"; }
    P.sbc_bot[1] = "N"; P.sbc_top[1] = "N";
    P.fixed_dt = (th > 0); P.dt = 2e-3; P.dt_max = 2e-3;
    Solver S(P);
    zero_velocity(S);
    const double kap = 1.0 / (P.re * P.pr[0]), hz = P.lz / S.nzg;
    // smooth y-profiles: Dirichlet sin(pi (y+1)/2), Neumann cos(pi (y+1)); their D2y is close to -lambda times
    // the profile, the difference is the y-truncation error (small at ny = 41, order 6)
    std::vector<double> fd(size_t(S.ny)), fn(size_t(S.ny));
    for (int j = 0; j < S.ny; ++j) {
      fd[size_t(j)] = std::sin(M_PI * (S.y[size_t(j)] + 1) / 2);
      fn[size_t(j)] = std::cos(M_PI * (S.y[size_t(j)] + 1));
    }
    const double ld = M_PI * M_PI / 4, ln = M_PI * M_PI;
    // z: explicit stencil of order fd_order on mode m, modified wavenumber squared
    auto kz2 = [&](int m) {
      const std::vector<double> a = central_weights(2, P.fd_order_z);
      double s = 0;
      for (int l = -P.fd_order_z / 2; l <= P.fd_order_z / 2; ++l) s += a[size_t(l + P.fd_order_z / 2)] * std::cos(l * m * hz);
      return -s / (hz * hz);
    };
    const int mz = 2, ix = 1;
    const double kx = S.kx_h[size_t(ix)];
    // scalar 0 (Dirichlet): mode (ix, mz) times fd;  scalar 1 (Neumann): mean fn + (0, mz) mode times fn
    auto ex0 = [&](int i, int k, int j, double t) -> std::complex<double> {
      if (i != ix) return 0.0;
      return std::exp(-kap * (kx * kx + kz2(mz) + ld) * t) * std::polar(1.0, mz * (k + S.z0) * hz) * fd[size_t(j)];
    };
    auto ex1 = [&](int i, int k, int j, double t) -> std::complex<double> {
      if (i != 0) return 0.0;
      return (std::exp(-kap * ln * t) + std::exp(-kap * (kz2(mz) + ln) * t) * std::cos(mz * (k + S.z0) * hz)) *
             fn[size_t(j)];
    };
    for (int n = 0; n < 2; ++n) {
      std::vector<cplx> h(size_t(S.nsp), cplx(0, 0));
      for (int j = 0; j < S.ny; ++j)
        for (int k = 0; k < S.nz; ++k)
          for (int i = 0; i < S.nkx; ++i) {
            const std::complex<double> a = n == 0 ? ex0(i, k, j, 0) : ex1(i, k, j, 0);
            h[size_t((idx(j) * S.nz + k) * S.nkx + i)] = cplx(real(a.real()), real(a.imag()));
          }
      S.Sf[n].upload(h);
    }
    while (S.time < 0.5) S.advance();
    const double t = S.time;
    // tolerances: theta = 1 is first order in time; the Neumann case has the larger y-truncation error of the
    // one-sided boundary closures (checked for convergence below)
    const double tol = th == 1.0 ? 2e-3 : 1e-6;
    char name[128];
    std::snprintf(name, sizeof(name), "theta = %.1f, ny = %d: Dirichlet mode (kx, kz) = (1, 2) decay (t = %.3f)", th, S.ny, t);
    report(name, maxerr(S, 0, [&](int i, int k, int j) { return ex0(i, k, j, t); }), tol);
    std::snprintf(name, sizeof(name), "theta = %.1f, ny = %d: Neumann mean + (0, 2) mode decay", th, S.ny);
    const double en = maxerr(S, 1, [&](int i, int k, int j) { return ex1(i, k, j, t); });
    report(name, en, th == 1.0 ? 2e-3 : 5e-4);
    if (run == 1) eneu[0] = en;
    if (run == 3) eneu[1] = en;
  }
  {
    const double ord = std::log2(eneu[0] / eneu[1]);
    if (root) std::printf("  Neumann decay error ny = 41 -> 81: %.2e -> %.2e, order %.1f\n", eneu[0], eneu[1], ord);
    if (sizeof(real) == 8)  // in single precision the error at ny = 41 is already at round-off
      report("Neumann decay: convergence order in y >= 3 (1 / order, ny 41 -> 81)", 1 / std::max(ord, 1e-3), 1.0 / 3);
  }

  // ---- 2. steady states (theta = 1, large time steps)
  {
    Params P = base();
    P.nscal = 2;
    for (int n = 0; n < 2; ++n) { P.pr[n] = 1; P.scal_theta[n] = 1; P.sinit[n] = "zero"; }
    // scalar 0: ds/dy(-1) = 0.7, s(1) = -0.3, source 2:  s = -Q/(2 kap) y^2 + a y + b
    P.sbc_bot[0] = "N"; P.sval_bot[0] = 0.7; P.sbc_top[0] = "D"; P.sval_top[0] = -0.3; P.ssrc_const[0] = 2;
    // scalar 1: ds/dy = 1 at the bottom, ds/dy = -1 at the top, source Q = kap: kap s'' = -Q, s' = -y
    P.sbc_bot[1] = "N"; P.sval_bot[1] = 1; P.sbc_top[1] = "N"; P.sval_top[1] = -1;
    // kappa = 1 (re = 1000 keeps the explicit viscous limit of the (zero) velocity away), nz = 8 for a mild
    // explicit z-diffusion limit; theta = 1 and large steps to the steady state
    P.re = 1000; P.nz = std::max(8, 4 * g_comm.size); P.lz = P.nz * M_PI / 4;  // dz = pi/4, >= 4 planes per rank
    for (int n = 0; n < 2; ++n) P.pr[n] = 1e-3;
    P.ssrc_const[1] = 1;
    P.fixed_dt = true; P.dt = 0.1;
    Solver S(P);
    zero_velocity(S);
    for (int n = 0; n < 2; ++n) S.init_scalar(n);
    while (S.time < 40) S.advance();
    const double kap = 1.0, Q = 2, a = 0.7 - Q / kap, b = -0.3 + Q / (2 * kap) - a;
    report("steady: Neumann bottom, Dirichlet top, source (quadratic profile)",
           maxerr(S, 0, [&](int i, int, int j) -> std::complex<double> {
             const double yy = S.y[size_t(j)];
             return i == 0 ? -Q / (2 * kap) * yy * yy + a * yy + b : 0.0;
           }), 1e-9);
    // scalar 1: -y^2/2 + c; c is not fixed by the problem (it changes slowly with the difference between the
    // discrete fluxes and the discrete integral of the source), compare with c from the centre value
    {
      std::vector<cplx> h = S.Sf[1].host();
      const int jc = S.ny / 2;
      const double c = double(h[size_t(idx(jc) * S.nz * S.nkx)].re) + 0.5 * S.y[size_t(jc)] * S.y[size_t(jc)];
      report("steady: Neumann both walls with a balancing source (shape)",
             maxerr(S, 1, [&](int i, int, int j) -> std::complex<double> {
               const double yy = S.y[size_t(j)];
               return i == 0 ? -0.5 * yy * yy + c : 0.0;
             }), 1e-9);
      if (root) std::printf("  (its mean drifted to %.2e at t = %.0f)\n", c - 1.0 / 6, S.time);
    }
  }

  // ---- 3. hydrostatic balance
  {
    Params P = base();
    P.nscal = 1; P.ri[0] = 2; P.sinit[0] = "linear"; P.sval_bot[0] = 1; P.sval_top[0] = -1;
    P.gravity[0] = 0; P.gravity[1] = -3; P.gravity[2] = 0;  // normalised internally
    P.re = 1000; P.fixed_dt = true; P.dt = 1e-2;
    Solver S(P);
    zero_velocity(S);
    S.init_scalar(0);
    for (int s = 0; s < 50; ++s) S.advance();
    double vm = 0;
    for (const DArray<cplx>* f : {&S.u, &S.v, &S.w})
      for (const cplx& c : f->host()) vm = std::isfinite(double(abs2(c))) ? std::max(vm, std::sqrt(double(abs2(c)))) : 1e300;
    vm = g_comm.allmax(vm);
    report("hydrostatic: max |u| after 50 steps (Ri = 2, stable stratification)", vm, 1e-12);
    report("hydrostatic: the linear profile is kept",
           maxerr(S, 0, [&](int i, int, int j) -> std::complex<double> { return i == 0 ? -S.y[size_t(j)] : 0.0; }),
           1e-12);
  }

  // ---- 4. the three forms of the advection term on a divergence-free flow (noise, re = 400, 40 steps)
  {
    std::vector<cplx> res[3];
    const char* form[3] = {"advective", "divergence", "skew"};
    for (int f = 0; f < 3; ++f) {
      Params P = base();
      P.nx = 16; P.ny = 49; P.nz = 32; P.re = 400; P.forcing = "cfr"; P.init_type = "noise"; P.init_amp = 0.3;
      P.nscal = 1; P.sinit[0] = "noise"; P.scal_form = form[f];
      P.fixed_dt = true; P.dt = 5e-3;
      Solver S(P);
      S.init_flow();
      for (int s = 0; s < 40; ++s) S.advance();
      res[f] = S.Sf[0].host();
    }
    for (int f = 1; f < 3; ++f) {
      double e = 0, m = 0;
      for (size_t a = 0; a < res[0].size(); ++a) {
        e = std::max(e, std::sqrt(double(abs2(res[f][a] - res[0][a]))));
        m = std::max(m, std::sqrt(double(abs2(res[0][a]))));
      }
      e = g_comm.allmax(e) / g_comm.allmax(m);
      char name[128];
      std::snprintf(name, sizeof(name), "scal_form %s vs advective (truncation error, 40 steps)", form[f]);
      report(name, e, 1e-2);
    }
  }

  if (root) std::printf(ok ? "scalars: OK\n" : "scalars: FAIL\n");
  g_comm.finalize();
  return ok ? 0 : 1;
}
