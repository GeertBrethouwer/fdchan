// Test of the compact z-derivatives (z_scheme = compact) and their distributed solve over the z-slabs.
//
//   make CXX=g++ build/test_zcompact && build/test_zcompact                  (one rank)
//   make MPI=1 MPICXX=mpicxx build/test_zcompact && mpirun -n 4 build/test_zcompact
//
// f = A(kx, y) (cos(m1 z) + 0.5 sin(m2 z)), z in [0, 2 pi). For each order (4, 6, 8):
//   * "vs scheme": D1z f and D2z f against the exact answer of the periodic compact scheme (its modified
//     wavenumbers times the modes): round-off on any number of ranks, if the distributed solve is right;
//   * "vs exact derivative": against the true derivatives, which must converge at the order of the scheme.
#include <cstdio>

#include "../src/solver.hpp"

namespace fd {
#ifdef FD_GPU
gpuStream_t g_stream;
#endif
Comm g_comm;
}  // namespace fd

using namespace fd;

int main(int argc, char** argv) {
  g_comm.init(&argc, &argv);
#ifdef FD_GPU
  gpu_check(gpuStreamCreate(&g_stream), "stream");
#else
  Pool::get().set_threads(1);
#endif
  const bool root = g_comm.root();
  if (root) std::printf("compact z-derivatives on %d rank(s)\n", g_comm.size);
  const int m1 = 3, m2 = 5;
  bool ok = true;
  for (int order : {4, 6, 8}) {
    double prev1 = 0, prev2 = 0;
    for (int npr : {64, 128, 256}) {  // z-planes per rank (the multi-rank solve needs enough, see setup_compact_z)
      const int nzg = npr * g_comm.size + (g_comm.size > 1 ? 1 : 0);  // uneven slabs on several ranks
      Params P;
      P.nx = 8; P.ny = 17; P.nz = nzg; P.lx = 2 * M_PI; P.lz = 2 * M_PI;
      P.fd_order = 6; P.fd_order_z = order; P.z_scheme = "compact";
      Solver S(P);
      const double h = P.lz / nzg;
      // compact scheme symbols (the same coefficients as the solver)
      double al[2];
      std::vector<double> c[2];
      compact_coeffs(1, order, al[0], c[0]);
      compact_coeffs(2, order, al[1], c[1]);
      auto k1 = [&](int m) {
        double s = 0;
        for (size_t l = 1; l <= c[0].size(); ++l) s += 2 * c[0][l - 1] * std::sin(double(l) * m * h);
        return s / (1 + 2 * al[0] * std::cos(m * h)) / h;
      };
      auto k2 = [&](int m) {
        double s = 0;
        for (size_t l = 1; l <= c[1].size(); ++l) s += 2 * c[1][l - 1] * (1 - std::cos(double(l) * m * h));
        return s / (1 + 2 * al[1] * std::cos(m * h)) / (h * h);
      };
      std::vector<cplx> hu(static_cast<size_t>(S.nsp));
      auto A = [&](int i, int j) { return cplx(real(1 + 0.1 * i), real(0.5 - 0.05 * j)); };
      for (int j = 0; j < S.ny; ++j)
        for (int k = 0; k < S.nz; ++k)
          for (int i = 0; i < S.nkx; ++i) {
            const double z = (S.z0 + k) * h;
            hu[size_t((idx(j) * S.nz + k) * S.nkx + i)] = real(std::cos(m1 * z) + 0.5 * std::sin(m2 * z)) * A(i, j);
          }
      S.u.upload(hu);
      S.halo(S.u.p, S.nkx, S.hbA.p);
      S.zderiv_full(0, S.u.p, S.nkx, S.hbA.p, S.DZ1.p, S.ZB1.p);
      S.zderiv_full(1, S.u.p, S.nkx, S.hbA.p, S.DZ2.p, S.ZB2.p);
      sync();
      std::vector<cplx> d1 = S.DZ1.host(), d2 = S.DZ2.host();
      double es1 = 0, es2 = 0, ee1 = 0, ee2 = 0, n1 = 0, n2 = 0;
      for (int j = 0; j < S.ny; ++j)
        for (int k = 0; k < S.nz; ++k)
          for (int i = 0; i < S.nkx; ++i) {
            const double z = (S.z0 + k) * h;
            const size_t n = size_t((idx(j) * S.nz + k) * S.nkx + i);
            const cplx a = A(i, j);
            const cplx s1 = real(-k1(m1) * std::sin(m1 * z) + 0.5 * k1(m2) * std::cos(m2 * z)) * a;
            const cplx s2 = real(-k2(m1) * std::cos(m1 * z) - 0.5 * k2(m2) * std::sin(m2 * z)) * a;
            const cplx t1 = real(-m1 * std::sin(m1 * z) + 0.5 * m2 * std::cos(m2 * z)) * a;
            const cplx t2 = real(-m1 * m1 * std::cos(m1 * z) - 0.5 * m2 * m2 * std::sin(m2 * z)) * a;
            es1 = std::max(es1, std::sqrt(double(abs2(d1[n] - s1))));
            es2 = std::max(es2, std::sqrt(double(abs2(d2[n] - s2))));
            ee1 = std::max(ee1, std::sqrt(double(abs2(d1[n] - t1))));
            ee2 = std::max(ee2, std::sqrt(double(abs2(d2[n] - t2))));
            n1 = std::max(n1, std::sqrt(double(abs2(t1))));
            n2 = std::max(n2, std::sqrt(double(abs2(t2))));
          }
      es1 = g_comm.allmax(es1 / n1); es2 = g_comm.allmax(es2 / n2);
      ee1 = g_comm.allmax(ee1 / n1); ee2 = g_comm.allmax(ee2 / n2);
      // round-off of a difference quotient grows as 1/h (D1) and 1/h^2 (D2), relative to the derivative
      const double eps = double(std::numeric_limits<real>::epsilon());
      const bool good = es1 < 20 * eps / h && es2 < 20 * eps / (h * h);
      ok = ok && good;
      if (root) {
        std::printf("  order %d, nz = %3d (%d..%d planes per rank): vs scheme %.1e %.1e %s; vs exact derivative %.2e %.2e",
                    order, nzg, S.zst[1] - S.zst[0], S.zst[size_t(S.nrank)] - S.zst[size_t(S.nrank - 1)], es1, es2,
                    good ? "ok" : "WRONG", ee1, ee2);
        if (prev1 > 0) std::printf("  (orders %.2f %.2f)", std::log2(prev1 / ee1), std::log2(prev2 / ee2));
        std::printf("\n");
      }
      prev1 = ee1; prev2 = ee2;
    }
  }
  if (root) std::printf(ok ? "compact z: OK\n" : "compact z: FAIL\n");
  g_comm.finalize();
  return ok ? 0 : 1;
}
