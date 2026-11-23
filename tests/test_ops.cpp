// Operator tests: order of accuracy of the y and z differences, quadrature, and the pressure band
// solver (no pivoting) against a dense LU with partial pivoting, over the whole range of k.
#include <cstdio>
#include <random>
#include <string>

#include "../src/operators.hpp"
#include "../src/psolve.hpp"

using namespace fd;

static double maxerr_d(const std::vector<double>& y, int der, int p) {
  RowOp D = make_yop(y, der, p);
  std::vector<double> f(y.size()), ex(y.size());
  for (size_t j = 0; j < y.size(); ++j) {
    const double t = y[j];
    f[j] = std::sin(2.3 * t + 0.4) + std::exp(t);
    ex[j] = der == 1 ? 2.3 * std::cos(2.3 * t + 0.4) + std::exp(t) : -5.29 * std::sin(2.3 * t + 0.4) + std::exp(t);
  }
  std::vector<double> g = D.apply(f);
  double e = 0;
  for (size_t j = 0; j < y.size(); ++j) e = std::max(e, std::abs(g[j] - ex[j]));
  return e;
}

// dense LU with partial pivoting
static std::vector<double> dense_solve(std::vector<double> M, std::vector<double> b, int n) {
  std::vector<int> piv(static_cast<size_t>(n));
  for (int k = 0; k < n; ++k) {
    int p = k;
    for (int i = k + 1; i < n; ++i)
      if (std::abs(M[size_t(i) * n + k]) > std::abs(M[size_t(p) * n + k])) p = i;
    if (p != k) {
      for (int c = 0; c < n; ++c) std::swap(M[size_t(k) * n + c], M[size_t(p) * n + c]);
      std::swap(b[size_t(k)], b[size_t(p)]);
    }
    for (int i = k + 1; i < n; ++i) {
      const double l = M[size_t(i) * n + k] / M[size_t(k) * n + k];
      if (l == 0) continue;
      for (int c = k; c < n; ++c) M[size_t(i) * n + c] -= l * M[size_t(k) * n + c];
      b[size_t(i)] -= l * b[size_t(k)];
    }
  }
  for (int i = n - 1; i >= 0; --i) {
    double s = b[size_t(i)];
    for (int c = i + 1; c < n; ++c) s -= M[size_t(i) * n + c] * b[size_t(c)];
    b[size_t(i)] = s / M[size_t(i) * n + i];
  }
  return b;
}

template <int K>
static void pressure_test(const std::vector<double>& y, int p, const BandMat& B) {
  const int n = int(y.size());
  std::vector<real> Ap(size_t(n) * (2 * K + 1), 0);
  for (int j = 0; j < n; ++j)
    for (int d = 0; d < B.width(); ++d) Ap[size_t(j) * (2 * K + 1) + d - B.kl + K] = real(B.a[size_t(j * B.width() + d)]);
  std::mt19937 rng(1);
  std::uniform_real_distribution<double> U(-1, 1);
  std::vector<double> rhs(size_t(n) * NRHS), x(size_t(n) * NRHS), su(size_t(n) * (K + 1)), sy(size_t(n) * NRHS);
  for (auto& v : rhs) v = U(rng);
  double worst = 0;
  for (double k : {1e-4, 1e-2, 0.1, 0.5, 1.0, 3.0, 10.0, 100.0, 1e3, 1e4}) {
    const double k2 = k * k;
    band_solve<K>(
        n, Ap.data(), real(k2), [&](int j, int c) { return real(rhs[size_t(j * NRHS + c)]); },
        [&](int j, int c, real v) { x[size_t(j * NRHS + c)] = v; },
        [&](int j, int m) -> real& { return reinterpret_cast<real&>(su[size_t(j * (K + 1) + m)]); },
        [&](int j, int c) -> real& { return reinterpret_cast<real&>(sy[size_t(j * NRHS + c)]); });
    // dense reference
    std::vector<double> M(size_t(n) * n, 0.0);
    for (int j = 0; j < n; ++j) {
      for (int c = std::max(0, j - B.kl); c <= std::min(n - 1, j + B.ku); ++c)
        M[size_t(j) * n + c] = B.a[size_t(j * B.width() + c - j + B.kl)];
      if (j > 0 && j < n - 1) M[size_t(j) * n + j] -= k2;
    }
    double rel = 0, res = 0, bn = 0;
    for (int c = 0; c < NRHS; ++c) {
      std::vector<double> b(static_cast<size_t>(n));
      for (int j = 0; j < n; ++j) b[size_t(j)] = rhs[size_t(j * NRHS + c)];
      std::vector<double> xr = dense_solve(M, b, n);
      double xn = 0, dmax = 0;
      for (int j = 0; j < n; ++j) {
        xn = std::max(xn, std::abs(xr[size_t(j)]));
        dmax = std::max(dmax, std::abs(xr[size_t(j)] - x[size_t(j * NRHS + c)]));
        double r = -b[size_t(j)];
        for (int cc = 0; cc < n; ++cc) r += M[size_t(j) * n + cc] * x[size_t(cc * NRHS + c)];
        res = std::max(res, std::abs(r));
        bn = std::max(bn, std::abs(b[size_t(j)]));
      }
      rel = std::max(rel, dmax / xn);
    }
    worst = std::max(worst, rel);
    std::printf("    k = %8.1e   |x - x_dense|/|x| = %8.1e   residual/|b| = %8.1e\n", k, rel, res / bn);
  }
  std::printf("  ny = %d, p = %d, kl = %d, ku = %d: worst relative difference %.1e\n", n, p, B.kl, B.ku, worst);
}

static std::string fmt(double v) { char b[32]; std::snprintf(b, sizeof(b), "%.1e", v); return b; }

int main() {
  std::printf("Order of accuracy on a tanh grid (alpha = 3), max error over all points incl. walls\n");
  for (int p : {2, 4, 6, 8}) {
    std::printf(" p = %d\n", p);
    double e1o = 0, e2o = 0;
    for (int ny : {33, 65, 129, 257}) {
      std::vector<double> y = make_ygrid(ny, "tanh", 3.0);
      const double e1 = maxerr_d(y, 1, p), e2 = maxerr_d(y, 2, p);
      std::vector<double> q = make_quadrature(y, p);
      double I = 0;
      for (int j = 0; j < ny; ++j) I += q[size_t(j)] * (std::sin(2.3 * y[size_t(j)] + 0.4) + std::exp(y[size_t(j)]));
      const double Iex = (-std::cos(2.7) + std::cos(-1.9)) / 2.3 + std::exp(1.0) - std::exp(-1.0);
      std::printf("   ny = %4d  D1 %8.2e (rate %4.1f)  D2 %8.2e (rate %4.1f)  quadrature %8.2e\n", ny, e1,
                  e1o > 0 ? std::log2(e1o / e1) : 0.0, e2, e2o > 0 ? std::log2(e2o / e2) : 0.0, std::abs(I - Iex));
      e1o = e1;
      e2o = e2;
    }
  }
  std::printf("\nModified wavenumbers (z, central): max k'h of D1 and max |symbol| h^2 of D2\n");
  for (int p : {2, 4, 6, 8}) {
    std::vector<double> a1 = central_weights(1, p), a2 = central_weights(2, p);
    double m1 = 0, m2 = 0;
    for (int t = 0; t <= 1000; ++t) {
      const double th = M_PI * t / 1000;
      double s1 = 0, s2 = a2[size_t(p / 2)];
      for (int l = 1; l <= p / 2; ++l) {
        s1 += 2 * a1[size_t(p / 2 + l)] * std::sin(l * th);
        s2 += 2 * a2[size_t(p / 2 + l)] * std::cos(l * th);
      }
      m1 = std::max(m1, s1);
      m2 = std::max(m2, -s2);
    }
    std::printf("  p = %d: %.4f  %.4f\n", p, m1, m2);
  }
  std::printf("\nPressure band solver (no pivoting) vs dense LU with pivoting\n");
  for (int p : {2, 4, 6})
    for (int ny : {33, 129, 385}) {
      std::vector<double> y = make_ygrid(ny, "tanh", 3.0);
      BandMat B = make_pressure_op(make_yop(y, 1, p));
      const int K = std::max(B.kl, B.ku);
      if (K <= 3) pressure_test<3>(y, p, B);
      else if (K <= 6) pressure_test<6>(y, p, B);
      else if (K <= 9) pressure_test<9>(y, p, B);
      else std::printf("K = %d not instantiated\n", K);
    }

  std::printf("\nCompact y differences on a tanh grid (alpha = 3), max error over all points incl. walls\n");
  for (int p : {4, 6, 8}) {
    double e1o = 0, e2o = 0;
    for (int ny : {33, 65, 129, 257}) {
      std::vector<double> y = make_ygrid(ny, "tanh", 3.0);
      double e[2];
      for (int der = 1; der <= 2; ++der) {
        YDeriv D = make_compact(y, der, p);
        std::vector<double> f(y.size()), ex(y.size());
        for (size_t j = 0; j < y.size(); ++j) {
          const double t = y[j];
          f[j] = std::sin(2.3 * t + 0.4) + std::exp(t);
          ex[j] = der == 1 ? 2.3 * std::cos(2.3 * t + 0.4) + std::exp(t) : -5.29 * std::sin(2.3 * t + 0.4) + std::exp(t);
        }
        std::vector<double> g = D.apply(f);
        double m = 0;
        for (size_t j = 0; j < y.size(); ++j) m = std::max(m, std::abs(g[j] - ex[j]));
        e[der - 1] = m;
      }
      std::printf("  p = %d ny = %4d  D1 %8.2e (rate %4.1f)  D2 %8.2e (rate %4.1f)\n", p, ny, e[0],
                  e1o > 0 ? std::log2(e1o / e[0]) : 0.0, e[1], e2o > 0 ? std::log2(e2o / e[1]) : 0.0);
      e1o = e[0];
      e2o = e[1];
    }
  }

  std::printf("\nPressure system (phi, D1 phi), banded LU with pivoting vs dense LU\n");
  for (std::string sch : {"compact", "explicit"})
    for (int p : {4, 6, 8})
      for (int ny : {33, 129, 385}) {
        std::vector<double> y = make_ygrid(ny, "tanh", 3.0);
        YDeriv D1 = sch == "compact" ? make_compact(y, 1, p) : make_explicit(y, 1, p);
        BandMat G0, G1;
        make_pressure_system(D1, G0, G1);
        const int N = 2 * ny, K = std::max(G0.kl, G0.ku);
        constexpr int KK = 16;
        if (K > KK) { std::printf("K = %d too large\n", K); continue; }
        std::vector<real> g0(size_t(N) * (2 * KK + 1), 0), g1 = g0;
        for (int r = 0; r < N; ++r)
          for (int d = 0; d < G0.width(); ++d) {
            g0[size_t(r) * (2 * KK + 1) + d - G0.kl + KK] = real(G0.a[size_t(r * G0.width() + d)]);
            g1[size_t(r) * (2 * KK + 1) + d - G0.kl + KK] = real(G1.a[size_t(r * G1.width() + d)]);
          }
        std::mt19937 rng(3);
        std::uniform_real_distribution<double> U(-1, 1);
        std::vector<double> rhs(size_t(N) * NRHS, 0.0), x(size_t(N) * NRHS), su(size_t(N) * (2 * KK + 1)), sy(size_t(N) * NRHS);
        for (int j = 0; j < ny; ++j)
          for (int c = 0; c < NRHS; ++c) rhs[size_t((2 * j + 1) * NRHS + c)] = U(rng);
        double worst = 0, worst_div = 0;
        // band_factor_piv + band_solve_factored: the same operations as band_solve_piv (identical results when
        // the compiler does not fuse multiply-adds differently in the two; otherwise round-off differences)
        long fact_mismatch = 0;
        double fact_maxrel = 0;
        std::vector<double> xf(size_t(N) * NRHS), suf(size_t(N) * (2 * KK + 1)), slf(size_t(N) * KK), syf(size_t(N) * NRHS);
        std::vector<unsigned char> spf(static_cast<size_t>(N));
        for (double k : {0.1, 0.5, 1.0, 10.0, 100.0, 1e3, 1e4}) {
          const double k2 = k * k;
          band_solve_piv<KK, KK>(
              N, g0.data(), g1.data(), real(k2), [&](int r, int c) { return real(rhs[size_t(r * NRHS + c)]); },
              [&](int r, int c, real v) { x[size_t(r * NRHS + c)] = v; },
              [&](int r, int m) -> real& { return su[size_t(r * (2 * KK + 1) + m)]; },
              [&](int r, int c) -> real& { return sy[size_t(r * NRHS + c)]; });
          band_factor_piv<KK, KK>(
              N, g0.data(), g1.data(), real(k2), [&](int r, int m) -> real& { return reinterpret_cast<real&>(suf[size_t(r * (2 * KK + 1) + m)]); },
              [&](int r, int q) -> real& { return reinterpret_cast<real&>(slf[size_t(r * KK + q)]); },
              [&](int r) -> unsigned char& { return spf[size_t(r)]; });
          band_solve_factored<KK, KK>(
              N, [&](int r, int c) { return real(rhs[size_t(r * NRHS + c)]); },
              [&](int r, int c, real v) { xf[size_t(r * NRHS + c)] = v; },
              [&](int r, int m) -> real { return reinterpret_cast<real&>(suf[size_t(r * (2 * KK + 1) + m)]); },
              [&](int r, int q) -> real { return reinterpret_cast<real&>(slf[size_t(r * KK + q)]); },
              [&](int r) -> int { return spf[size_t(r)]; },
              [&](int r, int c) -> real& { return reinterpret_cast<real&>(syf[size_t(r * NRHS + c)]); });
          {
            double xmax = 0, dmax = 0;
            for (size_t m = 0; m < xf.size(); ++m) {
              fact_mismatch += (xf[m] != x[m]);
              xmax = std::max(xmax, std::abs(x[m]));
              dmax = std::max(dmax, std::abs(xf[m] - x[m]));
            }
            fact_maxrel = std::max(fact_maxrel, dmax / xmax);
          }
          std::vector<double> M(size_t(N) * N, 0.0);
          for (int r = 0; r < N; ++r)
            for (int c = std::max(0, r - G0.kl); c <= std::min(N - 1, r + G0.ku); ++c)
              M[size_t(r) * N + c] = G0.a[size_t(r * G0.width() + c - r + G0.kl)] + k2 * G1.a[size_t(r * G1.width() + c - r + G1.kl)];
          for (int c = 0; c < NRHS; ++c) {
            std::vector<double> b(static_cast<size_t>(N)), xs(static_cast<size_t>(N));
            for (int r = 0; r < N; ++r) { b[size_t(r)] = rhs[size_t(r * NRHS + c)]; xs[size_t(r)] = x[size_t(r * NRHS + c)]; }
            std::vector<double> xr = dense_solve(M, b, N);
            double xn = 0, dm = 0;
            for (int j = 0; j < ny; ++j) {
              xn = std::max(xn, std::abs(xr[size_t(2 * j)]));
              dm = std::max(dm, std::abs(xr[size_t(2 * j)] - xs[size_t(2 * j)]));
            }
            worst = std::max(worst, dm / xn);
            // exactness: psi must equal D1 phi (applied directly), relative to |psi|
            std::vector<double> ph(static_cast<size_t>(ny)), ps(static_cast<size_t>(ny));
            for (int j = 0; j < ny; ++j) { ph[size_t(j)] = xs[size_t(2 * j)]; ps[size_t(j)] = xs[size_t(2 * j + 1)]; }
            std::vector<double> dph = D1.apply(ph);
            double pn = 0, pd = 0;
            for (int j = 0; j < ny; ++j) { pn = std::max(pn, std::abs(ps[size_t(j)])); pd = std::max(pd, std::abs(ps[size_t(j)] - dph[size_t(j)])); }
            worst_div = std::max(worst_div, pd / pn);
          }
        }
        std::printf("  %-8s p = %d ny = %3d kl = %d ku = %d: |phi - phi_dense|/|phi| %.1e, |psi - D1 phi|/|psi| %.1e, "
                    "factored once: %s\n",
                    sch.c_str(), p, ny, G0.kl, G0.ku, worst, worst_div,
                    fact_mismatch == 0 ? "identical" : (std::string("max|diff|/max|x| ") + fmt(fact_maxrel)).c_str());
      }
  return 0;
}
