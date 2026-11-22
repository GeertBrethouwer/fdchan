// Finite-difference operators (host side, set up once).
//
//   y: explicit (non-compact) finite differences of order p (2, 4, 6, 8) computed directly on the
//      non-uniform grid (Fornberg's algorithm). Interior rows are central (p+1 points); the rows near
//      the walls use one-sided stencils with p+1 points (first derivative) or p+2 points (second
//      derivative), so every row is of order p.
//   z: central differences of order p on the uniform periodic grid; their Fourier symbols are the
//      modified wavenumbers used by the pressure solver.
#pragma once
#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include "backend.hpp"

namespace fd {

// Fornberg (1988): weights c[k*n + j] of the k-th derivative at z0 from the values at x[0..n-1]
inline std::vector<double> fornberg(double z0, const double* x, int n, int m) {
  std::vector<double> c(size_t((m + 1) * n), 0.0);
  double c1 = 1, c4 = x[0] - z0;
  c[0] = 1;
  for (int i = 1; i < n; ++i) {
    const int mn = std::min(i, m);
    double c2 = 1;
    const double c5 = c4;
    c4 = x[i] - z0;
    for (int j = 0; j < i; ++j) {
      const double c3 = x[i] - x[j];
      c2 *= c3;
      if (j == i - 1) {
        for (int k = mn; k >= 1; --k)
          c[size_t(k * n + i)] = c1 * (k * c[size_t((k - 1) * n + i - 1)] - c5 * c[size_t(k * n + i - 1)]) / c2;
        c[size_t(i)] = -c1 * c5 * c[size_t(i - 1)] / c2;
      }
      for (int k = mn; k >= 1; --k)
        c[size_t(k * n + j)] = (c4 * c[size_t(k * n + j)] - k * c[size_t((k - 1) * n + j)]) / c3;
      c[size_t(j)] = c4 * c[size_t(j)] / c3;
    }
    c1 = c2;
  }
  return c;
}

// Banded operator with a fixed number W of coefficients per row: (D f)_j = sum_m c[j*W+m] f[s[j]+m]
struct RowOp {
  int n = 0, W = 0;
  std::vector<int> s;
  std::vector<double> c;
  double coef(int j, int col) const {
    const int m = col - s[size_t(j)];
    return (m >= 0 && m < W) ? c[size_t(j * W + m)] : 0.0;
  }
  std::vector<double> apply(const std::vector<double>& f) const {
    std::vector<double> g(size_t(n), 0.0);
    for (int j = 0; j < n; ++j)
      for (int m = 0; m < W; ++m) g[size_t(j)] += c[size_t(j * W + m)] * f[size_t(s[size_t(j)] + m)];
    return g;
  }
};

// derivative of order `der` with interior order p on the grid y (non-periodic)
inline RowOp make_yop(const std::vector<double>& y, int der, int p) {
  const int n = int(y.size());
  RowOp D;
  D.n = n;
  const int wi = p + 1;                    // interior (central) width
  const int wb = (der == 1) ? p + 1 : p + 2;  // boundary (one-sided) width
  D.W = std::max(wi, wb);
  if (n < D.W) fatal("ny too small for the finite-difference order");
  D.s.resize(size_t(n));
  D.c.assign(size_t(n * D.W), 0.0);
  for (int j = 0; j < n; ++j) {
    int a, w;
    if (j - p / 2 >= 0 && j + p / 2 <= n - 1) {
      a = j - p / 2;
      w = wi;
    } else {
      w = wb;
      a = std::clamp(j - w / 2, 0, n - w);
    }
    const int s = std::clamp(a, 0, n - D.W);
    D.s[size_t(j)] = s;
    std::vector<double> c = fornberg(y[size_t(j)], &y[size_t(a)], w, der);
    for (int m = 0; m < w; ++m) D.c[size_t(j * D.W + (a - s) + m)] = c[size_t(der * w + m)];
  }
  return D;
}

// central weights a[l], l = -p/2..p/2 (stored at a[l+p/2]) of derivative `der` on a unit grid
inline std::vector<double> central_weights(int der, int p) {
  std::vector<double> x(size_t(p + 1));
  for (int l = 0; l <= p; ++l) x[size_t(l)] = l - p / 2;
  std::vector<double> c = fornberg(0.0, x.data(), p + 1, der);
  return std::vector<double>(c.begin() + der * (p + 1), c.begin() + (der + 1) * (p + 1));
}

// wall-normal grid on [-1, 1], ny points including the walls
inline std::vector<double> make_ygrid(int ny, const std::string& type, double alpha) {
  std::vector<double> y(static_cast<size_t>(ny));
  for (int j = 0; j < ny; ++j) {
    const double xi = double(j) / (ny - 1);
    if (type == "tanh")
      y[size_t(j)] = std::tanh(alpha * (xi - 0.5)) / std::tanh(0.5 * alpha);
    else if (type == "uniform")
      y[size_t(j)] = -1.0 + 2.0 * xi;
    else if (type == "cosine")
      y[size_t(j)] = -std::cos(M_PI * xi);
    else
      fatal("grid_type must be tanh, uniform or cosine");
  }
  y[0] = -1.0;
  y[size_t(ny - 1)] = 1.0;
  return y;
}

// quadrature weights: on each interval the local interpolation polynomial through p+2 points is
// integrated exactly (Gauss-Legendre), so the rule is of order p+2 on smooth grids
inline std::vector<double> make_quadrature(const std::vector<double>& y, int p) {
  const int n = int(y.size()), w = std::min(p + 2, n);
  const double gx[4] = {-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526};
  const double gw[4] = {0.3478548451374538, 0.6521451548625461, 0.6521451548625461, 0.3478548451374538};
  std::vector<double> q(size_t(n), 0.0);
  for (int j = 0; j + 1 < n; ++j) {
    const int a = std::clamp(j - (w - 2) / 2, 0, n - w);
    const double h = y[size_t(j + 1)] - y[size_t(j)];
    for (int g = 0; g < 4; ++g) {
      // 4-point Gauss rule, applied on two halves of the interval (exact to degree 7 on each)
      for (int half = 0; half < 2; ++half) {
        const double lo = y[size_t(j)] + 0.5 * h * half;
        const double yy = lo + 0.25 * h * (1.0 + gx[g]);
        std::vector<double> L = fornberg(yy, &y[size_t(a)], w, 0);
        for (int m = 0; m < w; ++m) q[size_t(a + m)] += 0.25 * h * gw[g] * L[size_t(m)];
      }
    }
  }
  return q;
}

// Pressure operator A = D1 E D1 (E sets the wall rows to zero), as a band matrix
// band[j*(kl+ku+1) + (col - j + kl)]
struct BandMat {
  int n = 0, kl = 0, ku = 0;
  std::vector<double> a;
  int width() const { return kl + ku + 1; }
};

// ------------------------------------------------------------------ y derivatives A f^(d) = B f
// A is tridiagonal with unit diagonal (lo[j] f_{j-1} + f_j + up[j] f_{j+1}); B is a RowOp.
// Explicit schemes have lo = up = 0. Coefficients are computed on the stretched grid by requiring
// exactness for polynomials of the highest possible degree (as many conditions as unknowns).
struct YDeriv {
  int n = 0, der = 1;
  bool compact = false;
  std::vector<double> lo, up;  // A
  RowOp B;
  // Thomas factors of A: m[j] (multiplier), id[j] = 1/pivot, up[j] above
  std::vector<double> m, id;
  void factor() {
    m.assign(size_t(n), 0.0);
    id.assign(size_t(n), 1.0);
    double d = 1.0;
    id[0] = 1.0;
    for (int j = 1; j < n; ++j) {
      m[size_t(j)] = lo[size_t(j)] / d;
      d = 1.0 - m[size_t(j)] * up[size_t(j - 1)];
      if (std::abs(d) < 1e-12) fatal("compact scheme: zero pivot in A");
      id[size_t(j)] = 1.0 / d;
    }
  }
  std::vector<double> apply(const std::vector<double>& f) const {
    std::vector<double> g = B.apply(f);
    if (!compact) return g;
    for (int j = 1; j < n; ++j) g[size_t(j)] -= m[size_t(j)] * g[size_t(j - 1)];
    g[size_t(n - 1)] *= id[size_t(n - 1)];
    for (int j = n - 2; j >= 0; --j) g[size_t(j)] = (g[size_t(j)] - up[size_t(j)] * g[size_t(j + 1)]) * id[size_t(j)];
    return g;
  }
  // dense matrix of A^-1 B (tests, small n)
};

// x^q and d^der/dx^der of x^q, in long double. Plain host functions, not lambdas: C++17 lambdas are implicitly
// constexpr, and with nvcc --expt-relaxed-constexpr they are also compiled for the device, where long double is
// double (warning #20208-D). These run only on the host, in the set-up.
inline long double pow_ld(long double x, int q) { long double p = 1; for (int i = 0; i < q; ++i) p *= x; return p; }
inline long double dpow_ld(long double x, int q, int der) {
  if (q < der) return 0;
  long double c = 1;
  for (int i = 0; i < der; ++i) c *= (q - i);
  return c * pow_ld(x, q - der);
}

// Solve for one row: unknown A off-diagonals at the indices `ia`, B weights on `ib` (contiguous)
inline void compact_row(const std::vector<double>& y, int j, int der, const std::vector<int>& ia,
                        const std::vector<int>& ib, std::vector<double>& alpha, std::vector<double>& beta) {
  const int na = int(ia.size()), nb = int(ib.size()), Q = na + nb;
  // local scaling
  const int jn = (j + 1 < int(y.size())) ? j + 1 : j - 1;
  const double h = std::abs(y[size_t(jn)] - y[size_t(j)]);
  std::vector<long double> M(static_cast<size_t>(Q * Q)), r(static_cast<size_t>(Q));
  for (int q = 0; q < Q; ++q) {
    for (int a = 0; a < na; ++a) M[size_t(q * Q + a)] = -dpow_ld((y[size_t(ia[size_t(a)])] - y[size_t(j)]) / h, q, der);
    for (int b = 0; b < nb; ++b) M[size_t(q * Q + na + b)] = pow_ld((y[size_t(ib[size_t(b)])] - y[size_t(j)]) / h, q);
    r[size_t(q)] = dpow_ld(0.0L, q, der);
  }
  // Gaussian elimination with partial pivoting
  for (int k = 0; k < Q; ++k) {
    int p = k;
    for (int i = k + 1; i < Q; ++i)
      if (std::fabs((double)M[size_t(i * Q + k)]) > std::fabs((double)M[size_t(p * Q + k)])) p = i;
    for (int c = 0; c < Q; ++c) std::swap(M[size_t(k * Q + c)], M[size_t(p * Q + c)]);
    std::swap(r[size_t(k)], r[size_t(p)]);
    for (int i = k + 1; i < Q; ++i) {
      const long double l = M[size_t(i * Q + k)] / M[size_t(k * Q + k)];
      for (int c = k; c < Q; ++c) M[size_t(i * Q + c)] -= l * M[size_t(k * Q + c)];
      r[size_t(i)] -= l * r[size_t(k)];
    }
  }
  for (int i = Q - 1; i >= 0; --i) {
    long double s = r[size_t(i)];
    for (int c = i + 1; c < Q; ++c) s -= M[size_t(i * Q + c)] * r[size_t(c)];
    r[size_t(i)] = s / M[size_t(i * Q + i)];
  }
  alpha.assign(size_t(na), 0.0);
  beta.assign(size_t(nb), 0.0);
  for (int a = 0; a < na; ++a) alpha[size_t(a)] = double(r[size_t(a)]);
  for (int b = 0; b < nb; ++b) beta[size_t(b)] = double(r[size_t(na + b)] / pow_ld(h, der));
}

// Compact schemes, rows given for the bottom wall (the top wall is mirrored). A is tridiagonal.
//
// p = 4, 6 with closure "hichan" (default): exactly the stencils of HiCHAN (Lenaers et al. 2014),
//   (3-4-3) / (3-5-6-5-3):
//   wall row: A {0,1}, B 0..2 (D1) or B 0..3 (D2), 3rd order
//   p = 6, row 1: A {0,1,2}, B 0..3 (D1) or B 0..4 (D2), 5th order
//   interior: B j-2..j+2 (p = 6) or j-1..j+1 (p = 4)
// closure "explicit_wall": wall row explicit one-sided, B 0..4 (D1) or 0..5 (D2); near-wall rows
//   A {j-1,j,j+1}, B j-1..j+2.
// p = 8 (always "explicit_wall"): interior B j-3..j+3 (8th order on a uniform grid), near-wall rows 1 and 2
//   as above. In the Tollmien-Schlichting test (ny = 65) it is about 15x more accurate than p = 6 "hichan".
// Coefficients: exactness for polynomials of the highest possible degree, on the actual grid.
inline YDeriv make_compact(const std::vector<double>& y, int der, int p, const std::string& closure = "hichan") {
  const int n = int(y.size());
  if (p != 4 && p != 6 && p != 8) fatal("compact y differences: fd_order must be 4, 6 or 8");
  if (closure != "hichan" && closure != "explicit_wall") fatal("y_closure must be hichan or explicit_wall");
  const bool hc = (closure == "hichan") && p != 8;
  YDeriv D;
  D.n = n;
  D.der = der;
  D.compact = true;
  D.lo.assign(size_t(n), 0.0);
  D.up.assign(size_t(n), 0.0);
  const int hb = p / 2 - 2;  // 0 (p = 4), 1 (p = 6), 2 (p = 8)
  D.B.n = n;
  D.B.W = std::max(3 + 2 * hb, hc ? 5 : ((der == 1) ? 5 : 6));
  D.B.s.resize(size_t(n));
  D.B.c.assign(size_t(n * D.B.W), 0.0);
  if (n < D.B.W + 2) fatal("ny too small for the compact scheme");
  for (int j = 0; j < n; ++j) {
    const bool bot = j < n / 2;
    const int jj = bot ? j : n - 1 - j;  // distance from the nearest wall
    std::vector<int> ia, ib;
    auto mir = [&](int k) { return bot ? k : n - 1 - k; };
    if (jj == 0) {
      if (hc) {
        ia = {mir(1)};
        const int w = (der == 1) ? 3 : 4;
        for (int k = 0; k < w; ++k) ib.push_back(mir(k));
      } else {
        const int w = (der == 1) ? 5 : 6;
        for (int k = 0; k < w; ++k) ib.push_back(mir(k));
      }
    } else if (jj <= hb) {
      ia = {mir(jj - 1), mir(jj + 1)};
      const int last = (hc && der == 2) ? jj + 3 : jj + 2;
      for (int k = jj - 1; k <= last; ++k) ib.push_back(mir(k));
    } else {
      ia = {j - 1, j + 1};
      for (int k = j - 1 - hb; k <= j + 1 + hb; ++k) ib.push_back(k);
    }
    std::sort(ib.begin(), ib.end());
    std::vector<double> al, be;
    compact_row(y, j, der, ia, ib, al, be);
    for (size_t a = 0; a < ia.size(); ++a) {
      if (ia[a] == j - 1) D.lo[size_t(j)] = al[a];
      else D.up[size_t(j)] = al[a];
    }
    const int s = std::clamp(ib.front(), 0, n - D.B.W);
    D.B.s[size_t(j)] = s;
    for (size_t b = 0; b < ib.size(); ++b) D.B.c[size_t(j * D.B.W + ib[b] - s)] = be[b];
  }
  D.factor();
  return D;
}

inline YDeriv make_explicit(const std::vector<double>& y, int der, int p) {
  YDeriv D;
  D.n = int(y.size());
  D.der = der;
  D.compact = false;
  D.lo.assign(size_t(D.n), 0.0);
  D.up.assign(size_t(D.n), 0.0);
  D.B = make_yop(y, der, p);
  D.factor();
  return D;
}

// Pressure system for one (kx, kz), unknowns interleaved x = (phi_0, psi_0, phi_1, psi_1, ...), psi = D1 phi:
//   row 2j  :  (A1 psi - B1 phi)_j                  = 0
//   row 2j+1:  (B1 E psi)_j - k^2 (A1 P phi)_j      = rhs_j
// The matrix is G0 + k^2 G1; both are returned as band matrices with common kl, ku.
inline void make_pressure_system(const YDeriv& D1, BandMat& G0, BandMat& G1) {
  const int n = D1.n, N = 2 * n;
  std::vector<double> d0(size_t(N) * N, 0.0), d1(size_t(N) * N, 0.0);
  auto at = [&](std::vector<double>& d, int r, int c) -> double& { return d[size_t(r) * N + c]; };
  for (int j = 0; j < n; ++j) {
    // A1 psi
    at(d0, 2 * j, 2 * j + 1) += 1.0;
    if (j > 0) at(d0, 2 * j, 2 * (j - 1) + 1) += D1.lo[size_t(j)];
    if (j < n - 1) at(d0, 2 * j, 2 * (j + 1) + 1) += D1.up[size_t(j)];
    for (int m = 0; m < D1.B.W; ++m) {
      const int col = D1.B.s[size_t(j)] + m;
      const double c = D1.B.c[size_t(j * D1.B.W + m)];
      at(d0, 2 * j, 2 * col) -= c;                                         // - B1 phi
      if (col != 0 && col != n - 1) at(d0, 2 * j + 1, 2 * col + 1) += c;  // B1 E psi
    }
    // - k^2 A1 P phi
    auto addP = [&](int col, double a) {
      if (col > 0 && col < n - 1) at(d1, 2 * j + 1, 2 * col) -= a;
    };
    addP(j, 1.0);
    if (j > 0) addP(j - 1, D1.lo[size_t(j)]);
    if (j < n - 1) addP(j + 1, D1.up[size_t(j)]);
  }
  int kl = 0, ku = 0;
  for (int r = 0; r < N; ++r)
    for (int c = 0; c < N; ++c)
      if (at(d0, r, c) != 0.0 || at(d1, r, c) != 0.0) {
        kl = std::max(kl, r - c);
        ku = std::max(ku, c - r);
      }
  for (BandMat* B : {&G0, &G1}) {
    std::vector<double>& d = (B == &G0) ? d0 : d1;
    B->n = N;
    B->kl = kl;
    B->ku = ku;
    B->a.assign(size_t(N * B->width()), 0.0);
    for (int r = 0; r < N; ++r)
      for (int c = std::max(0, r - kl); c <= std::min(N - 1, r + ku); ++c) B->a[size_t(r * B->width() + c - r + kl)] = d[size_t(r) * N + c];
  }
}
inline BandMat make_pressure_op(const RowOp& D1) {
  const int n = D1.n;
  std::vector<double> dense(size_t(n) * n, 0.0);
  for (int j = 0; j < n; ++j)
    for (int m1 = 0; m1 < D1.W; ++m1) {
      const int m = D1.s[size_t(j)] + m1;
      if (m == 0 || m == n - 1) continue;  // E
      const double cjm = D1.c[size_t(j * D1.W + m1)];
      for (int m2 = 0; m2 < D1.W; ++m2) {
        const int col = D1.s[size_t(m)] + m2;
        dense[size_t(j) * n + col] += cjm * D1.c[size_t(m * D1.W + m2)];
      }
    }
  BandMat B;
  B.n = n;
  for (int j = 0; j < n; ++j)
    for (int c = 0; c < n; ++c)
      if (dense[size_t(j) * n + c] != 0.0) {
        B.kl = std::max(B.kl, j - c);
        B.ku = std::max(B.ku, c - j);
      }
  B.a.assign(size_t(n * B.width()), 0.0);
  for (int j = 0; j < n; ++j)
    for (int c = std::max(0, j - B.kl); c <= std::min(n - 1, j + B.ku); ++c)
      B.a[size_t(j * B.width() + c - j + B.kl)] = dense[size_t(j) * n + c];
  return B;
}

}  // namespace fd
