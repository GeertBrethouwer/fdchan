// Banded solver for the wall-normal pressure problem of one (kx, kz) pair.
//
//   M phi = s,   M = A - k^2 P,   A = D1 E D1,  P = identity on the interior rows, 0 on the wall rows.
//
// One thread per (kx, +-kz) pair: the matrix depends on kz only through kz'^2, so the two columns +kz
// and -kz share one factorisation (four real right-hand sides). LU without pivoting, computed on the fly
// in a single downward sweep. Only the last K rows of U and of the forward-substituted right-hand side are
// kept in registers; every U row is written once to a scratch array and read back in the upward sweep.
// The band of A is padded to K sub- and K super-diagonals: A[j*(2K+1) + (col - j + K)].
#pragma once
#include "backend.hpp"

namespace fd {

constexpr int NRHS = 4;

// rhs(j, c) / out(j, c, v): c = 0..3 (re/im of +kz, re/im of -kz)
// su(j, m): scratch for U(j, j+m), m = 0..K (stores 1/U(j,j) at m = 0); sy(j, c): scratch, 4 reals
template <int K, class Rhs, class Out, class SU, class SY>
HD inline void band_solve(int n, const real* A, real k2, Rhs rhs, Out out, SU su, SY sy) {
  real Uw[K][K + 1];
  real Yw[K][NRHS];
  for (int q = 0; q < K; ++q) {
    for (int m = 0; m <= K; ++m) Uw[q][m] = 0;
    for (int c = 0; c < NRHS; ++c) Yw[q][c] = 0;
  }
  for (int j = 0; j < n; ++j) {
    real r[2 * K + 1];
    const real* a = A + idx(j) * (2 * K + 1);
    for (int d = 0; d <= 2 * K; ++d) r[d] = a[d];
    if (j > 0 && j < n - 1) r[K] -= k2;
    real yv[NRHS];
    for (int c = 0; c < NRHS; ++c) yv[c] = rhs(j, c);
    // eliminate columns j-K .. j-1 with the stored rows (row i = j-K+q sits in Uw[q])
    for (int q = 0; q < K; ++q) {
      if (j - K + q < 0) continue;
      const real l = r[q] * Uw[q][0];  // Uw[q][0] = 1/U(i,i)
      for (int m = 1; m <= K; ++m)
        if (q + m <= 2 * K) r[q + m] -= l * Uw[q][m];
      for (int c = 0; c < NRHS; ++c) yv[c] -= l * Yw[q][c];
    }
    const real piv = real(1) / r[K];
    for (int q = 0; q < K - 1; ++q) {
      for (int m = 0; m <= K; ++m) Uw[q][m] = Uw[q + 1][m];
      for (int c = 0; c < NRHS; ++c) Yw[q][c] = Yw[q + 1][c];
    }
    Uw[K - 1][0] = piv;
    for (int m = 1; m <= K; ++m) Uw[K - 1][m] = r[K + m];
    for (int c = 0; c < NRHS; ++c) Yw[K - 1][c] = yv[c];
    for (int m = 0; m <= K; ++m) su(j, m) = Uw[K - 1][m];
    for (int c = 0; c < NRHS; ++c) sy(j, c) = yv[c];
  }
  // upward sweep, the last K solution values in Xw[0..K-1] (Xw[m-1] = x_{j+m})
  real Xw[K][NRHS];
  for (int q = 0; q < K; ++q)
    for (int c = 0; c < NRHS; ++c) Xw[q][c] = 0;
  for (int j = n - 1; j >= 0; --j) {
    real yv[NRHS];
    for (int c = 0; c < NRHS; ++c) yv[c] = sy(j, c);
    for (int m = 1; m <= K; ++m) {
      const real u = (j + m < n) ? su(j, m) : real(0);
      for (int c = 0; c < NRHS; ++c) yv[c] -= u * Xw[m - 1][c];
    }
    const real piv = su(j, 0);
    for (int q = K - 1; q > 0; --q)
      for (int c = 0; c < NRHS; ++c) Xw[q][c] = Xw[q - 1][c];
    for (int c = 0; c < NRHS; ++c) {
      Xw[0][c] = yv[c] * piv;
      out(j, c, Xw[0][c]);
    }
  }
}

// Banded LU with partial pivoting (as LAPACK gbtrf/gbtrs), on the fly, for G = G0 + k2*G1 of size N with
// KL sub- and KU super-diagonals: G0[r*(KL+KU+1) + (c - r + KL)]. One sweep down: a window of the KL+1 active
// rows (each KL+KU+1 wide after fill-in) is kept locally, every finished U row goes to scratch su(r, 0..KL+KU)
// (su(r,0) = 1/pivot) with its right-hand side sy(r, c). One sweep up.
template <int KL, int KU, class Rhs, class Out, class SU, class SY>
HD inline void band_solve_piv(int N, const real* G0, const real* G1, real k2, Rhs rhs, Out out, SU su, SY sy) {
  constexpr int W = KL + KU + 1;  // band width of a row
  constexpr int WU = KL + KU + 1;  // width of a U row after fill-in (columns j .. j+KL+KU)
  real Wn[KL + 1][WU];
  real R[KL + 1][NRHS];
  // load row r into window slot q, relative to the current column j (row r holds columns r-KL .. r+KU)
  auto load = [&](int q, int r, int j) {
    for (int c = 0; c < WU; ++c) Wn[q][c] = 0;
    for (int c = 0; c < NRHS; ++c) R[q][c] = 0;
    if (r >= N) return;
    const real* g0 = G0 + idx(r) * W;
    const real* g1 = G1 + idx(r) * W;
    for (int d = 0; d < W; ++d) {
      const int col = r - KL + d;
      const int off = col - j;
      if (off >= 0 && off < WU) Wn[q][off] = g0[d] + k2 * g1[d];
    }
    for (int c = 0; c < NRHS; ++c) R[q][c] = rhs(r, c);
  };
  for (int q = 0; q <= KL; ++q) load(q, q, 0);
  for (int j = 0; j < N; ++j) {
    // pivot search among the rows still present
    int p = 0;
    real best = Wn[0][0] < 0 ? -Wn[0][0] : Wn[0][0];
    for (int q = 1; q <= KL; ++q) {
      if (j + q >= N) break;
      const real a = Wn[q][0] < 0 ? -Wn[q][0] : Wn[q][0];
      if (a > best) { best = a; p = q; }
    }
    for (int q = 1; q <= KL; ++q)
      if (q == p) {
        for (int c = 0; c < WU; ++c) { const real t = Wn[0][c]; Wn[0][c] = Wn[q][c]; Wn[q][c] = t; }
        for (int c = 0; c < NRHS; ++c) { const real t = R[0][c]; R[0][c] = R[q][c]; R[q][c] = t; }
      }
    const real inv = real(1) / Wn[0][0];
    for (int q = 1; q <= KL; ++q) {
      const real l = Wn[q][0] * inv;
      for (int c = 1; c < WU; ++c) Wn[q][c] -= l * Wn[0][c];
      for (int c = 0; c < NRHS; ++c) R[q][c] -= l * R[0][c];
    }
    su(j, 0) = inv;
    for (int c = 1; c < WU; ++c) su(j, c) = Wn[0][c];
    for (int c = 0; c < NRHS; ++c) sy(j, c) = R[0][c];
    // shift the window by one row and one column, load row j+KL+1
    for (int q = 0; q < KL; ++q) {
      for (int c = 0; c < WU - 1; ++c) Wn[q][c] = Wn[q + 1][c + 1];
      Wn[q][WU - 1] = 0;
      for (int c = 0; c < NRHS; ++c) R[q][c] = R[q + 1][c];
    }
    load(KL, j + KL + 1, j + 1);
  }
  real X[WU][NRHS];  // X[m-1] = x_{j+m}
  for (int q = 0; q < WU; ++q)
    for (int c = 0; c < NRHS; ++c) X[q][c] = 0;
  for (int j = N - 1; j >= 0; --j) {
    real yv[NRHS];
    for (int c = 0; c < NRHS; ++c) yv[c] = sy(j, c);
    for (int m = 1; m < WU; ++m) {
      if (j + m >= N) break;
      const real u = su(j, m);
      for (int c = 0; c < NRHS; ++c) yv[c] -= u * X[m - 1][c];
    }
    const real inv = su(j, 0);
    for (int q = WU - 1; q > 0; --q)
      for (int c = 0; c < NRHS; ++c) X[q][c] = X[q - 1][c];
    for (int c = 0; c < NRHS; ++c) {
      X[0][c] = yv[c] * inv;
      out(j, c, X[0][c]);
    }
  }
}

}  // namespace fd
