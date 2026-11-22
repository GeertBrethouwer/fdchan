// Batched FFTs.
//
//   XFFT: real <-> complex along x, in place. Each line holds ld = 2*(mx/2+1) reals, i.e. mx/2+1 complex
//         numbers in spectral space. Unnormalised in both directions.
//   ZFFT: complex <-> complex along z, in place, contiguous lines of length nz. Unnormalised.
//
// GPU: cuFFT / hipFFT plans on the solver stream. CPU: a small mixed-radix FFT (any length; factors
// 2, 3, 4, 5 are fast, other primes use a direct DFT of that factor).
#pragma once
#include <complex>
#include <vector>

#include "backend.hpp"
#include "timer.hpp"

#if defined(FD_CUDA)
#include <cufft.h>
#define fftHandle cufftHandle
#define fftResult cufftResult
#define FFT_SUCCESS CUFFT_SUCCESS
#define fftPlanMany cufftPlanMany
#define fftSetStream cufftSetStream
#define fftDestroy cufftDestroy
#define FFT_FORWARD CUFFT_FORWARD
#define FFT_INVERSE CUFFT_INVERSE
#ifdef FD_SINGLE
#define FFT_R2C CUFFT_R2C
#define FFT_C2R CUFFT_C2R
#define FFT_C2C CUFFT_C2C
#define fftExecR2C cufftExecR2C
#define fftExecC2R cufftExecC2R
#define fftExecC2C cufftExecC2C
using fftReal = cufftReal;
using fftComplex = cufftComplex;
#else
#define FFT_R2C CUFFT_D2Z
#define FFT_C2R CUFFT_Z2D
#define FFT_C2C CUFFT_Z2Z
#define fftExecR2C cufftExecD2Z
#define fftExecC2R cufftExecZ2D
#define fftExecC2C cufftExecZ2Z
using fftReal = cufftDoubleReal;
using fftComplex = cufftDoubleComplex;
#endif
#elif defined(FD_HIP)
#include <hipfft/hipfft.h>
#define fftHandle hipfftHandle
#define fftResult hipfftResult
#define FFT_SUCCESS HIPFFT_SUCCESS
#define fftPlanMany hipfftPlanMany
#define fftSetStream hipfftSetStream
#define fftDestroy hipfftDestroy
#define FFT_FORWARD HIPFFT_FORWARD
#define FFT_INVERSE HIPFFT_BACKWARD
#ifdef FD_SINGLE
#define FFT_R2C HIPFFT_R2C
#define FFT_C2R HIPFFT_C2R
#define FFT_C2C HIPFFT_C2C
#define fftExecR2C hipfftExecR2C
#define fftExecC2R hipfftExecC2R
#define fftExecC2C hipfftExecC2C
using fftReal = hipfftReal;
using fftComplex = hipfftComplex;
#else
#define FFT_R2C HIPFFT_D2Z
#define FFT_C2R HIPFFT_Z2D
#define FFT_C2C HIPFFT_Z2Z
#define fftExecR2C hipfftExecD2Z
#define fftExecC2R hipfftExecZ2D
#define fftExecC2C hipfftExecZ2Z
using fftReal = hipfftDoubleReal;
using fftComplex = hipfftDoubleComplex;
#endif
#endif

namespace fd {

#ifndef FD_GPU
// ------------------------------------------------------------------ CPU FFT (written for this code)
// Recursive decimation in time over the factors of n. One plan per length; execute is thread safe.
class CpuFFT {
 public:
  using C = std::complex<double>;
  explicit CpuFFT(int n = 1) : n_(n) {
    int r = n;
    for (int p : {4, 2, 3, 5}) while (r % p == 0) { fac_.push_back(p); r /= p; }
    for (int p = 7; r > 1; p += 2) while (r % p == 0) { fac_.push_back(p); r /= p; }
    tw_.resize(size_t(n));
    for (int j = 0; j < n; ++j) tw_[size_t(j)] = std::polar(1.0, -2.0 * M_PI * j / n);
  }
  int size() const { return n_; }
  // in-place transform of x (length n), sign = -1 forward, +1 inverse; work has length n
  void exec(C* x, C* work, int sign) const {
    rec(x, 1, work, n_, 0, sign);
    for (int j = 0; j < n_; ++j) x[j] = work[j];
  }

 private:
  // out[0..n) = DFT of in[0], in[s], ..., in[(n-1)s]
  void rec(const C* in, int s, C* out, int n, int f, int sign) const {
    if (n == 1) { out[0] = in[0]; return; }
    const int p = fac_[size_t(f)], m = n / p;
    for (int r = 0; r < p; ++r) rec(in + r * s, s * p, out + r * m, m, f + 1, sign);
    const int step = n_ / n;  // W_n^j = tw[j*step]
    C a[64];
    for (int k = 0; k < m; ++k) {
      for (int r = 0; r < p; ++r) {
        C w = tw_[size_t((r * k * step) % n_)];
        if (sign > 0) w = std::conj(w);
        a[r] = out[r * m + k] * w;
      }
      if (p == 2) {
        out[k] = a[0] + a[1];
        out[k + m] = a[0] - a[1];
      } else if (p == 4) {
        const C t0 = a[0] + a[2], t1 = a[0] - a[2], t2 = a[1] + a[3];
        C t3 = a[1] - a[3];
        t3 = (sign < 0) ? C(t3.imag(), -t3.real()) : C(-t3.imag(), t3.real());  // -+ i t3
        out[k] = t0 + t2;
        out[k + m] = t1 + t3;
        out[k + 2 * m] = t0 - t2;
        out[k + 3 * m] = t1 - t3;
      } else {
        C b[64];
        for (int t = 0; t < p; ++t) {
          C acc = 0;
          for (int r = 0; r < p; ++r) {
            C w = tw_[size_t(((r * t) % p) * (n_ / p))];
            if (sign > 0) w = std::conj(w);
            acc += a[r] * w;
          }
          b[t] = acc;
        }
        for (int t = 0; t < p; ++t) out[k + t * m] = b[t];
      }
    }
  }
  int n_;
  std::vector<int> fac_;
  std::vector<C> tw_;
};
#endif

// ------------------------------------------------------------------ x transforms
class XFFT {
 public:
  int mx = 0, lp = 0, ld = 0;
  idx nlines = 0;
  void init(int mx_, idx nlines_) {
    mx = mx_;
    lp = mx / 2 + 1;
    ld = 2 * lp;
    nlines = nlines_;
#ifdef FD_GPU
    int n[1] = {mx}, re[1] = {ld}, ce[1] = {lp};
    if (fftPlanMany(&pf_, 1, n, re, 1, ld, ce, 1, lp, FFT_R2C, int(nlines)) != FFT_SUCCESS)
      fatal("FFT plan (x, forward)");
    if (fftPlanMany(&pb_, 1, n, ce, 1, lp, re, 1, ld, FFT_C2R, int(nlines)) != FFT_SUCCESS)
      fatal("FFT plan (x, backward)");
    fftSetStream(pf_, g_stream);
    fftSetStream(pb_, g_stream);
#else
    plan_ = CpuFFT(mx);
#endif
  }
  // real (line stride ld) -> complex (line stride lp), in place
  void forward(real* buf) {
    g_timer.tic(tm_xfft);
    forward_(buf);
    g_timer.toc(tm_xfft);
  }
  void backward(real* buf) {
    g_timer.tic(tm_xfft);
    backward_(buf);
    g_timer.toc(tm_xfft);
  }
  ~XFFT() {
#ifdef FD_GPU
    if (mx) { fftDestroy(pf_); fftDestroy(pb_); }
#endif
  }

 private:
  void forward_(real* buf) {
#ifdef FD_GPU
    if (fftExecR2C(pf_, (fftReal*)buf, (fftComplex*)buf) != FFT_SUCCESS) fatal("FFT exec (x, forward)");
#else
    const int m = mx, l = lp, d = ld;
    const CpuFFT& P = plan_;
    pfor_lines([&](idx line, std::vector<CpuFFT::C>& a, std::vector<CpuFFT::C>& w) {
      real* r = buf + line * d;
      for (int i = 0; i < m; ++i) a[size_t(i)] = CpuFFT::C(r[i], 0.0);
      P.exec(a.data(), w.data(), -1);
      for (int i = 0; i < l; ++i) { r[2 * i] = real(a[size_t(i)].real()); r[2 * i + 1] = real(a[size_t(i)].imag()); }
    });
#endif
  }
  // complex (line stride lp) -> real (line stride ld), in place; Im of bin 0 (and bin mx/2) ignored
  void backward_(real* buf) {
#ifdef FD_GPU
    if (fftExecC2R(pb_, (fftComplex*)buf, (fftReal*)buf) != FFT_SUCCESS) fatal("FFT exec (x, backward)");
#else
    const int m = mx, l = lp, d = ld;
    const CpuFFT& P = plan_;
    pfor_lines([&](idx line, std::vector<CpuFFT::C>& a, std::vector<CpuFFT::C>& w) {
      real* r = buf + line * d;
      a[0] = CpuFFT::C(r[0], 0.0);
      for (int i = 1; i < l; ++i) a[size_t(i)] = CpuFFT::C(r[2 * i], r[2 * i + 1]);
      if (m % 2 == 0) a[size_t(m / 2)] = CpuFFT::C(r[m], 0.0);
      for (int i = 1; i < (m + 1) / 2; ++i) a[size_t(m - i)] = std::conj(a[size_t(i)]);
      P.exec(a.data(), w.data(), +1);
      for (int i = 0; i < m; ++i) r[i] = real(a[size_t(i)].real());
    });
#endif
  }
#ifdef FD_GPU
  fftHandle pf_, pb_;
#else
  CpuFFT plan_;
  template <class F>
  void pfor_lines(F f) {
    const int m = mx;
    Pool& P = Pool::get();
    const int nt = P.threads();
    const idx nl = nlines;
    P.run([&](int t) {
      std::vector<CpuFFT::C> a(static_cast<size_t>(m)), w(static_cast<size_t>(m));
      for (idx line = nl * t / nt; line < nl * (t + 1) / nt; ++line) f(line, a, w);
    });
  }
#endif
};

// ------------------------------------------------------------------ z transforms
class ZFFT {
 public:
  int nz = 0;
  idx nlines = 0;
  void init(int nz_, idx nlines_) {
    nz = nz_;
    nlines = nlines_;
#ifdef FD_GPU
    int n[1] = {nz};
    if (fftPlanMany(&p_, 1, n, n, 1, nz, n, 1, nz, FFT_C2C, int(nlines)) != FFT_SUCCESS) fatal("FFT plan (z)");
    fftSetStream(p_, g_stream);
#else
    plan_ = CpuFFT(nz);
#endif
  }
  void exec(cplx* buf, int sign) {
    g_timer.tic(tm_zfft);
    exec_(buf, sign);
    g_timer.toc(tm_zfft);
  }
  ~ZFFT() {
#ifdef FD_GPU
    if (nz) fftDestroy(p_);
#endif
  }

 private:
  void exec_(cplx* buf, int sign) {
#ifdef FD_GPU
    if (fftExecC2C(p_, (fftComplex*)buf, (fftComplex*)buf, sign < 0 ? FFT_FORWARD : FFT_INVERSE) != FFT_SUCCESS)
      fatal("FFT exec (z)");
#else
    const int n = nz;
    const CpuFFT& P = plan_;
    Pool& pool = Pool::get();
    const int nt = pool.threads();
    const idx nl = nlines;
    pool.run([&](int t) {
      std::vector<CpuFFT::C> a(static_cast<size_t>(n)), w(static_cast<size_t>(n));
      for (idx line = nl * t / nt; line < nl * (t + 1) / nt; ++line) {
        cplx* c = buf + line * n;
        for (int i = 0; i < n; ++i) a[size_t(i)] = CpuFFT::C(c[i].re, c[i].im);
        P.exec(a.data(), w.data(), sign);
        for (int i = 0; i < n; ++i) c[i] = cplx(real(a[size_t(i)].real()), real(a[size_t(i)].imag()));
      }
    });
#endif
  }
#ifdef FD_GPU
  fftHandle p_;
#else
  CpuFFT plan_;
#endif
};

}  // namespace fd
