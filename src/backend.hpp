// Portability layer: the same kernels run on NVIDIA GPUs (CUDA), AMD GPUs (HIP) or the CPU.
//
//   make            CPU build (std::thread pool), used for testing
//   make GPU=cuda   nvcc, cuFFT
//   make GPU=hip    hipcc, hipFFT
//
// Kernels are written as lambdas `[=] HD (idx i) {...}` and launched with pfor(n, f).
#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(FD_CUDA)
#include <cuda_runtime.h>
#define FD_GPU 1
#define HD __host__ __device__
#define gpuStream_t cudaStream_t
#define gpuMalloc cudaMalloc
#define gpuFree cudaFree
#define gpuMemcpyAsync cudaMemcpyAsync
#define gpuMemsetAsync cudaMemsetAsync
#define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
#define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
#define gpuStreamSynchronize cudaStreamSynchronize
#define gpuStreamCreate cudaStreamCreate
#define gpuGetLastError cudaGetLastError
#define gpuGetErrorString cudaGetErrorString
#define gpuSuccess cudaSuccess
#define gpuError_t cudaError_t
#elif defined(FD_HIP)
#include <hip/hip_runtime.h>
#define FD_GPU 1
#define HD __host__ __device__
#define gpuStream_t hipStream_t
#define gpuMalloc hipMalloc
#define gpuFree hipFree
#define gpuMemcpyAsync hipMemcpyAsync
#define gpuMemsetAsync hipMemsetAsync
#define gpuMemcpyHostToDevice hipMemcpyHostToDevice
#define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
#define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define gpuStreamSynchronize hipStreamSynchronize
#define gpuStreamCreate hipStreamCreate
#define gpuGetLastError hipGetLastError
#define gpuGetErrorString hipGetErrorString
#define gpuSuccess hipSuccess
#define gpuError_t hipError_t
#else
#define HD
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#endif

namespace fd {

#ifdef FD_SINGLE
using real = float;
#else
using real = double;
#endif
using idx = long long;

[[noreturn]] inline void fatal(const char* msg) {
  std::fprintf(stderr, "fdchan error: %s\n", msg);
  std::exit(1);
}

// ------------------------------------------------------------------ complex numbers
struct cplx {
  real re, im;
  HD cplx() {}
  HD constexpr cplx(real r, real i = 0) : re(r), im(i) {}
};
HD inline cplx operator+(cplx a, cplx b) { return cplx(a.re + b.re, a.im + b.im); }
HD inline cplx operator-(cplx a, cplx b) { return cplx(a.re - b.re, a.im - b.im); }
HD inline cplx operator-(cplx a) { return cplx(-a.re, -a.im); }
HD inline cplx operator*(real s, cplx a) { return cplx(s * a.re, s * a.im); }
HD inline cplx operator*(cplx a, real s) { return cplx(s * a.re, s * a.im); }
HD inline cplx operator*(cplx a, cplx b) {
  return cplx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re);
}
HD inline cplx& operator+=(cplx& a, cplx b) { a.re += b.re; a.im += b.im; return a; }
HD inline cplx& operator-=(cplx& a, cplx b) { a.re -= b.re; a.im -= b.im; return a; }
HD inline cplx conj(cplx a) { return cplx(a.re, -a.im); }
HD inline cplx itimes(real k, cplx a) { return cplx(-k * a.im, k * a.re); }  // i k a
HD inline real abs2(cplx a) { return a.re * a.re + a.im * a.im; }

// ------------------------------------------------------------------ device memory
#ifdef FD_GPU
extern gpuStream_t g_stream;
inline void gpu_check(gpuError_t e, const char* where) {
  if (e != gpuSuccess) {
    std::fprintf(stderr, "GPU error in %s: %s\n", where, gpuGetErrorString(e));
    std::exit(1);
  }
}
inline void sync() { gpu_check(gpuStreamSynchronize(g_stream), "sync"); }
#else
inline void sync() {}
#endif

template <class T>
struct DArray {
  T* p = nullptr;
  idx n = 0;
  DArray() = default;
  explicit DArray(idx n_) { alloc(n_); }
  DArray(const DArray&) = delete;
  DArray& operator=(const DArray&) = delete;
  ~DArray() { release(); }
  void alloc(idx n_) {
    release();
    n = n_;
    if (n == 0) return;
#ifdef FD_GPU
    gpu_check(gpuMalloc((void**)&p, sizeof(T) * size_t(n)), "malloc");
#else
    p = static_cast<T*>(std::aligned_alloc(64, ((sizeof(T) * size_t(n) + 63) / 64) * 64));
    if (!p) fatal("out of memory");
#endif
    zero();
  }
  void release() {
    if (!p) return;
#ifdef FD_GPU
    gpuFree(p);
#else
    std::free(p);
#endif
    p = nullptr;
    n = 0;
  }
  void zero() {
#ifdef FD_GPU
    gpu_check(gpuMemsetAsync(p, 0, sizeof(T) * size_t(n), g_stream), "memset");
#else
    std::memset((void*)p, 0, sizeof(T) * size_t(n));
#endif
  }
  void upload(const T* h, idx cnt = -1) {
    if (cnt < 0) cnt = n;
#ifdef FD_GPU
    gpu_check(gpuMemcpyAsync(p, h, sizeof(T) * size_t(cnt), gpuMemcpyHostToDevice, g_stream), "upload");
    sync();
#else
    std::memcpy((void*)p, h, sizeof(T) * size_t(cnt));
#endif
  }
  void upload(const std::vector<T>& h) { upload(h.data(), idx(h.size())); }
  void download(T* h, idx cnt = -1) const {
    if (cnt < 0) cnt = n;
#ifdef FD_GPU
    gpu_check(gpuMemcpyAsync(h, p, sizeof(T) * size_t(cnt), gpuMemcpyDeviceToHost, g_stream), "download");
    sync();
#else
    std::memcpy((void*)h, p, sizeof(T) * size_t(cnt));
#endif
  }
  std::vector<T> host() const {
    std::vector<T> h(static_cast<size_t>(n));
    download(h.data());
    return h;
  }
};

// ------------------------------------------------------------------ parallel loops
#ifdef FD_GPU
template <class F>
__global__ void k_pfor(idx n, F f) {
  idx i = idx(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) f(i);
}
template <class F>
inline void pfor(idx n, F f) {
  if (n <= 0) return;
  const int bs = 256;
  k_pfor<<<unsigned((n + bs - 1) / bs), bs, 0, g_stream>>>(n, f);
#ifdef FD_DEBUG_SYNC
  gpu_check(gpuGetLastError(), "kernel launch");
  sync();
#endif
}

// out[r] = sum_c f(r, c), one block per row, result left on the device
template <class F>
__global__ void k_rowsum(int ncols, F f, double* out) {
  __shared__ double s[256];
  const int r = blockIdx.x;
  double acc = 0;
  for (int c = threadIdx.x; c < ncols; c += blockDim.x) acc += f(r, c);
  s[threadIdx.x] = acc;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) s[threadIdx.x] += s[threadIdx.x + w];
    __syncthreads();
  }
  if (threadIdx.x == 0) out[r] = s[0];
}
template <class F>
inline void rowsum(int nrows, int ncols, F f, double* d_out) {
  if (nrows <= 0) return;
  k_rowsum<<<nrows, 256, 0, g_stream>>>(ncols, f, d_out);
}

template <class F>
__global__ void k_maxred(idx n, F f, double* out) {
  __shared__ double s[256];
  double m = 0;
  for (idx i = idx(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += idx(gridDim.x) * blockDim.x)
    m = fmax(m, double(f(i)));
  s[threadIdx.x] = m;
  __syncthreads();
  for (int w = blockDim.x / 2; w > 0; w >>= 1) {
    if (threadIdx.x < w) s[threadIdx.x] = fmax(s[threadIdx.x], s[threadIdx.x + w]);
    __syncthreads();
  }
  if (threadIdx.x == 0) out[blockIdx.x] = s[0];
}
// max_i f(i) (f >= 0), returned to the host
template <class F>
inline double reduce_max(idx n, F f) {
  static DArray<double> part(1024);
  static std::vector<double> h(1024);
  int nb = int(std::min<idx>(1024, (n + 255) / 256));
  if (nb < 1) return 0;
  k_maxred<<<nb, 256, 0, g_stream>>>(n, f, part.p);
  part.download(h.data(), nb);
  double m = 0;
  for (int b = 0; b < nb; ++b) m = std::max(m, h[b]);
  return m;
}
#else
// ---- CPU: persistent thread pool
class Pool {
 public:
  static Pool& get() {
    static Pool p;
    return p;
  }
  void set_threads(int n) {
    stop();
    nthreads_ = std::max(1, n);
    for (int t = 1; t < nthreads_; ++t) workers_.emplace_back([this, t] { loop(t); });
  }
  int threads() const { return nthreads_; }
  // run job(t) for t = 0..nthreads-1, thread 0 is the caller
  void run(const std::function<void(int)>& job) {
    if (nthreads_ == 1) { job(0); return; }
    {
      std::unique_lock<std::mutex> lk(m_);
      job_ = &job;
      pending_ = nthreads_ - 1;
      ++gen_;
    }
    cv_.notify_all();
    job(0);
    std::unique_lock<std::mutex> lk(m_);
    done_.wait(lk, [this] { return pending_ == 0; });
    job_ = nullptr;
  }
  ~Pool() { stop(); }

 private:
  void stop() {
    {
      std::unique_lock<std::mutex> lk(m_);
      quit_ = true;
      ++gen_;
    }
    cv_.notify_all();
    for (auto& w : workers_) w.join();
    workers_.clear();
    quit_ = false;
  }
  void loop(int t) {
    uint64_t seen = gen_;
    for (;;) {
      const std::function<void(int)>* job;
      {
        std::unique_lock<std::mutex> lk(m_);
        cv_.wait(lk, [&] { return gen_ != seen; });
        seen = gen_;
        if (quit_) return;
        job = job_;
      }
      (*job)(t);
      {
        std::unique_lock<std::mutex> lk(m_);
        if (--pending_ == 0) done_.notify_one();
      }
    }
  }
  int nthreads_ = 1;
  std::vector<std::thread> workers_;
  std::mutex m_;
  std::condition_variable cv_, done_;
  const std::function<void(int)>* job_ = nullptr;
  int pending_ = 0;
  uint64_t gen_ = 0;
  bool quit_ = false;
};

template <class F>
inline void pfor(idx n, F f) {
  Pool& P = Pool::get();
  const int nt = P.threads();
  if (nt == 1 || n < 2048) {
    for (idx i = 0; i < n; ++i) f(i);
    return;
  }
  P.run([&](int t) {
    const idx a = n * t / nt, b = n * (t + 1) / nt;
    for (idx i = a; i < b; ++i) f(i);
  });
}
template <class F>
inline void rowsum(int nrows, int ncols, F f, double* out) {
  pfor(nrows, [&](idx r) {
    double acc = 0;
    for (int c = 0; c < ncols; ++c) acc += f(int(r), c);
    out[r] = acc;
  });
}
template <class F>
inline double reduce_max(idx n, F f) {
  Pool& P = Pool::get();
  const int nt = P.threads();
  std::vector<double> part(size_t(nt), 0.0);
  P.run([&](int t) {
    const idx a = n * t / nt, b = n * (t + 1) / nt;
    double m = 0;
    for (idx i = a; i < b; ++i) m = std::max(m, double(f(i)));
    part[size_t(t)] = m;
  });
  double m = 0;
  for (double v : part) m = std::max(m, v);
  return m;
}
#endif

// copy a few doubles from the device (diagnostics only)
inline std::vector<double> fetch(const double* d, int n) {
  std::vector<double> h(static_cast<size_t>(n));
#ifdef FD_GPU
  gpu_check(gpuMemcpyAsync(h.data(), d, sizeof(double) * size_t(n), gpuMemcpyDeviceToHost, g_stream), "fetch");
  sync();
#else
  std::memcpy(h.data(), d, sizeof(double) * size_t(n));
#endif
  return h;
}

}  // namespace fd
