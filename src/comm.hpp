// Communication layer for the z-slab decomposition (MPI; CUDA-aware MPI on GPUs: device pointers are
// passed directly). Without FD_MPI there is one rank and every call is a local copy.
//
// Decomposition: P ranks along z. Rank r holds z = z0 .. z0+nzl-1 (block distribution, uneven allowed) for all
// x-modes and all y; for the pressure solve it holds kx = kx0 .. kx0+nkl-1 for all z and y. py = 1 for now;
// the structure is meant to take a second split in y later.
#pragma once
#include <vector>

#include "backend.hpp"
#ifdef FD_MPI
#include <mpi.h>
#endif

namespace fd {

struct Comm {
  int rank = 0, size = 1;
#ifdef FD_MPI
  MPI_Comm comm = MPI_COMM_WORLD;
#endif
  void init(int* argc, char*** argv) {
#ifdef FD_MPI
    MPI_Init(argc, argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
#else
    (void)argc; (void)argv;
#endif
  }
  void finalize() {
#ifdef FD_MPI
    MPI_Finalize();
#endif
  }
  bool root() const { return rank == 0; }
  void allsum(double* x, int n) const {
#ifdef FD_MPI
    if (size > 1) MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DOUBLE, MPI_SUM, comm);
#else
    (void)x; (void)n;
#endif
  }
  void allsum(std::vector<double>& x) const { allsum(x.data(), int(x.size())); }
  double allsum(double x) const { allsum(&x, 1); return x; }
  double allmax(double x) const {
#ifdef FD_MPI
    if (size > 1) MPI_Allreduce(MPI_IN_PLACE, &x, 1, MPI_DOUBLE, MPI_MAX, comm);
#endif
    return x;
  }
  void barrier() const {
#ifdef FD_MPI
    MPI_Barrier(comm);
#endif
  }
  // device-to-device copy (used when source and destination are on the same rank)
  static void dcopy(void* dst, const void* src, size_t bytes) {
#ifdef FD_GPU
    gpu_check(gpuMemcpyAsync(dst, src, bytes, gpuMemcpyDeviceToDevice, g_stream), "dcopy");
#else
    std::memcpy(dst, src, bytes);
#endif
  }
  // send n reals to dest, receive n reals from src (real = double or float)
  void sendrecv(const real* sb, idx n, int dest, real* rb, int src, int tag) const {
    if (dest == rank && src == rank) { dcopy(rb, sb, sizeof(real) * size_t(n)); return; }
#ifdef FD_MPI
    sync();
    const MPI_Datatype T = sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT;
    MPI_Sendrecv(sb, int(n), T, dest, tag, rb, int(n), T, src, tag, comm, MPI_STATUS_IGNORE);
#endif
  }
  // all-to-all with counts/displacements in reals
  void alltoallv(const real* sb, const std::vector<int>& sc, const std::vector<int>& sd, real* rb,
                 const std::vector<int>& rc, const std::vector<int>& rd) const {
    if (size == 1) { dcopy(rb, sb, sizeof(real) * size_t(sc[0])); return; }
#ifdef FD_MPI
    sync();
    const MPI_Datatype T = sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT;
    MPI_Alltoallv(sb, sc.data(), sd.data(), T, rb, rc.data(), rd.data(), T, comm);
#endif
  }
};

extern Comm g_comm;

// block distribution of n items over p parts: start of part r
inline int block_start(int n, int p, int r) { return int((long long)n * r / p); }

}  // namespace fd
