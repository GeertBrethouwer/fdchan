// Communication layer for the z-slab decomposition (MPI; CUDA-aware MPI on GPUs: device pointers are
// passed directly). Without FD_MPI there is one rank and every call is a local copy.
//
// With FD_NCCL (make GPU=cuda MPI=1 NCCL=1; HIP: RCCL) the two field exchanges, the halo exchange (sendrecv) and
// the all-to-all transpose (alltoallv), go through NCCL instead of MPI when comm_backend = nccl (the default
// of such a build). NCCL uses NVLink between the GPUs of a node and is ordered on the solver stream, so no
// synchronisation is needed before the exchange. MPI still starts the ranks and does the reductions and the
// restart files.
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
#ifdef FD_NCCL
#if defined(FD_NCCL_MOCK)  // CPU test of the NCCL code path (tests/nccl_mock.hpp)
#include "../tests/nccl_mock.hpp"
#define FD_NCCL_STREAM nullptr
#elif !defined(FD_GPU) || !defined(FD_MPI)
#error "NCCL=1 needs GPU=cuda (or hip) and MPI=1"
#elif defined(FD_CUDA)
#include <nccl.h>
#define FD_NCCL_STREAM g_stream
#else
#include <rccl/rccl.h>
#define FD_NCCL_STREAM g_stream
#endif
#endif

namespace fd {

struct Comm {
  int rank = 0, size = 1;
  bool use_nccl = false;
#ifdef FD_MPI
  MPI_Comm comm = MPI_COMM_WORLD;
#endif
  // channel 0: the solver's main work; channel 1: the scalars when they run on their own stream (overlap_scalars).
  // With NCCL each channel has its own communicator: operations of one communicator must not run at the same time
  // from two streams. MPI calls are made by the host in program order on both channels.
  int channel = 0;
#ifdef FD_NCCL
  ncclComm_t nccl = nullptr, nccl2 = nullptr;
  ncclComm_t cur() const { return channel == 1 && nccl2 ? nccl2 : nccl; }
  static void nccl_check(ncclResult_t r, const char* where) {
    if (r != ncclSuccess) {
      std::fprintf(stderr, "NCCL error in %s: %s\n", where, ncclGetErrorString(r));
      stop(1);
    }
  }
  static ncclDataType_t nccl_real() { return sizeof(real) == 8 ? ncclDouble : ncclFloat; }
#endif
  static bool nccl_built() {
#ifdef FD_NCCL
    return true;
#else
    return false;
#endif
  }
  // after the GPU has been selected and the solver stream created (all ranks)
  void init_nccl(bool on) {
    use_nccl = on && size > 1;
#ifdef FD_NCCL
    if (!use_nccl) return;
    ncclUniqueId id;
    if (rank == 0) nccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");
    MPI_Bcast(&id, int(sizeof(id)), MPI_BYTE, 0, comm);
    nccl_check(ncclCommInitRank(&nccl, size, id, rank), "ncclCommInitRank");
#else
    if (use_nccl) fatal("comm_backend = nccl: build with NCCL=1");
#endif
  }
  // the communicator of channel 1 (overlap_scalars)
  void init_nccl2() {
#ifdef FD_NCCL
    if (!use_nccl || nccl2) return;
    ncclUniqueId id;
    if (rank == 0) nccl_check(ncclGetUniqueId(&id), "ncclGetUniqueId");
    MPI_Bcast(&id, int(sizeof(id)), MPI_BYTE, 0, comm);
    nccl_check(ncclCommInitRank(&nccl2, size, id, rank), "ncclCommInitRank (channel 1)");
#endif
  }
  void init(int* argc, char*** argv) {
#ifdef FD_MPI
    MPI_Init(argc, argv);
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    if (size > 1) g_abort = [](int code) { MPI_Abort(MPI_COMM_WORLD, code); };
#else
    (void)argc; (void)argv;
#endif
  }
  void finalize() {
#ifdef FD_NCCL
    if (nccl) { sync(); ncclCommDestroy(nccl); nccl = nullptr; }
    if (nccl2) { ncclCommDestroy(nccl2); nccl2 = nullptr; }
#endif
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
  void allmax(double* x, int n) const {
#ifdef FD_MPI
    if (size > 1) MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DOUBLE, MPI_MAX, comm);
#else
    (void)x; (void)n;
#endif
  }
  void allmin(double* x, int n) const {
#ifdef FD_MPI
    if (size > 1) MPI_Allreduce(MPI_IN_PLACE, x, n, MPI_DOUBLE, MPI_MIN, comm);
#else
    (void)x; (void)n;
#endif
  }
  double allmax(double x) const { allmax(&x, 1); return x; }
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
#ifdef FD_NCCL
    if (use_nccl) {
      (void)tag;  // NCCL matches the operations between two ranks in the order they are issued
      nccl_check(ncclGroupStart(), "ncclGroupStart");
      nccl_check(ncclSend(sb, size_t(n), nccl_real(), dest, cur(), FD_NCCL_STREAM), "ncclSend (halo)");
      nccl_check(ncclRecv(rb, size_t(n), nccl_real(), src, cur(), FD_NCCL_STREAM), "ncclRecv (halo)");
      nccl_check(ncclGroupEnd(), "ncclGroupEnd");
      return;
    }
#endif
#ifdef FD_MPI
    sync();
    const MPI_Datatype T = sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT;
    MPI_Sendrecv(sb, int(n), T, dest, tag, rb, int(n), T, src, tag, comm, MPI_STATUS_IGNORE);
#endif
  }
  // two exchanges at once (the two directions of a halo exchange): send s1 (n reals) to d1 and receive r1 from
  // o1, send s2 to d2 and receive r2 from o2; with NCCL in one group (one launch instead of two)
  void sendrecv2(const real* s1, real* r1, int d1, int o1, const real* s2, real* r2, int d2, int o2, idx n) const {
    if (d1 == rank && o1 == rank && d2 == rank && o2 == rank) {
      dcopy(r1, s1, sizeof(real) * size_t(n));
      dcopy(r2, s2, sizeof(real) * size_t(n));
      return;
    }
#ifdef FD_NCCL
    if (use_nccl) {
      nccl_check(ncclGroupStart(), "ncclGroupStart");
      nccl_check(ncclSend(s1, size_t(n), nccl_real(), d1, cur(), FD_NCCL_STREAM), "ncclSend (halo)");
      nccl_check(ncclRecv(r1, size_t(n), nccl_real(), o1, cur(), FD_NCCL_STREAM), "ncclRecv (halo)");
      nccl_check(ncclSend(s2, size_t(n), nccl_real(), d2, cur(), FD_NCCL_STREAM), "ncclSend (halo)");
      nccl_check(ncclRecv(r2, size_t(n), nccl_real(), o2, cur(), FD_NCCL_STREAM), "ncclRecv (halo)");
      nccl_check(ncclGroupEnd(), "ncclGroupEnd");
      return;
    }
#endif
    sendrecv(s1, n, d1, r1, o1, 11);
    sendrecv(s2, n, d2, r2, o2, 12);
  }
  // all-to-all with counts/displacements in reals
  void alltoallv(const real* sb, const std::vector<int>& sc, const std::vector<int>& sd, real* rb,
                 const std::vector<int>& rc, const std::vector<int>& rd) const {
    if (size == 1) { dcopy(rb + rd[0], sb + sd[0], sizeof(real) * size_t(sc[0])); return; }
#ifdef FD_NCCL
    if (use_nccl) {
      const size_t me = size_t(rank);
      dcopy(rb + rd[me], sb + sd[me], sizeof(real) * size_t(sc[me]));  // own block: a local copy
      nccl_check(ncclGroupStart(), "ncclGroupStart");
      for (int r = 0; r < size; ++r) {
        if (r == rank) continue;
        const size_t q = size_t(r);
        if (sc[q] > 0) nccl_check(ncclSend(sb + sd[q], size_t(sc[q]), nccl_real(), r, cur(), FD_NCCL_STREAM), "ncclSend (all-to-all)");
        if (rc[q] > 0) nccl_check(ncclRecv(rb + rd[q], size_t(rc[q]), nccl_real(), r, cur(), FD_NCCL_STREAM), "ncclRecv (all-to-all)");
      }
      nccl_check(ncclGroupEnd(), "ncclGroupEnd");
      return;
    }
#endif
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
