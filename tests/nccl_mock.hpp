// Test stand-in for NCCL on the CPU: the NCCL calls used by comm.hpp, done with non-blocking MPI.
// Checks the NCCL code path of comm.hpp (offsets, counts, peers, order, own block) without a GPU:
//
//   make MPI=1 MPICXX=mpicxx DEFS="-DFD_MPI -DFD_NCCL -DFD_NCCL_MOCK" EXE=build/fdchan_mpi_ncclmock
//   mpirun -n 4 build/fdchan_mpi_ncclmock input.txt        (comm_backend = nccl is the default)
//
// The results must agree bit for bit with build/fdchan_mpi (MPI path) on the same number of ranks.
// Like NCCL, operations between two ranks are matched in the order they are issued (MPI's non-overtaking
// rule with one tag). comm.hpp issues every send and receive inside ncclGroupStart/ncclGroupEnd.
#pragma once
#include <mpi.h>

#include <vector>

namespace fd {

typedef enum { ncclSuccess = 0, ncclSystemError = 2 } ncclResult_t;
typedef enum { ncclFloat = 7, ncclDouble = 8 } ncclDataType_t;
struct ncclUniqueId { char internal[128]; };
struct ncclMockComm { MPI_Comm comm; };
typedef ncclMockComm* ncclComm_t;

inline std::vector<MPI_Request>& nccl_mock_reqs() { static std::vector<MPI_Request> r; return r; }
inline int& nccl_mock_depth() { static int d = 0; return d; }
inline const char* ncclGetErrorString(ncclResult_t) { return "nccl mock error"; }
inline ncclResult_t ncclGetUniqueId(ncclUniqueId* id) { for (char& c : id->internal) c = 0; return ncclSuccess; }
inline ncclResult_t ncclCommInitRank(ncclComm_t* c, int, ncclUniqueId, int) {
  *c = new ncclMockComm{MPI_COMM_WORLD};
  return ncclSuccess;
}
inline ncclResult_t ncclCommDestroy(ncclComm_t c) { delete c; return ncclSuccess; }
inline MPI_Datatype nccl_mock_type(ncclDataType_t t) { return t == ncclDouble ? MPI_DOUBLE : MPI_FLOAT; }
inline ncclResult_t ncclGroupStart() { ++nccl_mock_depth(); return ncclSuccess; }
inline ncclResult_t ncclGroupEnd() {
  if (--nccl_mock_depth() == 0 && !nccl_mock_reqs().empty()) {
    MPI_Waitall(int(nccl_mock_reqs().size()), nccl_mock_reqs().data(), MPI_STATUSES_IGNORE);
    nccl_mock_reqs().clear();
  }
  return ncclSuccess;
}
inline ncclResult_t ncclSend(const void* b, size_t n, ncclDataType_t t, int peer, ncclComm_t c, void*) {
  MPI_Request r;
  MPI_Isend(b, int(n), nccl_mock_type(t), peer, 77, c->comm, &r);
  nccl_mock_reqs().push_back(r);
  return ncclSuccess;
}
inline ncclResult_t ncclRecv(void* b, size_t n, ncclDataType_t t, int peer, ncclComm_t c, void*) {
  MPI_Request r;
  MPI_Irecv(b, int(n), nccl_mock_type(t), peer, 77, c->comm, &r);
  nccl_mock_reqs().push_back(r);
  return ncclSuccess;
}

}  // namespace fd
