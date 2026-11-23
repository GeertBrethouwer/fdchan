// NCCL check across GPUs and nodes: does NCCL start, is the data right, and how fast is it?
//
//   make GPU=cuda ARCH=sm_90 MPI=1 NCCL=1 MPICXX=mpicxx build/nccl_check
//   srun --mpi=pmix build/nccl_check [MiB per pair, default 64]
//
// One rank per GPU. Three exchanges with ncclSend/ncclRecv (as FDCHAN's halo exchanges and all-to-alls):
//   1. all-to-all between all ranks
//   2. inter-node ring: every rank sends to the rank with the same node-local index on the next node
//      (only the network between nodes, e.g. Slingshot)
//   3. all-to-all inside each node (NVLink)
// Exchanges 1 and 2 check every received value. Rank 0 prints one line "RESULT ..." per exchange and a
// final "RESULT: OK" or "RESULT: FAIL ...". NCCL_DEBUG=INFO shows the network NCCL uses
// ("Using network ...", "NET/OFI Selected provider is ...").
#include <cuda_runtime.h>
#include <mpi.h>
#include <nccl.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int g_rank = 0;

static void cuda_ck(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "rank %d: CUDA error in %s: %s\n", g_rank, where, cudaGetErrorString(e));
    MPI_Abort(MPI_COMM_WORLD, 2);
  }
}
static bool nccl_ok(ncclResult_t r, ncclComm_t comm, const char* where) {
  if (r == ncclSuccess) return true;
  std::fprintf(stderr, "rank %d: NCCL error in %s: %s (%s)\n", g_rank, where, ncclGetErrorString(r),
               ncclGetLastError(comm));
  return false;
}

// value sent from rank `src` to rank `dst`, element i: exact in double
static double pattern(int src, int dst, size_t i) { return src * 1.0e6 + dst * 1.0e3 + double(i % 997); }

struct Exchange {
  std::string name;
  std::vector<int> to, from;  // peers this rank sends to / receives from (same length, same order on all ranks)
  bool check;
};

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int size;
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const int rank = g_rank;
  const double mib = argc > 1 ? std::atof(argv[1]) : 64.0;
  const size_t n = size_t(mib * 1048576.0 / sizeof(double));  // doubles per pair

  // node of every rank (leader = lowest world rank on the node) and node-local index
  MPI_Comm node;
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
  int local, nlocal, leader = rank;
  MPI_Comm_rank(node, &local);
  MPI_Comm_size(node, &nlocal);
  MPI_Allreduce(MPI_IN_PLACE, &leader, 1, MPI_INT, MPI_MIN, node);
  std::vector<int> leaders(static_cast<size_t>(size)), locals(static_cast<size_t>(size));
  MPI_Allgather(&leader, 1, MPI_INT, leaders.data(), 1, MPI_INT, MPI_COMM_WORLD);
  MPI_Allgather(&local, 1, MPI_INT, locals.data(), 1, MPI_INT, MPI_COMM_WORLD);
  std::vector<int> nodes = leaders;
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());
  const int nnodes = int(nodes.size());
  const int mynode = int(std::find(nodes.begin(), nodes.end(), leader) - nodes.begin());

  // GPU: node-local rank mod visible GPUs (as FDCHAN)
  int ndev = 0;
  cuda_ck(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
  const int dev = local % std::max(ndev, 1);
  cuda_ck(cudaSetDevice(dev), "cudaSetDevice");
  char bus[32] = "?", host[64] = "?";
  cudaDeviceGetPCIBusId(bus, sizeof(bus), dev);
  gethostname(host, sizeof(host));
  {
    char line[160];
    std::snprintf(line, sizeof(line), "rank %3d  node %d (%s)  local %d  GPU %d of %d  bus %s", rank, mynode, host, local,
                  dev, ndev, bus);
    std::vector<char> all(size_t(size) * sizeof(line));
    MPI_Gather(line, int(sizeof(line)), MPI_CHAR, all.data(), int(sizeof(line)), MPI_CHAR, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      int major = 0, minor = 0, patch = 0, v = 0;
      ncclGetVersion(&v);
      major = v / 10000; minor = (v % 10000) / 100; patch = v % 100;
      std::printf("nccl_check: %d ranks on %d node(s), %.1f MiB per pair, NCCL %d.%d.%d\n", size, nnodes, mib, major,
                  minor, patch);
      for (int r = 0; r < size; ++r) std::printf("  %s\n", &all[size_t(r) * sizeof(line)]);
      std::fflush(stdout);
    }
  }

  // NCCL communicator
  ncclUniqueId id;
  if (rank == 0 && !nccl_ok(ncclGetUniqueId(&id), nullptr, "ncclGetUniqueId")) MPI_Abort(MPI_COMM_WORLD, 3);
  MPI_Bcast(&id, int(sizeof(id)), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclComm_t comm = nullptr;
  const double t0 = MPI_Wtime();
  int ok = nccl_ok(ncclCommInitRank(&comm, size, id, rank), comm, "ncclCommInitRank") ? 1 : 0;
  const double tinit = MPI_Wtime() - t0;
  MPI_Allreduce(MPI_IN_PLACE, &ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
  if (!ok) {
    if (rank == 0) std::printf("RESULT: FAIL ncclCommInitRank (see the NCCL WARN lines)\n");
    MPI_Finalize();
    return 1;
  }
  if (rank == 0) std::printf("ncclCommInitRank: OK in %.2f s\n", tinit);

  // exchanges
  std::vector<Exchange> ex;
  {
    Exchange a{"all-to-all, all ranks", {}, {}, true};
    for (int s = 1; s < size; ++s) { a.to.push_back((rank + s) % size); a.from.push_back((rank - s + size) % size); }
    ex.push_back(a);
    if (nnodes > 1) {
      // same node-local index on the next / previous node (needs the same number of ranks on every node)
      auto find = [&](int nd) {
        for (int r = 0; r < size; ++r)
          if (leaders[size_t(r)] == nodes[size_t(nd)] && locals[size_t(r)] == local) return r;
        return -1;
      };
      Exchange b{"inter-node ring (network only)", {find((mynode + 1) % nnodes)}, {find((mynode - 1 + nnodes) % nnodes)}, true};
      int good = (b.to[0] >= 0 && b.from[0] >= 0) ? 1 : 0;
      MPI_Allreduce(MPI_IN_PLACE, &good, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
      if (good) ex.push_back(b);
      else if (rank == 0) std::printf("inter-node ring skipped: the nodes have different numbers of ranks\n");
    }
    Exchange c{"all-to-all inside each node (NVLink)", {}, {}, false};
    std::vector<int> mine;
    for (int r = 0; r < size; ++r)
      if (leaders[size_t(r)] == leader) mine.push_back(r);
    const int m = int(mine.size());
    for (int s = 1; s < m; ++s) { c.to.push_back(mine[size_t((local + s) % m)]); c.from.push_back(mine[size_t((local - s + m) % m)]); }
    if (m > 1) ex.push_back(c);
  }

  const size_t npeer = size_t(std::max(size - 1, 1));
  double *sbuf = nullptr, *rbuf = nullptr;
  cuda_ck(cudaMalloc(&sbuf, npeer * n * sizeof(double)), "cudaMalloc send");
  cuda_ck(cudaMalloc(&rbuf, npeer * n * sizeof(double)), "cudaMalloc recv");
  cudaStream_t st;
  cuda_ck(cudaStreamCreate(&st), "stream");
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::vector<double> h(n);
  bool all_ok = true;

  for (const Exchange& X : ex) {
    const int np = int(X.to.size());
    for (int p = 0; p < np; ++p) {  // send block p: data for X.to[p]
      for (size_t i = 0; i < n; ++i) h[i] = pattern(rank, X.to[size_t(p)], i);
      cuda_ck(cudaMemcpy(sbuf + size_t(p) * n, h.data(), n * sizeof(double), cudaMemcpyHostToDevice), "upload");
    }
    cuda_ck(cudaMemset(rbuf, 0, size_t(np) * n * sizeof(double)), "memset");
    auto run = [&]() -> bool {
      bool g = nccl_ok(ncclGroupStart(), comm, "ncclGroupStart");
      for (int p = 0; p < np && g; ++p) {
        g = nccl_ok(ncclSend(sbuf + size_t(p) * n, n, ncclDouble, X.to[size_t(p)], comm, st), comm, "ncclSend") &&
            nccl_ok(ncclRecv(rbuf + size_t(p) * n, n, ncclDouble, X.from[size_t(p)], comm, st), comm, "ncclRecv");
      }
      return nccl_ok(ncclGroupEnd(), comm, "ncclGroupEnd") && g;
    };
    // warm-up and check
    int good = 1;
    for (int it = 0; it < 2 && good; ++it) good = run() ? 1 : 0;
    if (good) cuda_ck(cudaStreamSynchronize(st), "sync");
    long bad = 0;
    if (good && X.check) {
      for (int p = 0; p < np; ++p) {
        cuda_ck(cudaMemcpy(h.data(), rbuf + size_t(p) * n, n * sizeof(double), cudaMemcpyDeviceToHost), "download");
        for (size_t i = 0; i < n; ++i) bad += h[i] != pattern(X.from[size_t(p)], rank, i);
      }
    }
    // timing
    const int iters = 10;
    float ms = 0;
    if (good) {
      MPI_Barrier(MPI_COMM_WORLD);
      cudaEventRecord(e0, st);
      for (int it = 0; it < iters && good; ++it) good = run() ? 1 : 0;
      cudaEventRecord(e1, st);
      cuda_ck(cudaEventSynchronize(e1), "event");
      cudaEventElapsedTime(&ms, e0, e1);
    }
    double tmax = ms / 1000.0 / iters;
    MPI_Allreduce(MPI_IN_PLACE, &good, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &bad, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(MPI_IN_PLACE, &tmax, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    if (rank == 0) {
      const double gb = double(np) * n * sizeof(double) / 1e9;  // sent by each rank
      if (!good)
        std::printf("RESULT %-38s FAIL (NCCL error)\n", X.name.c_str());
      else
        std::printf("RESULT %-38s %s  %8.3f ms per exchange, %7.2f GB sent per GPU, %8.1f GB/s per GPU%s\n",
                    X.name.c_str(), bad ? "WRONG DATA" : "ok", 1000 * tmax, gb, gb / tmax,
                    X.check ? "" : " (not checked)");
      std::fflush(stdout);
    }
    all_ok = all_ok && good && bad == 0;
    if (!good) break;
  }
  if (rank == 0) std::printf(all_ok ? "RESULT: OK\n" : "RESULT: FAIL\n");
  cudaFree(sbuf);
  cudaFree(rbuf);
  ncclCommDestroy(comm);
  MPI_Finalize();
  return all_ok ? 0 : 1;
}
