// NCCL check across GPUs and nodes: does NCCL start, is the data right, and how fast is it?
// Independent of MPI: the ranks come from Slurm (SLURM_PROCID, SLURM_NTASKS, SLURM_LOCALID, SLURM_NODEID)
// and rank 0 passes NCCL's unique id to the others through a file in the current directory.
//
//   make GPU=cuda ARCH=sm_90 NCCL=1 build/nccl_check
//   srun --mpi=none -l build/nccl_check [MiB per pair, default 64]
//
// One rank per GPU. Three exchanges with ncclSend/ncclRecv (as FDCHAN's halo exchanges and all-to-alls):
//   1. all-to-all between all ranks
//   2. inter-node ring: every rank sends to the rank with the same node-local index on the next node
//      (only the network between nodes, e.g. Slingshot)
//   3. all-to-all inside each node (NVLink)
// Exchanges 1 and 2 check every received value. Every rank prints how far it got ("stage ..."); rank 0 prints
// one line "RESULT ..." per exchange and a final "RESULT: OK" or "RESULT: FAIL ...". NCCL_DEBUG=INFO shows the
// network NCCL uses ("Using network ...", "NET/OFI Selected provider is ...").
// Ranks must be numbered node by node (Slurm's default block distribution) with the same number on every node.
#include <cuda_runtime.h>
#include <nccl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

static int g_rank = 0;

static void stage(const char* what) {
  std::fprintf(stderr, "stage rank %d: %s\n", g_rank, what);
  std::fflush(stderr);
}
static void cuda_ck(cudaError_t e, const char* where) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "rank %d: CUDA error in %s: %s\n", g_rank, where, cudaGetErrorString(e));
    std::exit(2);
  }
}
static bool nccl_ok(ncclResult_t r, ncclComm_t comm, const char* where) {
  if (r == ncclSuccess) return true;
  std::fprintf(stderr, "rank %d: NCCL error in %s: %s (%s)\n", g_rank, where, ncclGetErrorString(r),
               ncclGetLastError(comm));
  return false;
}
static int env_int(const char* name, int def) {
  const char* s = std::getenv(name);
  return s ? std::atoi(s) : def;
}

// value sent from rank `src` to rank `dst`, element i: exact in double
static double pattern(int src, int dst, size_t i) { return src * 1.0e6 + dst * 1.0e3 + double(i % 997); }

struct Exchange {
  std::string name;
  std::vector<int> to, from;  // peers this rank sends to / receives from (same order on all ranks)
  bool check;
};

int main(int argc, char** argv) {
  const int rank = env_int("SLURM_PROCID", 0), size = env_int("SLURM_NTASKS", 1);
  const int local = env_int("SLURM_LOCALID", 0), mynode = env_int("SLURM_NODEID", 0);
  const int nnodes = env_int("SLURM_JOB_NUM_NODES", env_int("SLURM_NNODES", 1));
  g_rank = rank;
  const double mib = argc > 1 ? std::atof(argv[1]) : 64.0;
  const size_t n = size_t(mib * 1048576.0 / sizeof(double));  // doubles per pair
  if (size % nnodes) {
    if (rank == 0) std::printf("RESULT: FAIL the nodes must have the same number of ranks\n");
    return 1;
  }
  const int nlocal = size / nnodes;  // ranks per node, numbered node by node
  stage("started");

  int ndev = 0;
  cuda_ck(cudaGetDeviceCount(&ndev), "cudaGetDeviceCount");
  const int dev = local % std::max(ndev, 1);
  cuda_ck(cudaSetDevice(dev), "cudaSetDevice");
  char bus[32] = "?", host[64] = "?";
  cudaDeviceGetPCIBusId(bus, sizeof(bus), dev);
  gethostname(host, sizeof(host));
  std::fprintf(stderr, "rank %3d  node %d (%s)  local %d  GPU %d of %d  bus %s\n", rank, mynode, host, local, dev, ndev, bus);
  int v = 0;
  ncclGetVersion(&v);
  if (rank == 0)
    std::printf("nccl_check: %d ranks on %d node(s), %.1f MiB per pair, NCCL %d.%d.%d\n", size, nnodes, mib, v / 10000,
                (v % 10000) / 100, v % 100);
  std::fflush(stdout);

  // NCCL's unique id from rank 0 through a file (one per job step)
  const std::string idfile = "nccl_check_id." + std::string(std::getenv("SLURM_JOB_ID") ? std::getenv("SLURM_JOB_ID") : "0") +
                             "." + std::string(std::getenv("SLURM_STEP_ID") ? std::getenv("SLURM_STEP_ID") : "0");
  ncclUniqueId id;
  if (rank == 0) {
    if (!nccl_ok(ncclGetUniqueId(&id), nullptr, "ncclGetUniqueId")) return 3;
    const std::string tmp = idfile + ".tmp";
    std::FILE* f = std::fopen(tmp.c_str(), "wb");
    if (!f || std::fwrite(&id, sizeof(id), 1, f) != 1) { std::fprintf(stderr, "cannot write %s\n", tmp.c_str()); return 3; }
    std::fclose(f);
    std::rename(tmp.c_str(), idfile.c_str());
  } else {
    bool got = false;
    for (int t = 0; t < 1200 && !got; ++t) {  // up to 120 s
      std::FILE* f = std::fopen(idfile.c_str(), "rb");
      if (f) { got = std::fread(&id, sizeof(id), 1, f) == 1; std::fclose(f); }
      if (!got) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    if (!got) { std::fprintf(stderr, "rank %d: no NCCL id in %s after 120 s\n", rank, idfile.c_str()); return 3; }
  }
  stage("have the NCCL id");

  ncclComm_t comm = nullptr;
  const auto t0 = std::chrono::steady_clock::now();
  if (!nccl_ok(ncclCommInitRank(&comm, size, id, rank), comm, "ncclCommInitRank")) {
    if (rank == 0) std::printf("RESULT: FAIL ncclCommInitRank (see the NCCL WARN lines)\n");
    return 1;
  }
  const double tinit = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  stage("ncclCommInitRank done");
  if (rank == 0) std::printf("ncclCommInitRank: OK in %.2f s\n", tinit);
  std::fflush(stdout);
  if (rank == 0) std::remove(idfile.c_str());

  cudaStream_t st;
  cuda_ck(cudaStreamCreate(&st), "stream");
  double* dred = nullptr;  // small buffer for reductions over the ranks
  cuda_ck(cudaMalloc(&dred, 2 * sizeof(double)), "cudaMalloc");
  // [0] summed, [1] maximum over the ranks
  auto reduce = [&](double sum, double mx, double& rsum, double& rmax) -> bool {
    double hb[2] = {sum, mx};
    cuda_ck(cudaMemcpy(dred, hb, sizeof(hb), cudaMemcpyHostToDevice), "upload");
    bool g = nccl_ok(ncclAllReduce(dred, dred, 1, ncclDouble, ncclSum, comm, st), comm, "ncclAllReduce sum") &&
             nccl_ok(ncclAllReduce(dred + 1, dred + 1, 1, ncclDouble, ncclMax, comm, st), comm, "ncclAllReduce max");
    cuda_ck(cudaStreamSynchronize(st), "sync");
    cuda_ck(cudaMemcpy(hb, dred, sizeof(hb), cudaMemcpyDeviceToHost), "download");
    rsum = hb[0]; rmax = hb[1];
    return g;
  };

  std::vector<Exchange> ex;
  {
    Exchange a{"all-to-all, all ranks", {}, {}, true};
    for (int s = 1; s < size; ++s) { a.to.push_back((rank + s) % size); a.from.push_back((rank - s + size) % size); }
    ex.push_back(a);
    if (nnodes > 1)
      ex.push_back(Exchange{"inter-node ring (network only)",
                            {((mynode + 1) % nnodes) * nlocal + local},
                            {((mynode - 1 + nnodes) % nnodes) * nlocal + local}, true});
    Exchange c{"all-to-all inside each node (NVLink)", {}, {}, false};
    for (int s = 1; s < nlocal; ++s) {
      c.to.push_back(mynode * nlocal + (local + s) % nlocal);
      c.from.push_back(mynode * nlocal + (local - s + nlocal) % nlocal);
    }
    if (nlocal > 1) ex.push_back(c);
  }

  const size_t npeer = size_t(std::max(size - 1, 1));
  double *sbuf = nullptr, *rbuf = nullptr;
  cuda_ck(cudaMalloc(&sbuf, npeer * n * sizeof(double)), "cudaMalloc send");
  cuda_ck(cudaMalloc(&rbuf, npeer * n * sizeof(double)), "cudaMalloc recv");
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  std::vector<double> h(n);
  bool all_ok = true;

  for (const Exchange& X : ex) {
    const int np = int(X.to.size());
    for (int p = 0; p < np; ++p) {
      for (size_t i = 0; i < n; ++i) h[i] = pattern(rank, X.to[size_t(p)], i);
      cuda_ck(cudaMemcpy(sbuf + size_t(p) * n, h.data(), n * sizeof(double), cudaMemcpyHostToDevice), "upload");
    }
    cuda_ck(cudaMemset(rbuf, 0, size_t(np) * n * sizeof(double)), "memset");
    auto run = [&]() -> bool {
      bool g = nccl_ok(ncclGroupStart(), comm, "ncclGroupStart");
      for (int p = 0; p < np && g; ++p)
        g = nccl_ok(ncclSend(sbuf + size_t(p) * n, n, ncclDouble, X.to[size_t(p)], comm, st), comm, "ncclSend") &&
            nccl_ok(ncclRecv(rbuf + size_t(p) * n, n, ncclDouble, X.from[size_t(p)], comm, st), comm, "ncclRecv");
      return nccl_ok(ncclGroupEnd(), comm, "ncclGroupEnd") && g;
    };
    bool good = run() && run();  // warm-up
    cuda_ck(cudaStreamSynchronize(st), "sync");
    double bad = 0;
    if (good && X.check)
      for (int p = 0; p < np; ++p) {
        cuda_ck(cudaMemcpy(h.data(), rbuf + size_t(p) * n, n * sizeof(double), cudaMemcpyDeviceToHost), "download");
        for (size_t i = 0; i < n; ++i) bad += h[i] != pattern(X.from[size_t(p)], rank, i);
      }
    const int iters = 10;
    float ms = 0;
    double dummy, rmax;
    reduce(0, 0, dummy, rmax);  // all ranks start the timing together
    if (good) {
      cudaEventRecord(e0, st);
      for (int it = 0; it < iters && good; ++it) good = run();
      cudaEventRecord(e1, st);
      cuda_ck(cudaEventSynchronize(e1), "event");
      cudaEventElapsedTime(&ms, e0, e1);
    }
    double nbad, nfail;
    reduce(bad, ms / 1000.0 / iters, nbad, rmax);
    reduce(good ? 0 : 1, 0, nfail, dummy);
    stage(("done: " + X.name).c_str());
    if (rank == 0) {
      const double gb = double(np) * n * sizeof(double) / 1e9;  // sent by each rank
      if (nfail > 0)
        std::printf("RESULT %-38s FAIL (NCCL error on %.0f rank(s))\n", X.name.c_str(), nfail);
      else
        std::printf("RESULT %-38s %s  %8.3f ms per exchange, %7.2f GB sent per GPU, %8.1f GB/s per GPU%s\n",
                    X.name.c_str(), nbad > 0 ? "WRONG DATA" : "ok", 1000 * rmax, gb, gb / rmax,
                    X.check ? "" : " (not checked)");
      std::fflush(stdout);
    }
    all_ok = all_ok && nfail == 0 && nbad == 0;
    if (nfail > 0) break;
  }
  if (rank == 0) std::printf(all_ok ? "RESULT: OK\n" : "RESULT: FAIL\n");
  std::fflush(stdout);
  cudaFree(sbuf);
  cudaFree(rbuf);
  cudaFree(dred);
  ncclCommDestroy(comm);
  return all_ok ? 0 : 1;
}
