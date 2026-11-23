// MPI check across nodes: does MPI start and reach every rank? (FDCHAN uses MPI to start the ranks, for the
// reductions and the restart files, and with comm_backend = mpi for all communication.)
//
//   make MPI=1 MPICXX=mpicxx build/mpi_check        (CPU build is enough)
//   srun --mpi=pmix -l build/mpi_check
//
// Every rank prints how far it got; rank 0 prints "RESULT: OK" or the MPI error.
#include <mpi.h>
#include <unistd.h>

#include <cstdio>
#include <vector>

int main(int argc, char** argv) {
  std::fprintf(stderr, "stage: process started on host\n");
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  char host[64] = "?";
  gethostname(host, sizeof(host));
  std::fprintf(stderr, "stage rank %d of %d on %s: MPI_Init done\n", rank, size, host);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
  auto ck = [&](int e, const char* what) {
    if (e == MPI_SUCCESS) return true;
    char s[MPI_MAX_ERROR_STRING];
    int l;
    MPI_Error_string(e, s, &l);
    std::fprintf(stderr, "rank %d: %s failed: %s\n", rank, what, s);
    return false;
  };
  // every rank sends 1 MiB to every other rank, then a reduction
  const int n = 131072;
  std::vector<double> sb(size_t(n) * size, rank), rb(size_t(n) * size, -1.0);
  bool ok = ck(MPI_Alltoall(sb.data(), n, MPI_DOUBLE, rb.data(), n, MPI_DOUBLE, MPI_COMM_WORLD), "MPI_Alltoall");
  long bad = 0;
  for (int r = 0; r < size && ok; ++r)
    for (int i = 0; i < n; ++i) bad += rb[size_t(r) * n + i] != r;
  std::fprintf(stderr, "stage rank %d: all-to-all %s\n", rank, ok ? (bad ? "WRONG DATA" : "ok") : "failed");
  long tot = 0;
  ok = ck(MPI_Allreduce(&bad, &tot, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD), "MPI_Allreduce") && ok;
  if (rank == 0) std::printf(ok && tot == 0 ? "RESULT: OK (%d ranks)\n" : "RESULT: FAIL\n", size);
  std::fflush(stdout);
  MPI_Finalize();
  return ok && tot == 0 ? 0 : 1;
}
