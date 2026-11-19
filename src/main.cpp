// FDCHAN main program: [mpirun -n P] fdchan [input file, default input.txt]
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <fstream>

#include "solver.hpp"

namespace fd {
#ifdef FD_GPU
gpuStream_t g_stream;
#endif
Comm g_comm;
}  // namespace fd

using namespace fd;

// Restart file: header (4 ints: nx, ny, nz (global), sizeof(real); 3 doubles: lx, lz, t), then u, v, w as global
// complex arrays (ny, nz, nkx), the same layout for any number of ranks.
static void write_restart(Solver& S, const std::string& file) {
  std::vector<cplx> h[3] = {S.u.host(), S.v.host(), S.w.host()};
  const int hdr[4] = {S.nx, S.ny, S.nzg, int(sizeof(real))};
  const double d[3] = {S.P.lx, S.P.lz, S.time};
#ifdef FD_MPI
  MPI_File fh;
  const std::string tmp = file + ".tmp";
  if (g_comm.root()) std::remove(tmp.c_str());
  g_comm.barrier();
  MPI_File_open(g_comm.comm, tmp.c_str(), MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);
  if (g_comm.root()) {
    MPI_File_write_at(fh, 0, hdr, 4, MPI_INT, MPI_STATUS_IGNORE);
    MPI_File_write_at(fh, 16, d, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
  }
  const MPI_Datatype T = sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT;
  int gs[3] = {S.ny, S.nzg, 2 * S.nkx}, ls[3] = {S.ny, S.nz, 2 * S.nkx}, st[3] = {0, S.z0, 0};
  MPI_Datatype sub;
  MPI_Type_create_subarray(3, gs, ls, st, MPI_ORDER_C, T, &sub);
  MPI_Type_commit(&sub);
  const MPI_Offset field = MPI_Offset(S.ny) * S.nzg * S.nkx * 2 * sizeof(real);
  for (int c = 0; c < 3; ++c) {
    MPI_File_set_view(fh, 40 + c * field, T, sub, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, h[c].data(), int(2 * h[c].size()), T, MPI_STATUS_IGNORE);
  }
  MPI_Type_free(&sub);
  MPI_File_close(&fh);
  g_comm.barrier();
  if (g_comm.root()) std::rename(tmp.c_str(), file.c_str());
  g_comm.barrier();
#else
  std::FILE* f = std::fopen((file + ".tmp").c_str(), "wb");
  if (!f) fatal("cannot write restart file");
  std::fwrite(hdr, sizeof(int), 4, f);
  std::fwrite(d, sizeof(double), 3, f);
  for (auto& a : h) std::fwrite(a.data(), sizeof(cplx), a.size(), f);
  std::fclose(f);
  std::rename((file + ".tmp").c_str(), file.c_str());
#endif
}

static void read_restart(Solver& S, const std::string& file) {
  int hdr[4];
  double d[3];
  std::vector<cplx> h[3];
  for (auto& a : h) a.resize(size_t(S.nsp));
#ifdef FD_MPI
  MPI_File fh;
  if (MPI_File_open(g_comm.comm, file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    fatal(("cannot open restart file " + file).c_str());
  MPI_File_read_at_all(fh, 0, hdr, 4, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_read_at_all(fh, 16, d, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
  if (hdr[0] != S.nx || hdr[1] != S.ny || hdr[2] != S.nzg || hdr[3] != int(sizeof(real)))
    fatal("restart file does not match the grid or the precision");
  const MPI_Datatype T = sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT;
  int gs[3] = {S.ny, S.nzg, 2 * S.nkx}, ls[3] = {S.ny, S.nz, 2 * S.nkx}, st[3] = {0, S.z0, 0};
  MPI_Datatype sub;
  MPI_Type_create_subarray(3, gs, ls, st, MPI_ORDER_C, T, &sub);
  MPI_Type_commit(&sub);
  const MPI_Offset field = MPI_Offset(S.ny) * S.nzg * S.nkx * 2 * sizeof(real);
  for (int c = 0; c < 3; ++c) {
    MPI_File_set_view(fh, 40 + c * field, T, sub, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, h[c].data(), int(2 * h[c].size()), T, MPI_STATUS_IGNORE);
  }
  MPI_Type_free(&sub);
  MPI_File_close(&fh);
#else
  std::FILE* f = std::fopen(file.c_str(), "rb");
  if (!f) fatal(("cannot open restart file " + file).c_str());
  if (std::fread(hdr, sizeof(int), 4, f) != 4 || std::fread(d, sizeof(double), 3, f) != 3) fatal("restart header");
  if (hdr[0] != S.nx || hdr[1] != S.ny || hdr[2] != S.nzg || hdr[3] != int(sizeof(real)))
    fatal("restart file does not match the grid or the precision");
  for (auto& a : h)
    if (std::fread(a.data(), sizeof(cplx), a.size(), f) != a.size()) fatal("restart data");
  std::fclose(f);
#endif
  S.set_fields(h[0], h[1], h[2]);
  if (!S.P.reset_time) S.time = d[2];
}

int main(int argc, char** argv) {
  g_comm.init(&argc, &argv);
  const bool root = g_comm.root();
  const std::string inp = argc > 1 ? argv[1] : "input.txt";
  Params P;
  P.read(inp);
#ifdef FD_GPU
  {
    int ndev = 1, local = 0;
#if defined(FD_CUDA)
    cudaGetDeviceCount(&ndev);
#else
    hipGetDeviceCount(&ndev);
#endif
#ifdef FD_MPI
    MPI_Comm node;
    MPI_Comm_split_type(g_comm.comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node);
    MPI_Comm_rank(node, &local);
    MPI_Comm_free(&node);
#endif
#if defined(FD_CUDA)
    cudaSetDevice(local % std::max(ndev, 1));
#else
    hipSetDevice(local % std::max(ndev, 1));
#endif
  }
  gpu_check(gpuStreamCreate(&g_stream), "stream");
  if (root) std::printf("FDCHAN (GPU build, %d rank(s), %s precision)\n", g_comm.size, sizeof(real) == 8 ? "double" : "single");
#else
  const int nt = P.nthreads > 0 ? P.nthreads : int(std::max(1u, std::thread::hardware_concurrency()));
  Pool::get().set_threads(nt);
  if (root)
    std::printf("FDCHAN (CPU build, %d rank(s) x %d threads, %s precision)\n", g_comm.size, nt,
                sizeof(real) == 8 ? "double" : "single");
#endif
  if (root) P.print();
  if (root) mkdir(P.data_dir.c_str(), 0755);
  g_comm.barrier();
  Solver S(P);
  if (root) S.print_setup();
  if (P.init_type == "restart") {
    read_restart(S, P.restart_file);
    S.project_current();
  } else {
    S.init_flow();
  }
  std::FILE* lg = nullptr;
  if (root) {
    lg = std::fopen((P.data_dir + "/log.dat").c_str(), P.init_type == "restart" ? "a" : "w");
    if (!lg) fatal("cannot open log.dat");
    std::fprintf(lg, "# step t dt ubulk -dpdx tau_w(bottom) tau_w(top) re_tau max|div| E_fluct E_v(mode_log_kx)\n");
  }
  auto log_line = [&]() {
    Solver::Log L = S.diagnostics();  // collective
    if (!root) return;
    std::fprintf(lg, "%8ld %.10e %.6e %.10e %.10e %.10e %.10e %.6e %.3e %.10e %.10e\n", S.step, S.time, S.dt, L.ub,
                 S.dpdx_step, L.tau_b, L.tau_t, L.retau, L.div, L.ef, L.emode);
    std::fflush(lg);
    std::printf("step %8ld  t = %.5f  dt = %.3e  Ub = %.6f  -dp/dx = %.6e  Re_tau = %.2f  div = %.1e  E' = %.4e\n",
                S.step, S.time, S.dt, L.ub, S.dpdx_step, L.retau, L.div, L.ef);
    std::fflush(stdout);
  };
  log_line();
  // statistics: U V W and <u'u'> <v'v'> <w'w'> <u'v'>
  std::vector<double> st(size_t(8 * S.ny), 0.0);
  long nsamp = 0;
  auto sample = [&]() {
    std::vector<double> m[3] = {S.mean_profile(0), S.mean_profile(1), S.mean_profile(2)};
    std::vector<double> uu = S.profile(0, 0, false), vv = S.profile(1, 1, false), ww = S.profile(2, 2, false),
                        uv = S.profile(0, 1, true);
    for (int j = 0; j < S.ny; ++j) {
      const size_t o = size_t(8 * j);
      st[o + 0] += m[0][size_t(j)];
      st[o + 1] += m[1][size_t(j)];
      st[o + 2] += m[2][size_t(j)];
      st[o + 3] += uu[size_t(j)] - m[0][size_t(j)] * m[0][size_t(j)];
      st[o + 4] += vv[size_t(j)] - m[1][size_t(j)] * m[1][size_t(j)];
      st[o + 5] += ww[size_t(j)] - m[2][size_t(j)] * m[2][size_t(j)];
      st[o + 6] += uv[size_t(j)] - m[0][size_t(j)] * m[1][size_t(j)];
    }
    ++nsamp;
  };
  auto save_stats = [&]() {
    if (nsamp == 0 || !root) return;
    std::FILE* f = std::fopen((P.data_dir + "/stats.dat").c_str(), "w");
    std::fprintf(f, "# %ld samples, t = %.6f\n# y U V W <u'u'> <v'v'> <w'w'> <u'v'>\n", nsamp, S.time);
    for (int j = 0; j < S.ny; ++j) {
      std::fprintf(f, "%.10e", S.y[size_t(j)]);
      for (int c = 0; c < 7; ++c) std::fprintf(f, " %.10e", st[size_t(8 * j + c)] / nsamp);
      std::fprintf(f, "\n");
    }
    std::fclose(f);
  };
  g_comm.barrier();
  const auto w0 = std::chrono::steady_clock::now();
  auto wall = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(); };
  long n0 = S.step;
  bool go = true;
  while (go) {
    if (S.step - n0 >= P.nstep || S.time >= P.tmax - 1e-12) break;
    if (P.fixed_dt && S.time + P.dt > P.tmax + 1e-12) break;
    S.advance();
    if (P.iout_log > 0 && S.step % P.iout_log == 0) log_line();
    if (P.iout_stats > 0 && S.step % P.iout_stats == 0 && S.time >= P.tstats_start) sample();
    if (P.iout_save_stats > 0 && S.step % P.iout_save_stats == 0) save_stats();
    if (P.iout_restart > 0 && S.step % P.iout_restart == 0) write_restart(S, P.data_dir + "/restart.bin");
    // the same wall-clock decision on all ranks
    go = g_comm.allmax(wall() < P.wall_time_max ? 0.0 : 1.0) == 0.0;
  }
  sync();
  g_comm.barrier();
  const double tw = wall();
  const long ns = S.step - n0;
  if (P.iout_log <= 0 || S.step % P.iout_log != 0) log_line();
  save_stats();
  write_restart(S, P.data_dir + "/restart.bin");
  if (root) {
    std::fclose(lg);
    std::printf("%ld steps in %.3f s: %.4e s per step, %.3e s per step and grid point (nx*ny*nz)\n", ns, tw,
                tw / std::max(ns, 1L), tw / std::max(ns, 1L) / (double(S.nx) * S.ny * S.nzg));
    if (P.timing && ns > 0)
      std::printf("per step (rank 0): nonlinear+FFT %.3e s, rhs %.3e s, pressure %.3e s, other %.3e s, of which "
                  "communication %.3e s\n",
                  S.t_nl / ns, S.t_rhs / ns, S.t_p / ns, S.t_other / ns, S.t_comm / ns);
  }
  g_comm.finalize();
  return 0;
}
