// FDCHAN main program: fdchan [input file, default input.txt]
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <fstream>

#include "solver.hpp"

#ifdef FD_GPU
namespace fd {
gpuStream_t g_stream;
}
#endif

using namespace fd;

static void write_restart(Solver& S, const std::string& file) {
  std::vector<cplx> hu = S.u.host(), hv = S.v.host(), hw = S.w.host();
  std::FILE* f = std::fopen((file + ".tmp").c_str(), "wb");
  if (!f) fatal("cannot write restart file");
  const int hdr[4] = {S.nx, S.ny, S.nz, int(sizeof(real))};
  const double d[3] = {S.P.lx, S.P.lz, S.time};
  std::fwrite(hdr, sizeof(int), 4, f);
  std::fwrite(d, sizeof(double), 3, f);
  for (auto* a : {&hu, &hv, &hw}) std::fwrite(a->data(), sizeof(cplx), a->size(), f);
  std::fclose(f);
  std::rename((file + ".tmp").c_str(), file.c_str());
}

static void read_restart(Solver& S, const std::string& file) {
  std::FILE* f = std::fopen(file.c_str(), "rb");
  if (!f) fatal(("cannot open restart file " + file).c_str());
  int hdr[4];
  double d[3];
  if (std::fread(hdr, sizeof(int), 4, f) != 4 || std::fread(d, sizeof(double), 3, f) != 3) fatal("restart header");
  if (hdr[0] != S.nx || hdr[1] != S.ny || hdr[2] != S.nz || hdr[3] != int(sizeof(real)))
    fatal("restart file does not match the grid or the precision");
  std::vector<cplx> h[3];
  for (auto& a : h) {
    a.resize(size_t(S.nsp));
    if (std::fread(a.data(), sizeof(cplx), a.size(), f) != a.size()) fatal("restart data");
  }
  std::fclose(f);
  S.set_fields(h[0], h[1], h[2]);
  if (!S.P.reset_time) S.time = d[2];
}

int main(int argc, char** argv) {
  const std::string inp = argc > 1 ? argv[1] : "input.txt";
  Params P;
  P.read(inp);
#ifdef FD_GPU
  gpu_check(gpuStreamCreate(&g_stream), "stream");
  std::printf("FDCHAN (GPU build, %s precision)\n", sizeof(real) == 8 ? "double" : "single");
#else
  const int nt = P.nthreads > 0 ? P.nthreads : int(std::max(1u, std::thread::hardware_concurrency()));
  Pool::get().set_threads(nt);
  std::printf("FDCHAN (CPU build, %d threads, %s precision)\n", nt, sizeof(real) == 8 ? "double" : "single");
#endif
  P.print();
  mkdir(P.data_dir.c_str(), 0755);
  Solver S(P);
  S.print_setup();
  if (P.init_type == "restart") {
    read_restart(S, P.restart_file);
    S.project_current();
  } else {
    S.init_flow();
  }
  std::FILE* lg = std::fopen((P.data_dir + "/log.dat").c_str(), P.init_type == "restart" ? "a" : "w");
  if (!lg) fatal("cannot open log.dat");
  std::fprintf(lg, "# step t dt ubulk -dpdx tau_w(bottom) tau_w(top) re_tau max|div| E_fluct E_v(mode_log_kx)\n");
  auto log_line = [&]() {
    Solver::Log L = S.diagnostics();
    std::fprintf(lg, "%8ld %.10e %.6e %.10e %.10e %.10e %.10e %.6e %.3e %.10e %.10e\n", S.step, S.time, S.dt, L.ub,
                 S.dpdx_step, L.tau_b, L.tau_t, L.retau, L.div, L.ef, L.emode);
    std::fflush(lg);
    std::printf("step %8ld  t = %.5f  dt = %.3e  Ub = %.6f  -dp/dx = %.6e  Re_tau = %.2f  div = %.1e  E' = %.4e\n",
                S.step, S.time, S.dt, L.ub, S.dpdx_step, L.retau, L.div, L.ef);
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
    if (nsamp == 0) return;
    std::FILE* f = std::fopen((P.data_dir + "/stats.dat").c_str(), "w");
    std::fprintf(f, "# %ld samples, t = %.6f\n# y U V W <u'u'> <v'v'> <w'w'> <u'v'>\n", nsamp, S.time);
    for (int j = 0; j < S.ny; ++j) {
      std::fprintf(f, "%.10e", S.y[size_t(j)]);
      for (int c = 0; c < 7; ++c) std::fprintf(f, " %.10e", st[size_t(8 * j + c)] / nsamp);
      std::fprintf(f, "\n");
    }
    std::fclose(f);
  };
  const auto w0 = std::chrono::steady_clock::now();
  auto wall = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(); };
  long n0 = S.step;
  while (S.step - n0 < P.nstep && S.time < P.tmax - 1e-12 && wall() < P.wall_time_max) {
    if (P.fixed_dt && S.time + P.dt > P.tmax + 1e-12) break;
    S.advance();
    if (P.iout_log > 0 && S.step % P.iout_log == 0) log_line();
    if (P.iout_stats > 0 && S.step % P.iout_stats == 0 && S.time >= P.tstats_start) sample();
    if (P.iout_save_stats > 0 && S.step % P.iout_save_stats == 0) save_stats();
    if (P.iout_restart > 0 && S.step % P.iout_restart == 0) write_restart(S, P.data_dir + "/restart.bin");
  }
  sync();
  const double tw = wall();
  const long ns = S.step - n0;
  if (P.iout_log <= 0 || S.step % P.iout_log != 0) log_line();
  save_stats();
  write_restart(S, P.data_dir + "/restart.bin");
  std::fclose(lg);
  std::printf("%ld steps in %.3f s: %.4e s per step, %.3e s per step and grid point (nx*ny*nz)\n", ns, tw,
              tw / std::max(ns, 1L), tw / std::max(ns, 1L) / (double(S.nx) * S.ny * S.nz));
  if (P.timing && ns > 0)
    std::printf("per step: nonlinear+FFT %.3e s, rhs %.3e s, pressure %.3e s, other %.3e s\n", S.t_nl / ns,
                S.t_rhs / ns, S.t_p / ns, S.t_other / ns);
  return 0;
}
