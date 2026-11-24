// FDCHAN main program: [mpirun -n P] fdchan [input file, default input.txt]
#include <sys/stat.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <fstream>

#include "solver.hpp"

namespace fd {
#ifdef FD_GPU
gpuStream_t g_stream;
#endif
Comm g_comm;
}  // namespace fd

using namespace fd;

// Restart file: header (4 ints: nx, ny, nz (global), sizeof(real); 3 doubles: lx, lz, t), then u, v, w and the
// nscal scalars as global complex arrays (ny, nz, nkx), the same layout for any number of ranks. A file with fewer
// scalars (e.g. from a run without scalars) can be read: the missing scalars are initialised as given by sinit.
static void write_restart(Solver& S, const std::string& file) {
  const int nf = 3 + S.nsc;
  std::vector<std::vector<cplx>> h;
  h.push_back(S.u.host()); h.push_back(S.v.host()); h.push_back(S.w.host());
  for (int n = 0; n < S.nsc; ++n) h.push_back(S.Sf[n].host());
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
  for (int c = 0; c < nf; ++c) {
    MPI_File_set_view(fh, 40 + c * field, T, sub, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, h[size_t(c)].data(), int(2 * h[size_t(c)].size()), T, MPI_STATUS_IGNORE);
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
  for (int c = 0; c < nf; ++c) std::fwrite(h[size_t(c)].data(), sizeof(cplx), h[size_t(c)].size(), f);
  std::fclose(f);
  std::rename((file + ".tmp").c_str(), file.c_str());
#endif
}

static void read_restart(Solver& S, const std::string& file) {
  int hdr[4];
  double d[3];
  const int nf = 3 + S.nsc;
  std::vector<std::vector<cplx>> h(static_cast<size_t>(nf));
  for (auto& a : h) a.resize(size_t(S.nsp));
  const long long field = (long long)S.ny * S.nzg * S.nkx * 2 * (long long)sizeof(real);
  long long fsize = 0;
  int nread = 0;
#ifdef FD_MPI
  MPI_File fh;
  if (MPI_File_open(g_comm.comm, file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh) != MPI_SUCCESS)
    fatal(("cannot open restart file " + file).c_str());
  MPI_File_read_at_all(fh, 0, hdr, 4, MPI_INT, MPI_STATUS_IGNORE);
  MPI_File_read_at_all(fh, 16, d, 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
  if (hdr[0] != S.nx || hdr[1] != S.ny || hdr[2] != S.nzg || hdr[3] != int(sizeof(real)))
    fatal("restart file does not match the grid or the precision");
  {
    MPI_Offset sz;
    MPI_File_get_size(fh, &sz);
    fsize = (long long)sz;
  }
  nread = int(std::min<long long>(nf, (fsize - 40) / field));
  if (nread < 3) fatal("restart file too short");
  const MPI_Datatype T = sizeof(real) == 8 ? MPI_DOUBLE : MPI_FLOAT;
  int gs[3] = {S.ny, S.nzg, 2 * S.nkx}, ls[3] = {S.ny, S.nz, 2 * S.nkx}, st[3] = {0, S.z0, 0};
  MPI_Datatype sub;
  MPI_Type_create_subarray(3, gs, ls, st, MPI_ORDER_C, T, &sub);
  MPI_Type_commit(&sub);
  for (int c = 0; c < nread; ++c) {
    MPI_File_set_view(fh, 40 + c * MPI_Offset(field), T, sub, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, h[size_t(c)].data(), int(2 * h[size_t(c)].size()), T, MPI_STATUS_IGNORE);
  }
  MPI_Type_free(&sub);
  MPI_File_close(&fh);
#else
  std::FILE* f = std::fopen(file.c_str(), "rb");
  if (!f) fatal(("cannot open restart file " + file).c_str());
  if (std::fread(hdr, sizeof(int), 4, f) != 4 || std::fread(d, sizeof(double), 3, f) != 3) fatal("restart header");
  if (hdr[0] != S.nx || hdr[1] != S.ny || hdr[2] != S.nzg || hdr[3] != int(sizeof(real)))
    fatal("restart file does not match the grid or the precision");
  std::fseek(f, 0, SEEK_END);
  fsize = std::ftell(f);
  std::fseek(f, 40, SEEK_SET);
  nread = int(std::min<long long>(nf, (fsize - 40) / field));
  if (nread < 3) fatal("restart file too short");
  for (int c = 0; c < nread; ++c)
    if (std::fread(h[size_t(c)].data(), sizeof(cplx), h[size_t(c)].size(), f) != h[size_t(c)].size())
      fatal("restart data");
  std::fclose(f);
#endif
  S.set_fields(h[0], h[1], h[2]);
  for (int n = 0; n < S.nsc; ++n) {
    if (3 + n < nread) S.Sf[n].upload(h[size_t(3 + n)]);
    else {
      if (g_comm.root())
        std::printf("restart: scalar %d not in %s, initialised with sinit = %s\n", n + 1, file.c_str(),
                    S.P.sinit[n].c_str());
      S.init_scalar(n);
    }
  }
  if (!S.P.reset_time) S.time = d[2];
}

int main(int argc, char** argv) {
  g_comm.init(&argc, &argv);
  const bool root = g_comm.root();
  const std::string inp = argc > 1 ? argv[1] : "input.txt";
  Params P;
  P.read(inp);
  g_timer.on = P.timing;
  g_timer.sync_gpu = P.timing_sync;
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
  if (P.comm_backend.empty()) P.comm_backend = Comm::nccl_built() ? "nccl" : "mpi";
  g_comm.init_nccl(P.comm_backend == "nccl");
  if (root)
    std::printf("FDCHAN (GPU build, %d rank(s), %s precision, halos and all-to-alls with %s)\n", g_comm.size,
                sizeof(real) == 8 ? "double" : "single", g_comm.size == 1 ? "local copies" : g_comm.use_nccl ? "NCCL" : "MPI");
#else
  const int nt = P.nthreads > 0 ? P.nthreads : int(std::max(1u, std::thread::hardware_concurrency()));
  Pool::get().set_threads(nt);
  if (P.comm_backend.empty()) P.comm_backend = Comm::nccl_built() ? "nccl" : "mpi";
  g_comm.init_nccl(P.comm_backend == "nccl");  // CPU: only with the test stand-in (tests/nccl_mock.hpp)
  if (root)
    std::printf("FDCHAN (CPU build, %d rank(s) x %d threads, %s precision%s)\n", g_comm.size, nt,
                sizeof(real) == 8 ? "double" : "single", g_comm.use_nccl ? ", NCCL test stand-in" : "");
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
  // log.dat and stats.dat in the format of HiCHAN (columns, headers, number formats)
  std::FILE* lg = nullptr;
  if (root) {
    lg = std::fopen((P.data_dir + "/log.dat").c_str(), P.init_type == "restart" ? "a" : "w");
    if (!lg) fatal("cannot open log.dat");
    std::fprintf(lg, "#    step          time            dt         ubulk  dpdx_applied         tau_w        Re_tau"
                     "     div_max          Ekin        E_mode");
    for (int n = 1; n <= S.nsc; ++n) std::fprintf(lg, "      dS%ddy_b      dS%ddy_t", n, n);
    std::fprintf(lg, "\n");
  }
  auto log_line = [&]() {
    Solver::Log L = S.diagnostics();  // collective
    std::vector<double> sg;            // dS/dy of the mean scalars at the bottom and top walls
    for (int n = 0; n < S.nsc; ++n) {
      const std::vector<double> g = S.D1.apply(S.mean_of(S.Sf[n].p));
      sg.push_back(g[0]);
      sg.push_back(g[size_t(S.ny - 1)]);
    }
    if (!root) return;
    // HiCHAN: (i9, 6es14.6, es11.3, 2es14.6), then 2es14.6 per scalar
    std::fprintf(lg, "%9ld%14.6E%14.6E%14.6E%14.6E%14.6E%14.6E%11.3E%14.6E%14.6E", S.step, S.time, S.dt, L.ub,
                 S.dpdx_step, L.tau_w, L.retau, L.div, L.ef, L.emode);
    for (double x : sg) std::fprintf(lg, "%14.6E", x);
    std::fprintf(lg, "\n");
    std::fflush(lg);
    std::printf("step %8ld  t = %.5f  dt = %.3e  Ub = %.6f  -dp/dx = %.6e  Re_tau = %.2f  div = %.1e  E' = %.4e\n",
                S.step, S.time, S.dt, L.ub, S.dpdx_step, L.retau, L.div, L.ef);
    std::fflush(stdout);
  };
  log_line();
  // statistics, as HiCHAN: time-weighted averages (weight dt * iout_stats) of U V W, <u'u'> <v'v'> <w'w'> <u'v'>
  // <u'w'> <v'w'>, P and <p'p'> (P: the pressure of the last RK stage; with nl_form = rotational the total
  // pressure p + |u|^2/2). Sampled right after the step, before the log line reuses the array that holds phi.
  // per scalar n (as HiCHAN): S<n>, <s'_n s'_n>, <u's'_n>, <v's'_n>, <w's'_n>
  const int NQ = 11 + 5 * S.nsc;
  std::vector<std::string> qname = {"U", "V", "W", "uu", "vv", "ww", "uv", "uw", "vw", "P", "pp"};
  for (int n = 1; n <= S.nsc; ++n) {
    const std::string c = std::to_string(n);
    for (const std::string& q : {"S" + c, "s" + c + "s" + c, "us" + c, "vs" + c, "ws" + c}) qname.push_back(q);
  }
  std::vector<double> st(size_t(NQ) * S.ny, 0.0);
  double tsum = 0;
  long nsamp = 0;
  auto sample = [&]() {
    const double wgt = S.dt * P.iout_stats;
    std::vector<double> m[3] = {S.mean_profile(0), S.mean_profile(1), S.mean_profile(2)};
    std::vector<double> c[6] = {S.profile(0, 0, false), S.profile(1, 1, false), S.profile(2, 2, false),
                                S.profile(0, 1, true),  S.profile(0, 2, true),  S.profile(1, 2, true)};
    const int ca[6] = {0, 1, 2, 0, 0, 1}, cb[6] = {0, 1, 2, 1, 2, 2};
    const cplx* phi = S.T1.p;  // phi of the last stage (see Solver::pscale)
    std::vector<double> pm = S.mean_of(phi), pp = S.profile_of(phi, phi);
    const double ps = S.pscale;
    for (int j = 0; j < S.ny; ++j) {
      const size_t jj = size_t(j), o = size_t(NQ) * jj;
      for (int q = 0; q < 3; ++q) st[o + size_t(q)] += wgt * m[q][jj];
      for (int q = 0; q < 6; ++q) st[o + 3 + size_t(q)] += wgt * (c[q][jj] - m[ca[q]][jj] * m[cb[q]][jj]);
      st[o + 9] += wgt * ps * pm[jj];
      st[o + 10] += wgt * ps * ps * (pp[jj] - pm[jj] * pm[jj]);
    }
    const cplx* F[3] = {S.u.p, S.v.p, S.w.p};
    for (int n = 0; n < S.nsc; ++n) {
      const cplx* sf = S.Sf[n].p;
      const std::vector<double> sm = S.mean_of(sf), ss = S.profile_of(sf, sf);
      std::vector<double> us[3];
      for (int c = 0; c < 3; ++c) us[c] = S.profile_of(F[c], sf);
      for (int j = 0; j < S.ny; ++j) {
        const size_t jj = size_t(j), o = size_t(NQ) * jj + 11 + 5 * size_t(n);
        st[o] += wgt * sm[jj];
        st[o + 1] += wgt * (ss[jj] - sm[jj] * sm[jj]);
        for (int c = 0; c < 3; ++c) st[o + 2 + size_t(c)] += wgt * (us[c][jj] - m[c][jj] * sm[jj]);
      }
    }
    tsum += wgt;
    ++nsamp;
  };
  // Fortran ESw.dEe (e.g. HiCHAN's es16.7e3): d decimals, e exponent digits, right-aligned in w
  auto fes = [](double v, int w, int d, int e) {
    char m[64], out[96];
    std::snprintf(m, sizeof(m), "%.*E", d, v);
    char* ep = std::strchr(m, 'E');
    const int ex = std::atoi(ep + 1);
    *ep = '\0';
    std::snprintf(out, sizeof(out), "%sE%c%0*d", m, ex < 0 ? '-' : '+', e, ex < 0 ? -ex : ex);
    std::string r(out);
    if (int(r.size()) < w) r.insert(0, size_t(w) - r.size(), ' ');
    return r;
  };
  auto save_stats = [&]() {
    if (nsamp == 0 || !root) return;
    std::FILE* f = std::fopen((P.data_dir + "/stats.dat").c_str(), "w");
    std::fprintf(f, "# time-averaged profiles; t_end = %14.6E, averaging time = %14.6E, samples = %ld\n", S.time, tsum,
                 nsamp);
    std::fprintf(f, "# fluctuation moments are <f'g'>; P and pp are the (total, if nl_form=rotational) pressure\n");
    std::fprintf(f, "#            y");
    for (int q = 0; q < NQ; ++q) std::fprintf(f, "%16s", qname[size_t(q)].c_str());
    std::fprintf(f, " \n");
    for (int j = 0; j < S.ny; ++j) {
      std::string line = fes(S.y[size_t(j)], 16, 7, 3);
      for (int q = 0; q < NQ; ++q) line += fes(st[size_t(NQ) * size_t(j) + size_t(q)] / tsum, 16, 7, 3);
      std::fprintf(f, "%s\n", line.c_str());
    }
    std::fclose(f);
  };
  g_comm.barrier();
  const auto w0 = std::chrono::steady_clock::now();
  auto wall = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - w0).count(); };
  long n0 = S.step;
  bool go = true;
  g_timer.start(P.data_dir);
  while (go) {
    if (S.step - n0 >= P.nstep || S.time >= P.tmax - 1e-12) break;
    if (P.fixed_dt && S.time + P.dt > P.tmax + 1e-12) break;
    g_timer.tic(tm_rk);
    S.advance();
    g_timer.toc(tm_rk);
    // statistics first: they need phi of the last stage, which the log line overwrites
    g_timer.tic(tm_stats);
    if (P.iout_stats > 0 && S.step % P.iout_stats == 0 && S.time >= P.tstats_start) sample();
    if (P.iout_save_stats > 0 && S.step % P.iout_save_stats == 0) save_stats();
    g_timer.toc(tm_stats);
    if (P.iout_log > 0 && S.step % P.iout_log == 0) {
      g_timer.tic(tm_log);
      log_line();
      g_timer.toc(tm_log);
    }
    if (P.iout_restart > 0 && S.step % P.iout_restart == 0) {
      g_timer.tic(tm_io);
      write_restart(S, P.data_dir + "/restart.bin");
      g_timer.toc(tm_io);
    }
    // the same wall-clock decision on all ranks
    go = g_comm.allmax(wall() < P.wall_time_max ? 0.0 : 1.0) == 0.0;
    g_timer.step(S.step, S.time, S.dt);
  }
  sync();
  g_comm.barrier();
  const double tw = wall();
  const long ns = S.step - n0;
  if (P.iout_log <= 0 || S.step % P.iout_log != 0) {
    g_timer.tic(tm_log);
    log_line();
    g_timer.toc(tm_log);
  }
  g_timer.tic(tm_stats);
  save_stats();
  g_timer.toc(tm_stats);
  g_timer.tic(tm_io);
  write_restart(S, P.data_dir + "/restart.bin");
  g_timer.toc(tm_io);
  if (root) {
    std::fclose(lg);
    std::printf("%ld steps in %.3f s: %.4e s per step, %.3e s per step and grid point (nx*ny*nz)\n", ns, tw,
                tw / std::max(ns, 1L), tw / std::max(ns, 1L) / (double(S.nx) * S.ny * S.nzg));
  }
  // the time loop of the report includes the final output (log line, statistics, restart file)
  g_timer.report(ns, double(S.nx) * S.ny * S.nzg, P.data_dir);
  g_comm.finalize();
  return 0;
}
