// Optional timers (timing = 1 in the input), as in HiCHAN's mod_timer:
//   * accumulated wall-clock time of the main parts of the code, reported at the end of the run (stdout and
//     data_dir/timing_report.txt), with the number of calls and the min/mean/max over the MPI ranks;
//   * wall-clock time of every time step (max/min/mean over the ranks) and of its main parts, written to
//     data_dir/timing_steps.dat, to look for spikes in the performance; the report lists the slowest steps.
//
// GPU kernels, FFTs and copies run asynchronously on the solver stream. With timing_sync = 1 (default) every
// timer waits for the GPU when it starts and stops, so that the time is attributed to the right part; this
// removes some overlap of kernel launches. With timing_sync = 0 the parts are less exact, but the time per step
// is exact (every step ends with a wait). The results do not depend on the timing settings.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <string>
#include <vector>

#include "comm.hpp"

namespace fd {

enum TimerId {
  tm_loop, tm_rk, tm_nl, tm_dt, tm_rhs, tm_p, tm_cfr,
  tm_xfft, tm_zfft, tm_a2a, tm_halo,
  tm_log, tm_stats, tm_io, NTM
};

struct Timer {
  bool on = false, sync_gpu = true;
  double ttot[NTM] = {}, tbeg[NTM] = {};
  long ncall[NTM] = {};

  static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  }
  void tic(int id) {
    if (!on) return;
    if (sync_gpu) sync();
    tbeg[id] = now();
  }
  void toc(int id) {
    if (!on) return;
    if (sync_gpu) sync();
    ttot[id] += now() - tbeg[id];
    ++ncall[id];
  }

  // ------------------------------------------------------------ per-step records
  // columns of timing_steps.dat after step, t, dt: step time (max, min, mean over the ranks), then the time of
  // the step (max over the ranks) in the nonlinear terms, the pressure, all-to-alls, halo exchanges and output
  static constexpr int NPART = 5, NCOL = 1 + NPART, NBUF = 200;
  double part(int c) const {
    switch (c) {
      case 0: return ttot[tm_nl];
      case 1: return ttot[tm_p];
      case 2: return ttot[tm_a2a];
      case 3: return ttot[tm_halo];
      default: return ttot[tm_log] + ttot[tm_stats] + ttot[tm_io];
    }
  }
  double prev[NPART] = {}, mark = 0;
  std::vector<double> buf, bufh;  // NCOL and 3 values per buffered step
  std::vector<double> hist;       // rank 0: step, t_max, t_min, t_mean, parts (max) of every step
  std::FILE* fs = nullptr;

  // call just before the time loop (all ranks)
  void start(const std::string& dir) {
    if (!on) return;
    sync();
    if (g_comm.root()) {
      fs = std::fopen((dir + "/timing_steps.dat").c_str(), "w");
      if (!fs) fatal("cannot open timing_steps.dat");
      std::fprintf(fs, "# wall-clock time per time step [s]: max/min/mean over the MPI ranks, and the time (max over the "
                       "ranks) spent in parts of the step\n");
      std::fprintf(fs, "#    step          time            dt    t_step_max    t_step_min   t_step_mean"
                       "   t_nonlinear    t_pressure   t_alltoall       t_halo      t_output\n");
    }
    for (int c = 0; c < NPART; ++c) prev[c] = part(c);
    mark = now();
    tic(tm_loop);
  }
  // call at the end of every time step (all ranks)
  void step(long istep, double t, double dt) {
    if (!on) return;
    sync();
    buf.push_back(now() - mark);
    for (int c = 0; c < NPART; ++c) { buf.push_back(part(c) - prev[c]); prev[c] = part(c); }
    bufh.push_back(double(istep)); bufh.push_back(t); bufh.push_back(dt);
    if (int(bufh.size()) / 3 == NBUF) flush();
    mark = now();  // the flush is not counted in the next step
  }
  void flush() {
    const int nb = int(bufh.size()) / 3;
    if (nb == 0) return;
    std::vector<double> mx = buf, mn(static_cast<size_t>(nb)), sm(static_cast<size_t>(nb));
    for (int j = 0; j < nb; ++j) mn[size_t(j)] = sm[size_t(j)] = buf[size_t(j * NCOL)];
    g_comm.allmax(mx.data(), int(mx.size()));
    g_comm.allmin(mn.data(), nb);
    g_comm.allsum(sm.data(), nb);
    if (g_comm.root()) {
      for (int j = 0; j < nb; ++j) {
        const double* m = &mx[size_t(j * NCOL)];
        const double mean = sm[size_t(j)] / g_comm.size;
        std::fprintf(fs, "%9ld %13.6e %13.6e", long(bufh[size_t(3 * j)]), bufh[size_t(3 * j + 1)], bufh[size_t(3 * j + 2)]);
        std::fprintf(fs, " %13.5e %13.5e %13.5e", m[0], mn[size_t(j)], mean);
        for (int c = 1; c < NCOL; ++c) std::fprintf(fs, " %13.5e", m[c]);
        std::fprintf(fs, "\n");
        hist.push_back(bufh[size_t(3 * j)]);
        hist.push_back(m[0]);
        hist.push_back(mn[size_t(j)]);
        hist.push_back(mean);
        for (int c = 1; c < NCOL; ++c) hist.push_back(m[c]);
      }
      std::fflush(fs);
    }
    buf.clear();
    bufh.clear();
  }

  // ------------------------------------------------------------ report
  // call after the time loop and the final output (all ranks)
  void report(long nsteps, double npoints, const std::string& dir) {
    if (!on) return;
    toc(tm_loop);
    flush();
    const int np = g_comm.size;
    double tmin[NTM], tmax[NTM], tsum[NTM];
    for (int i = 0; i < NTM; ++i) tmin[i] = tmax[i] = tsum[i] = ttot[i];
    g_comm.allmin(tmin, NTM);
    g_comm.allmax(tmax, NTM);
    g_comm.allsum(tsum, NTM);
    if (!g_comm.root()) return;
    if (fs) std::fclose(fs);
    fs = nullptr;
    std::FILE* fr = std::fopen((dir + "/timing_report.txt").c_str(), "w");
    std::FILE* out[2] = {stdout, fr};
    const double tloop = std::max(tsum[tm_loop] / np, 1e-30), ns = double(std::max(nsteps, 1L));
    struct Row { const char* name; int id; };
    static const Row rows[] = {
        {"time loop (total)", tm_loop},
        {"  RK steps", tm_rk},
        {"    nonlinear terms (incl. x-FFTs, halos)", tm_nl},
        {"    time-step limit (CFL, MPI_Allreduce)", tm_dt},
        {"    right-hand side (viscous terms, RK)", tm_rhs},
        {"    pressure (z-FFT, all-to-alls, LU)", tm_p},
        {"    flow-rate correction (cfr)", tm_cfr},
        {"    rest of the RK steps", -1},
        {"x-FFTs (all calls)", tm_xfft},
        {"z-FFTs (all calls)", tm_zfft},
        {"all-to-all transposes (MPI, all calls)", tm_a2a},
        {"halo exchanges (MPI, all calls)", tm_halo},
        {"log output (diagnostics, log.dat)", tm_log},
        {"statistics (sample, write)", tm_stats},
        {"restart files", tm_io},
        {"rest of the time loop", -2},
    };
    const char* line = "==========================================================================================\n";
    const char* dash = "------------------------------------------------------------------------------------------\n";
    for (std::FILE* u : out) {
      if (!u) continue;
      std::fprintf(u, "%s FDCHAN timing report: %ld time steps, %d MPI rank(s), timing_sync = %d\n", line, nsteps, np,
                   int(sync_gpu));
      std::fprintf(u, " wall-clock times in s; min/mean/max over the ranks; %% of the mean time-loop time\n%s", dash);
      std::fprintf(u, "%-44s%8s%11s%11s%11s%8s%11s\n", " part of the code", "calls", "min", "mean", "max", "%", "ms/step");
      for (const Row& r : rows) {
        if (r.id >= 0) {
          const double m = tsum[r.id] / np;
          std::fprintf(u, " %-43s%8ld%11.3f%11.3f%11.3f%8.1f%11.3f\n", r.name, ncall[r.id], tmin[r.id], m, tmax[r.id],
                       100 * m / tloop, 1000 * m / ns);
        } else {
          const double s = r.id == -1
              ? (tsum[tm_rk] - tsum[tm_nl] - tsum[tm_dt] - tsum[tm_rhs] - tsum[tm_p] - tsum[tm_cfr]) / np
              : (tsum[tm_loop] - tsum[tm_rk] - tsum[tm_log] - tsum[tm_stats] - tsum[tm_io]) / np;
          std::fprintf(u, " %-43s%8s%11s%11.3f%11s%8.1f%11.3f\n", r.name, "", "", s, "", 100 * s / tloop, 1000 * s / ns);
        }
      }
      const double comm = (tsum[tm_a2a] + tsum[tm_halo]) / np;
      std::fprintf(u, " %-43s%8s%11s%11.3f%11s%8.1f%11.3f\n", "communication (all-to-alls + halos)", "", "", comm, "",
                   100 * comm / tloop, 1000 * comm / ns);
      std::fprintf(u, " (FFTs, all-to-alls and halos include the calls in the diagnostics and the initial projection)\n");
      if (!sync_gpu) std::fprintf(u, " (timing_sync = 0: GPU work may be attributed to a later part)\n");
    }
    // time per step: distribution and spikes
    const int W = 4 + NPART, nh = int(hist.size()) / W;
    if (nh > 0) {
      std::vector<double> ts(static_cast<size_t>(nh));
      for (int j = 0; j < nh; ++j) ts[size_t(j)] = hist[size_t(j * W + 1)];
      std::vector<double> srt = ts;
      std::sort(srt.begin(), srt.end());
      const double med = 0.5 * (srt[size_t((nh - 1) / 2)] + srt[size_t(nh / 2)]);
      double sum = 0;
      int nspike = 0;
      for (double x : ts) { sum += x; nspike += x > 2 * med; }
      std::vector<int> order(static_cast<size_t>(nh));
      for (int j = 0; j < nh; ++j) order[size_t(j)] = j;
      const int ntop = std::min(10, nh);
      std::partial_sort(order.begin(), order.begin() + ntop, order.end(),
                        [&](int a, int b) { return ts[size_t(a)] > ts[size_t(b)]; });
      for (std::FILE* u : out) {
        if (!u) continue;
        std::fprintf(u, "%s time per step (slowest rank), see timing_steps.dat:\n", dash);
        std::fprintf(u, "   min / median / mean / max [s] : %11.3e%11.3e%11.3e%11.3e\n", srt.front(), med, sum / nh, srt.back());
        std::fprintf(u, "   per step and grid point       : %11.3e s x #ranks\n", sum / nh / npoints * np);
        std::fprintf(u, "   steps slower than 2 x median  : %d (%.2f %%)\n", nspike, 100.0 * nspike / nh);
        std::fprintf(u, "   slowest steps [s]:     step     t_step  nonlin.   press.  alltoall    halo    output\n");
        for (int k = 0; k < ntop; ++k) {
          const double* h = &hist[size_t(order[size_t(k)] * W)];
          std::fprintf(u, "%17s%12ld%9.4f", "", long(h[0]), h[1]);
          for (int c = 0; c < NPART; ++c) std::fprintf(u, "%9.4f", h[4 + c]);
          std::fprintf(u, "\n");
        }
        std::fprintf(u, "%s", line);
      }
    }
    if (fr) std::fclose(fr);
    std::fflush(stdout);
  }
};

inline Timer g_timer;

}  // namespace fd
