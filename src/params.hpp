// Input: a plain text file with lines `key = value`; '!' or '#' start a comment.
#pragma once
#include <fstream>
#include <map>
#include <sstream>
#include <string>

#include "backend.hpp"

namespace fd {

struct Params {
  // grid
  int nx = 64, ny = 65, nz = 64;
  double lx = 4 * M_PI, lz = 2 * M_PI;
  std::string grid_type = "tanh";
  double grid_alpha = 3.0;
  int fd_order = 6;
  std::string y_scheme = "compact";  // compact (4, 6, 8) | explicit (2, 4, 6, 8)
  std::string nl_form = "rotational";  // FDCHAN: rotational (9 x-FFTs per stage) | advective (15);
                                       // SCCHAN: divergence (default for it, 13) | skew (33)
  std::string y_closure = "hichan";  // compact wall closures: hichan | explicit_wall
  int fd_order_z = 0;  // 0: same as fd_order
  std::string z_scheme = "explicit";  // explicit (fd_order_z 2, 4, 6, 8) | compact (fd_order_z 4, 6, 8)
  bool dealias = true;
  // physics
  double re = 2800;               // U h / nu
  std::string forcing = "cfr";    // cfr | cpg
  double ubulk = 1.0, dpdx = 0.0;  // dpdx: -dp/dx for cpg
  // time
  long nstep = 1000;
  double tmax = 1e30, wall_time_max = 1e30;
  bool fixed_dt = false;
  double dt = 1e-4, dt_max = 0.05, dt_growth = 1.1;
  double cfl = 1.2, cfl_visc = 2.0;
  int rk_stages = 3;  // low-storage RK3: 3 stages (Wray) or 4 stages (SIMSON user guide, Table 5.1)
  // initial condition
  std::string init_type = "laminar";  // laminar | noise | mode | restart
  double init_amp = 0.1;
  int init_seed = 1;
  int mode_kx = 1, mode_kz = 0;
  std::string mode_file;  // optional v(y) profile for init_type = mode: lines "y Re(v) Im(v)"
  std::string restart_file = "restart.bin";
  bool reset_time = false;
  // output
  std::string data_dir = "data";
  int iout_log = 100, iout_stats = 0, iout_save_stats = 0, iout_restart = 0;
  double tstats_start = 0;
  int mode_log_kx = -1, mode_log_kz = 0;  // energy of one Fourier mode in log.dat (as HiCHAN; kx < 0: none)
  bool timing = false;       // timers: report at the end (timing_report.txt), time per step in timing_steps.dat
  bool timing_sync = true;   // timers wait for the GPU (exact attribution, a little overhead)
  std::string comm_backend;  // nccl | mpi: halo exchanges and all-to-alls; default nccl if built with NCCL=1
  bool fuse_y = true;  // FDCHAN: y-derivatives formed inside the kernels that use them (one thread per line)
  // CPU only
  int nthreads = 0;

  void read(const std::string& file) {
    std::ifstream in(file);
    if (!in) fatal(("cannot open input file " + file).c_str());
    std::map<std::string, std::string> kv;
    std::string line;
    while (std::getline(in, line)) {
      for (char c : {'!', '#'}) {
        auto p = line.find(c);
        if (p != std::string::npos) line = line.substr(0, p);
      }
      auto e = line.find('=');
      if (e == std::string::npos) continue;
      auto trim = [](std::string s) {
        const char* ws = " \t\r\n'\"";
        s.erase(0, s.find_first_not_of(ws));
        s.erase(s.find_last_not_of(ws) + 1);
        return s;
      };
      kv[trim(line.substr(0, e))] = trim(line.substr(e + 1));
    }
    auto I = [&](const char* k, int& v) { if (kv.count(k)) { v = std::stoi(kv[k]); kv.erase(k); } };
    auto L = [&](const char* k, long& v) { if (kv.count(k)) { v = std::stol(kv[k]); kv.erase(k); } };
    auto D = [&](const char* k, double& v) { if (kv.count(k)) { v = std::stod(kv[k]); kv.erase(k); } };
    auto S = [&](const char* k, std::string& v) { if (kv.count(k)) { v = kv[k]; kv.erase(k); } };
    auto B = [&](const char* k, bool& v) {
      if (kv.count(k)) {
        const std::string s = kv[k];
        v = (s == "1" || s == "true" || s == ".true." || s == "yes");
        kv.erase(k);
      }
    };
    I("nx", nx); I("ny", ny); I("nz", nz); D("lx", lx); D("lz", lz); S("grid_type", grid_type);
    D("grid_alpha", grid_alpha); I("fd_order", fd_order); S("y_scheme", y_scheme); S("y_closure", y_closure); S("nl_form", nl_form); I("fd_order_z", fd_order_z); S("z_scheme", z_scheme); B("dealias", dealias);
    D("re", re); S("forcing", forcing); D("ubulk", ubulk); D("dpdx", dpdx);
    L("nstep", nstep); D("tmax", tmax); D("wall_time_max", wall_time_max); B("fixed_dt", fixed_dt);
    D("dt", dt); D("dt_max", dt_max); D("dt_growth", dt_growth); D("cfl", cfl); D("cfl_visc", cfl_visc); I("rk_stages", rk_stages);
    S("init_type", init_type); D("init_amp", init_amp); I("init_seed", init_seed); I("mode_kx", mode_kx);
    I("mode_kz", mode_kz); S("restart_file", restart_file); S("mode_file", mode_file); B("reset_time", reset_time);
    S("data_dir", data_dir); I("iout_log", iout_log); I("iout_stats", iout_stats);
    I("iout_save_stats", iout_save_stats); I("iout_restart", iout_restart); D("tstats_start", tstats_start);
    I("mode_log_kx", mode_log_kx); I("mode_log_kz", mode_log_kz); B("timing", timing); B("timing_sync", timing_sync); B("fuse_y", fuse_y); S("comm_backend", comm_backend); I("nthreads", nthreads);
    for (auto& p : kv) std::fprintf(stderr, "warning: unknown input item '%s'\n", p.first.c_str());
    if (nx % 2) fatal("nx must be even");
    if (fd_order % 2 || fd_order < 2 || fd_order > 8) fatal("fd_order must be 2, 4, 6 or 8");
    if (y_scheme != "compact" && y_scheme != "explicit") fatal("y_scheme must be compact or explicit");
    if (y_scheme == "compact" && fd_order != 4 && fd_order != 6 && fd_order != 8) fatal("compact y_scheme: fd_order must be 4, 6 or 8");
    if (nl_form != "rotational" && nl_form != "advective" && nl_form != "divergence" && nl_form != "skew")
      fatal("nl_form must be rotational, advective, divergence or skew");
    if (fd_order_z == 0) fd_order_z = fd_order;
    if (comm_backend != "" && comm_backend != "nccl" && comm_backend != "mpi") fatal("comm_backend must be nccl or mpi");
    if (fd_order_z % 2 || fd_order_z < 2 || fd_order_z > 8) fatal("fd_order_z must be 2, 4, 6 or 8");
    if (z_scheme != "explicit" && z_scheme != "compact") fatal("z_scheme must be explicit or compact");
    if (z_scheme == "compact" && fd_order_z < 4) fatal("compact z_scheme: fd_order_z must be 4, 6 or 8");
  }
  void print() const {
    std::printf("grid      nx = %d, ny = %d, nz = %d, lx = %.6g, lz = %.6g, %s (alpha = %g), order %d (y) %d (z), dealias %d\n",
                nx, ny, nz, lx, lz, grid_type.c_str(), grid_alpha, fd_order, fd_order_z, int(dealias));
    std::printf("physics   re = %g, forcing = %s, ubulk = %g, dpdx = %g\n", re, forcing.c_str(), ubulk, dpdx);
    std::printf("time      nstep = %ld, tmax = %g, fixed_dt = %d, dt = %g, cfl = %g, cfl_visc = %g, rk_stages = %d\n",
                nstep, tmax, int(fixed_dt), dt, cfl, cfl_visc, rk_stages);
    std::printf("init      %s, amp = %g\n", init_type.c_str(), init_amp);
    if (timing) std::printf("timing    on%s (timing_report.txt, timing_steps.dat)\n", timing_sync ? ", GPU-synchronised" : "");
  }
};

}  // namespace fd
