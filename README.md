# FDCHAN: channel-flow DNS for GPUs, Fourier in x, compact differences in y, explicit differences in z

FDCHAN solves the incompressible Navier–Stokes equations in a plane channel (walls at y = ±1, periodic in x
and z) with

* **x:** Fourier series, pseudo-spectral nonlinear terms, 3/2-rule dealiasing (in x);
* **y:** compact finite differences (default; HiCHAN's 4th- and 6th-order schemes and an 8th-order variant) or
  explicit ones (order 2–8), computed directly on the stretched grid. With a z-slab decomposition every y-column
  is on one GPU, so the compact solves need no communication;
* **z:** explicit central differences of order 2–8 (`fd_order_z`, default = `fd_order`);
* **pressure:** exact projection. Continuity (Fourier x, the same first derivatives in y and z) holds to
  round-off at every grid point, walls included. One z-FFT and, per (kx, ±kz) pair, one banded LU solve for φ
  and D1 φ together; no influence matrix (see [DESIGN.md](DESIGN.md));
* **nonlinear terms:** rotational form u × ω (9 x-transforms per stage) or advective form (15);
* **time:** low-storage RK3 (γ = 8/15, 5/12, 3/4; ζ = 0, −17/60, −5/12), or the 4-stage RK3 of SIMSON
  (`rk_stages = 4`: γ = 8/17, 17/60, 5/12, 3/4; ζ = 0, −15/68, −17/60, −5/12), explicit viscous terms, adaptive Δt
  from the advective and viscous limits;
* **forcing:** constant flow rate (`cfr`, exact correction of the mean flow every stage) or constant pressure
  gradient (`cpg`).

The motivation and the comparison with HiCHAN (cost model, pressure-solver options, influence matrix,
velocity–vorticity formulation, accuracy) are in [DESIGN.md](DESIGN.md).

**Parallelisation:** MPI with z-slabs, one rank per GPU (or per CPU socket/node with threads). z-stencils use
halo exchanges with the two neighbouring ranks; the pressure is solved after an all-to-all transpose to kx-slabs
(two per RK stage). x-FFTs and the compact y-solves are local. Results are identical (to round-off) for any number
of ranks, and restart files can be read on a different number of ranks. With `NCCL=1` (GPU builds) the halo
exchanges and all-to-alls go through NCCL (RCCL on AMD), which uses NVLink between the GPUs of a node and runs on
the solver stream; MPI still starts the ranks and does the reductions and restart files.

**Status:** the CUDA and HIP builds have not been compiled yet (no GPU compiler was available where the code was
written; [Test.md](Test.md) describes the tests on Arrhenius); the CPU build runs exactly the same kernel code and is tested, with 1–4 MPI ranks. Scalars, an implicit
wall-normal viscous term, 3D field output and a second split in y (for several hundred GPUs) are not written yet.

## Building

```bash
make                       # CPU (std::thread pool), build/fdchan
make GPU=cuda ARCH=sm_90   # NVIDIA GH200/H100: nvcc + cuFFT
make GPU=hip ARCH=gfx90a   # AMD MI250X: hipcc + hipFFT
make SINGLE=1 ...          # single precision
make MPI=1 ...             # several ranks: build/fdchan_mpi (MPICXX, default ~/opt/openmpi/bin/mpicxx)
make GPU=cuda MPI=1 NCCL=1 # halo exchanges and all-to-alls with NCCL (recommended for several GPUs)
make test                  # operator and pressure-solver tests
```

On NSC Arrhenius, `module load GPU/buildenv-nvhpc/25.9-cu13.0` provides nvcc, cuFFT and a CUDA-aware MPI:
`make GPU=cuda MPI=1 NCCL=1 MPICXX=mpicxx`. NCCL is taken from the HPC SDK next to nvcc (`tools/find_nccl.sh`; set
`NCCL_HOME` to override). On one node Open MPI's device-pointer path through shared memory reached only 1–2 GB/s
per GPU there, about 50× slower than NVLink; NCCL avoids it. The MPI compile and link flags are taken from `mpicxx --showme:compile/link`
(Open MPI / HPC-X); for another MPI set `MPI_CFLAGS` and `MPI_LFLAGS`. Device pointers are passed to MPI, so the MPI
must be CUDA-aware (HIP: GPU-aware). On macOS with Open MPI and gfortran-built codes, see `../compare/COMPARISON.md`
for the `MPI_IN_PLACE` link flag (FDCHAN itself is not affected).
`-DFD_DEBUG_SYNC` (add to `CXXFLAGS`) checks for errors after every kernel; use it for the first GPU run.

## Running

```bash
build/fdchan input.txt                         # one GPU / one CPU node
mpirun -n 8 build/fdchan_mpi input.txt         # 8 ranks (GPUs), z-slabs
```

With MPI each rank picks GPU (node-local rank) mod (GPUs per node). The z-slabs need at least p/2 = 3 planes per
rank (6th order); for efficiency use at least ~16 (nz/P ≥ 16). `nx/2` x-modes are distributed over the ranks for
the pressure solve. With `timing = 1` the time in each part of the code, communication included, is reported
(see Output).

The input is a text file with `key = value` lines (`!` or `#` start a comment); unknown keys are reported.

| key | default | meaning |
|---|---|---|
| `nx, ny, nz` | 64, 65, 64 | Fourier modes in x (even), points in y including the walls, points in z |
| `lx, lz` | 4π, 2π | box size (channel half height 1) |
| `grid_type`, `grid_alpha` | tanh, 3.0 | `tanh` (as HiCHAN: y = tanh(α(ξ−½))/tanh(α/2)), `uniform`, `cosine` |
| `y_scheme` | compact | `compact` (`fd_order` 4, 6 or 8) or `explicit` (`fd_order` 2, 4, 6 or 8) |
| `y_closure` | hichan | compact wall closures for orders 4 and 6: `hichan` (exactly HiCHAN's (3-4-3) and (3-5-6-5-3) schemes) or `explicit_wall`; order 8 always uses explicit wall rows |
| `fd_order`, `fd_order_z` | 6, = fd_order | order of the y and z differences |
| `nl_form` | rotational | `rotational` or `advective` |
| `dealias` | true | 3/2 rule in x |
| `re` | 2800 | U h/ν |
| `forcing`, `ubulk`, `dpdx` | cfr, 1, 0 | `cfr` keeps the bulk velocity at `ubulk`; `cpg` applies −dp/dx = `dpdx` |
| `nstep`, `tmax`, `wall_time_max` | 1000, ∞, ∞ | stop criteria (steps of this run, simulation time, seconds) |
| `fixed_dt`, `dt`, `dt_max`, `dt_growth` | false, 1e-4, 0.05, 1.1 | fixed or adaptive time step |
| `cfl`, `cfl_visc` | 1.2, 2.0 | RK3 limits (< √3 advective, < 2.51 diffusive) |
| `rk_stages` | 3 | 3: Wray's low-storage RK3; 4: the 4-stage third-order RK3 of the SIMSON user guide (Table 5.1): limits √8 = 2.83 (`cfl`) and 2.79 (`cfl_visc`) at 4/3 of the cost per step, and a much smaller error at the same Δt (turbulent case: 35–500×) |
| `init_type` | laminar | `laminar`, `noise` (random large-scale modes, amplitude `init_amp`, `init_seed`), `mode` (a wave with kx index `mode_kx`, kz index `mode_kz`), `restart` |
| `restart_file`, `reset_time` | restart.bin, false | for `init_type = restart` |
| `data_dir` | data | output directory |
| `iout_log`, `iout_stats`, `iout_save_stats`, `iout_restart`, `tstats_start` | 100, 0, 0, 0, 0 | output intervals in steps |
| `mode_log_kx` | 1 | kx index of the mode whose v-energy is logged |
| `timing`, `timing_sync` | false, true | timers as in HiCHAN: `timing_report.txt` and `timing_steps.dat` (see Output). `timing_sync` makes every timer wait for the GPU (exact attribution of the time to the parts) |
| `comm_backend` | nccl (NCCL=1 builds), else mpi | `nccl` or `mpi` for the halo exchanges and all-to-alls; `mpi` needs a CUDA-aware MPI |
| `fuse_y` | true | form the y-derivatives inside the kernels that use them (one thread per y-line, no derivative arrays written and read back; as x3d2's fused transport kernels). `false`: separate derivative and pointwise kernels. Same results to the last bit |
| `nthreads` | all cores | CPU build only (threads per rank) |

## Tools

`python3 tools/plus_units.py data/stats.dat RE [data/log.dat t_start]`: u_τ, Re_τ, centreline U⁺ and
the peaks of the rms and shear-stress profiles in wall units.

## Output

* `log.dat`: step, t, Δt, bulk velocity, −dp/dx, τ_w (bottom, top), Re_τ, max |div u| (all points, walls
  included), fluctuation energy, v-energy of mode `mode_log_kx`.
* `stats.dat`: time averages of U, V, W, ⟨u′u′⟩, ⟨v′v′⟩, ⟨w′w′⟩, ⟨u′v′⟩ (Parseval in x, averages over z).
* `restart.bin`: the spectral fields (kx, z, y) and the time; written at the end and every `iout_restart` steps.
* With `timing = 1` (as HiCHAN's `timing = .true.`):
  * `timing_report.txt` (also printed at the end of the run): the wall-clock time spent in each part of the
    code: RK steps, nonlinear terms, time-step limit, right-hand side, pressure, flow-rate correction, x- and
    z-FFTs, all-to-all transposes, halo exchanges, log, statistics and restart files, and their sum of
    communication. For each part it gives the number of calls, the min/mean/max over the MPI ranks (load
    imbalance), the percentage of the time loop and the time per step. It also gives the distribution of the
    time per step (min/median/mean/max), the number of steps slower than 2× the median, and the 10 slowest steps
    with their breakdown.
  * `timing_steps.dat`: one line per time step. Columns: step, t, Δt, the wall-clock time of the step (slowest
    rank, fastest rank, mean), and the time of the step spent in the nonlinear terms, pressure, all-to-alls, halo
    exchanges and output (max over the ranks).
  * `timing_sync = 1` (default) makes every timer wait for the GPU, so that asynchronous GPU work is attributed
    to the right part. This removes some overlap of kernel launches. With `timing_sync = 0` the parts are less
    exact, but the time per step stays exact. The results do not depend on the timing settings (checked bit for
    bit on the CPU, 1 and 4 ranks).

## Examples

| directory | case |
|---|---|
| `examples/laminar` | laminar Poiseuille flow, CFR: −dp/dx = 3/Re, divergence at round-off (seconds) |
| `examples/ts_wave` | Tollmien–Schlichting wave as in HiCHAN's `examples/ts_wave`; `python3 fit.py data/log.dat 250` gives the decay rate (minutes on a CPU) |
| `examples/turb_coarse` | coarse turbulent channel from noise, Re_b = 2800 (about 30 minutes on a laptop CPU) |
| `examples/retau180` | Re_τ = 180 in HiCHAN's box (4π × 4π/3), for a GPU |
| `examples/arrhenius_bench.sh` | time per step of FDCHAN and HiCHAN on one Arrhenius GH200 |

## Tests

| test | result |
|---|---|
| operators (`make test`) | explicit D1, D2 converge at orders 2–8 on a tanh grid; compact schemes at the order of their wall rows in the max norm |
| pressure solver | banded LU with pivoting vs dense LU, ny = 33–385, k = 0.1–1e4, all schemes: 1e-12 or better, D1 φ to round-off |
| laminar Poiseuille, CFR | −dp/dx = 3/Re to all printed digits, divergence 3e-16 |
| decay of random disturbances, Re = 200 | E′ decays monotonically, −dp/dx → 3/Re |
| TS wave, Re = 5000, α = 1.12, tanh α_s = 4 | with HiCHAN's stencils and the advective form, HiCHAN's published errors are reproduced (2.65e-5 vs 2.6e-5 for 4th order, 3.4e-7 vs 3e-7 for 6th order, ny = 65); compact 8th order: 1.5e-7 (65), 2e-9 (97). DESIGN.md §5 |
| oblique wave (Squire), z-convergence | see DESIGN.md §5 |
| turbulent channel, Re_b = 2800, coarse 32 × 65 × 48 | sustained turbulence to t = 200 with explicit 6th, compact 6th and compact 8th order in y: Re_τ = 177–179, U_c⁺ = 18.3–18.8, divergence ≤ 3e-13 (DESIGN.md §6) |
