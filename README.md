# FDCHAN: channel-flow DNS for GPUs, Fourier in x, compact differences in y, explicit differences in z

FDCHAN solves the incompressible Navier–Stokes equations in a plane channel (walls at y = ±1, periodic in x
and z) with

* **x:** Fourier series, pseudo-spectral nonlinear terms, 3/2-rule dealiasing (in x);
* **y:** compact finite differences (default; HiCHAN's 4th- and 6th-order schemes and an 8th-order variant) or
  explicit ones (order 2–8), computed directly on the stretched grid. With a z-slab decomposition every y-column
  is on one GPU, so the compact solves need no communication;
* **z:** explicit central differences of order 2–8 (`fd_order_z`, default = `fd_order`), or compact (tridiagonal)
  differences of order 4, 6 or 8 (`z_scheme = compact`). Compact z resolves more of the wavenumber range: for the
  same accuracy about 30 % (6th order) or 40 % (8th order) fewer z-points than explicit 6th order. Across the
  z-slabs the compact derivatives are solved as in x3d2 (DistD2-TDS): a local Thomas sweep per block, one exchange
  of a plane with each neighbour, a 2×2 system per block boundary. This needs at least 30 / 40 / 49 z-planes per
  rank (order 4 / 6 / 8, double precision; about half in single precision), checked at start-up;
* **pressure:** exact projection. Continuity (Fourier x, the same first derivatives in y and z) holds to
  round-off at every grid point, walls included. One z-FFT and, per (kx, ±kz) pair, one banded LU solve for φ
  and D1 φ together (factorised once at the start, since the matrix depends only on kx² + kz′²; every stage
  does only the substitutions); no influence matrix (see [DESIGN.md](DESIGN.md));
* **nonlinear terms:** rotational form u × ω (9 x-transforms per stage) or advective form (15);
* **time:** low-storage RK3 (γ = 8/15, 5/12, 3/4; ζ = 0, −17/60, −5/12), or the 4-stage RK3 of SIMSON
  (`rk_stages = 4`: γ = 8/17, 17/60, 5/12, 3/4; ζ = 0, −15/68, −17/60, −5/12), explicit viscous terms, adaptive Δt
  from the advective and viscous limits;
* **forcing:** constant flow rate (`cfr`, exact correction of the mean flow every stage) or constant pressure
  gradient (`cpg`);
* **scalars** (`nscal` ≤ 8, as HiCHAN): passive or active (Boussinesq buoyancy −Ri s ĝ) scalars with Dirichlet or
  Neumann walls, sources Q_c + Q_u u/U_b, advective, divergence or skew-symmetric form. Diffusion in x and y with
  the θ-method (HiCHAN's implicit y-solve with the influence matrix of the wall conditions), in z explicit.

The motivation and the comparison with HiCHAN (cost model, pressure-solver options, influence matrix,
velocity–vorticity formulation, accuracy) are in [DESIGN.md](DESIGN.md).

**Parallelisation:** MPI with z-slabs, one rank per GPU (or per CPU socket/node with threads). z-stencils use
halo exchanges with the two neighbouring ranks; the pressure is solved after an all-to-all transpose to kx-slabs
(two per RK stage). x-FFTs and the compact y-solves are local. Results are identical (to round-off) for any number
of ranks, and restart files can be read on a different number of ranks. With `NCCL=1` (GPU builds) the halo
exchanges and all-to-alls go through NCCL (RCCL on AMD), which uses NVLink between the GPUs of a node and runs on
the solver stream; MPI still starts the ranks and does the reductions and restart files.

**Status:** the CUDA and HIP builds have not been compiled yet (no GPU compiler was available where the code was
written; [Test.md](Test.md) describes the tests on Arrhenius); the CPU build runs exactly the same kernel code and is tested, with 1–4 MPI ranks. An implicit
wall-normal viscous term (velocity), 3D field output and a second split in y (for several hundred GPUs) are not written yet.

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
| `z_scheme` | explicit | `explicit` (`fd_order_z` 2, 4, 6, 8) or `compact` (`fd_order_z` 4, 6, 8); compact needs ≥ 30 / 40 / 49 z-planes per rank (double precision) |
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
| `iout_samples`, `tout_samples`, `tsamples_start`, `samples_dir` | 0, 0, 0, data_dir/samples | numbered restart files (time samples, e.g. for spectra with `rsspec` of the converter repository): `restart_t<time>.bin` every `iout_samples` steps and/or every `tout_samples` time units from `tsamples_start` on (0: off). Named by the time because the step count starts at 0 in every run; `tout_samples` gives evenly spaced samples across restarts (see Output) |
| `iout_log`, `iout_stats`, `iout_save_stats`, `iout_restart`, `tstats_start` | 100, 0, 0, 0, 0 | output intervals in steps |
| `mode_log_kx`, `mode_log_kz` | −1, 0 | Fourier mode (kx index ≥ 0, kz index of either sign) whose energy is logged as `E_mode` (as HiCHAN; kx < 0: none) |
| `timing`, `timing_sync` | false, true | timers as in HiCHAN: `timing_report.txt` and `timing_steps.dat` (see Output). `timing_sync` makes every timer wait for the GPU (exact attribution of the time to the parts) |
| `comm_backend` | nccl (NCCL=1 builds), else mpi | `nccl` or `mpi` for the halo exchanges and all-to-alls; `mpi` needs a CUDA-aware MPI |
| `fuse_y` | true | form the y-derivatives inside the kernels that use them (one thread per y-line, no derivative arrays written and read back; as x3d2's fused transport kernels). `false`: separate derivative and pointwise kernels. Same results to the last bit |
| `nthreads` | all cores | CPU build only (threads per rank) |

Scalars (HiCHAN's `&scalars` and `&buoyancy` items; lists are comma or space separated, one value per scalar):

| item | default | meaning |
|---|---|---|
| `nscal` | 0 | number of scalars (0–8) |
| `pr` | 1 | Prandtl/Schmidt number, diffusivity κ = ν/Pr |
| `scal_theta` | 0.5 | θ of the θ-method for the diffusion in x and y (0 explicit, 0.5 Crank–Nicolson, 1 implicit Euler); z is always explicit |
| `sbc_bot`, `sbc_top` | D, D | wall conditions: `D` (s = value) or `N` (∂s/∂y = value) |
| `sval_bot`, `sval_top` | 1, −1 | the wall values (uniform along the wall) |
| `ssrc_const`, `ssrc_u` | 0, 0 | source Q_c + Q_u u/`ubulk` |
| `sinit` | linear | `linear` (conduction profile if both walls are D, else zero), `zero`, `noise` (linear + random large-scale modes of rms `init_amp`) |
| `scal_form` | advective | `advective` u·∇s, `divergence` ∇·(us), `skew` (the mean of both) |
| `ri` | 0 | Richardson number per scalar: momentum source −Ri s ĝ (0: passive) |
| `gravity` | 0, −1, 0 | direction of gravity (normalised) |

## Tools

`python3 tools/plus_units.py data/stats.dat RE [data/log.dat t_start]`: u_τ, Re_τ, centreline U⁺ and
the peaks of the rms and shear-stress profiles in wall units.

## Output

`log.dat` and `stats.dat` have the same columns, headers and number formats as HiCHAN's, scalar columns included,
so the same tools read both.

* `log.dat`: `step time dt ubulk dpdx_applied tau_w Re_tau div_max Ekin E_mode`: τ_w the mean of both walls,
  Re_τ = √τ_w/ν, div_max = max |div u| over all points (walls included), Ekin = ½ ⟨u′·u′⟩ averaged over the
  channel, E_mode = ∫ (|û|² + |v̂|² + |ŵ|²) dy of the Fourier mode (`mode_log_kx`, `mode_log_kz`). Format
  `(i9, 6es14.6, es11.3, 2es14.6)`. Then per scalar n `dS<n>dy_b dS<n>dy_t`: dS/dy of the mean scalar at the
  bottom and top wall (`2es14.6`).
* `stats.dat`: time-weighted averages (weight Δt × `iout_stats`) of `U V W uu vv ww uv uw vw P pp` against y:
  the means, the moments ⟨f′g′⟩ (Parseval in x, averages over z), and the pressure of the last RK stage. As in
  HiCHAN the mean (kx = kz = 0) mode of the pressure is not computed, so P is 0 and pp is the variance of its
  fluctuations; with `nl_form = rotational` it is the total pressure p + |u|²/2. Then per scalar n
  `S<n> s<n>s<n> us<n> vs<n> ws<n>` (mean, variance, fluxes). Format `es16.7e3`, three header lines.
* `samples/restart_t<time>.bin` (with `iout_samples` or `tout_samples`): restart files kept as time samples, in the
  format of `restart.bin`, e.g. `restart_t000123.4560.bin` (the names sort in time). With `tout_samples` a sample is
  written at the first step at or after each multiple tsamples_start + k·tout_samples; a restarted run continues
  with the next multiple (no duplicate at its start). Point `samples_dir` of a chain of jobs to the same directory.
  Each file has the size of `restart.bin` (about 76 GB for 2560 × 385 × 1920 with one scalar in double
  precision): choose the interval with the disk quota in mind. Use them for restarts (`restart_file`) or for
  spectra (`rsspec`, converter repository).
* `restart.bin`: the spectral fields (kx, z, y) of u, v, w and the scalars, and the time; written at the end and
  every `iout_restart` steps. A file with fewer scalars (e.g. from a run without scalars) can be read: the missing
  scalars are initialised as given by `sinit`.
* With `timing = 1` (as HiCHAN's `timing = .true.`):
  * `timing_report.txt` (also printed at the end of the run): the wall-clock time spent in each part of the
    time loop: RK steps, nonlinear terms, time-step limit, right-hand side, pressure (split into its
    right-hand side, packing for the all-to-alls, all-to-alls, z-FFTs, banded LU solves and gradient
    correction), flow-rate correction, x-FFTs, halo exchanges, log, statistics and restart files, and the sum of
    the communication. For each part it gives the number of calls, the min/mean/max over the MPI ranks (load
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
| `examples/turb_scalars` | `turb_coarse` with two scalars: a weakly stably stratified one (Ri = 0.05, D/D) and a passive one with a Neumann wall and sources, skew form |
| `examples/retau180` | Re_τ = 180 in HiCHAN's box (4π × 4π/3), for a GPU |
| `examples/arrhenius_bench.sh` | time per step of FDCHAN and HiCHAN on one Arrhenius GH200 |

## Tests

| test | result |
|---|---|
| compact z (`make test`, `tests/test_zcompact.cpp`; with MPI on several ranks) | D1z, D2z of order 4, 6, 8 against the exact answer of the periodic scheme: round-off on 1–4 ranks, also uneven slabs and single precision; against the true derivative: orders 4.0, 6.0, 8.0 |
| compact z, full solver | turbulent channel 32 × 65 × 96 (100): divergence ≤ 8e-15, 1 and 2 ranks agree to all printed digits; explicit z unchanged (bit-identical on 1, 2, 3, 5 ranks) |
| operators (`make test`) | explicit D1, D2 converge at orders 2–8 on a tanh grid; compact schemes at the order of their wall rows in the max norm |
| pressure solver | banded LU with pivoting vs dense LU, ny = 33–385, k = 0.1–1e4, all schemes: 1e-12 or better, D1 φ to round-off; factorised once + substitutions: identical to factorising in every solve (to round-off where the compiler fuses multiply-adds differently) |
| scalars (`make test`, `tests/test_scalars.cpp`; with MPI on several ranks) | decay of single modes (Dirichlet/Neumann, mean and non-mean, θ = 0, 0.5, 1) against the exact decay: 6e-8 (θ ≤ 0.5), Neumann case 3rd order in y (the wall closures); steady quadratic profiles with Neumann walls and sources: 3e-11; hydrostatic balance (Ri = 2): velocity stays at 1e-17; advective, divergence and skew forms agree to the truncation error; the same on 2 ranks; single precision to round-off (≤ 2e-4) |
| scalars, full solver | turbulent channel 32 × 65 × 100 with two scalars, compact z: 1 and 2 ranks agree to all printed digits; restart written on 1 rank, read on 2: identical; restart without scalars read with `nscal = 2` |
| laminar Poiseuille, CFR | −dp/dx = 3/Re to all printed digits, divergence 3e-16 |
| decay of random disturbances, Re = 200 | E′ decays monotonically, −dp/dx → 3/Re |
| TS wave, Re = 5000, α = 1.12, tanh α_s = 4 | with HiCHAN's stencils and the advective form, HiCHAN's published errors are reproduced (2.65e-5 vs 2.6e-5 for 4th order, 3.4e-7 vs 3e-7 for 6th order, ny = 65); compact 8th order: 1.5e-7 (65), 2e-9 (97). DESIGN.md §5 |
| oblique wave (Squire), z-convergence | see DESIGN.md §5 |
| turbulent channel, Re_b = 2800, coarse 32 × 65 × 48 | sustained turbulence to t = 200 with explicit 6th, compact 6th and compact 8th order in y: Re_τ = 177–179, U_c⁺ = 18.3–18.8, divergence ≤ 3e-13 (DESIGN.md §6) |
