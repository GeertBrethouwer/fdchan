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
* **time:** low-storage RK3 (γ = 8/15, 5/12, 3/4; ζ = 0, −17/60, −5/12), explicit viscous terms, adaptive Δt
  from the advective and viscous limits;
* **forcing:** constant flow rate (`cfr`, exact correction of the mean flow every stage) or constant pressure
  gradient (`cpg`).

The motivation and the comparison with HiCHAN (cost model, pressure-solver options, influence matrix,
velocity–vorticity formulation, accuracy) are in [DESIGN.md](DESIGN.md).

**Status:** single GPU / single CPU node. The CUDA and HIP builds have not been compiled yet (no GPU compiler
was available where the code was written); the CPU build runs exactly the same kernel code and is tested.
Multi-GPU (z-slabs: halo exchanges plus two all-to-alls per stage), scalars, an implicit wall-normal viscous term
and 3D field output are not written yet.

## Building

```bash
make                       # CPU (std::thread pool), build/fdchan
make GPU=cuda ARCH=sm_90   # NVIDIA GH200/H100: nvcc + cuFFT
make GPU=hip ARCH=gfx90a   # AMD MI250X: hipcc + hipFFT
make SINGLE=1 ...          # single precision
make test                  # operator and pressure-solver tests
```

On NSC Arrhenius, `module load GPU/buildenv-nvhpc/25.9-cu13.0` provides nvcc and cuFFT.
`-DFD_DEBUG_SYNC` (add to `CXXFLAGS`) checks for errors after every kernel; use it for the first GPU run.

## Running

```bash
build/fdchan input.txt
```

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
| `init_type` | laminar | `laminar`, `noise` (random large-scale modes, amplitude `init_amp`, `init_seed`), `mode` (a wave with kx index `mode_kx`, kz index `mode_kz`), `restart` |
| `restart_file`, `reset_time` | restart.bin, false | for `init_type = restart` |
| `data_dir` | data | output directory |
| `iout_log`, `iout_stats`, `iout_save_stats`, `iout_restart`, `tstats_start` | 100, 0, 0, 0, 0 | output intervals in steps |
| `mode_log_kx` | 1 | kx index of the mode whose v-energy is logged |
| `timing` | false | per-part timers (synchronises the GPU) |
| `nthreads` | all cores | CPU build only |

## Tools

`python3 tools/plus_units.py data/stats.dat RE [data/log.dat t_start]`: u_τ, Re_τ, centreline U⁺ and
the peaks of the rms and shear-stress profiles in wall units.

## Output

* `log.dat`: step, t, Δt, bulk velocity, −dp/dx, τ_w (bottom, top), Re_τ, max |div u| (all points, walls
  included), fluctuation energy, v-energy of mode `mode_log_kx`.
* `stats.dat`: time averages of U, V, W, ⟨u′u′⟩, ⟨v′v′⟩, ⟨w′w′⟩, ⟨u′v′⟩ (Parseval in x, averages over z).
* `restart.bin`: the spectral fields (kx, z, y) and the time; written at the end and every `iout_restart` steps.

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
