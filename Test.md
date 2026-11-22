# Testing FDCHAN on Arrhenius

FDCHAN has never been compiled with nvcc or run on a GPU. The CUDA code is the same kernel code that runs in the
tested CPU build, but expect a first round of compiler errors. Everything below marked *expected* comes from CPU
runs on a laptop or from estimates, not from measurements on Arrhenius.

Module, account and partition names follow HiCHAN's `examples/arrhenius_job.sh`; adjust them if yours differ.

## 1. Get and build the code

```bash
cd $HOME
git clone https://github.com/GeertBrethouwer/fdchan.git
cd fdchan
module load GPU/buildenv-nvhpc/25.9-cu13.0
```

Build four executables. `EXE=` gives each its own name; without it every build writes `build/fdchan`.

```bash
make GPU=cuda ARCH=sm_90 EXE=build/fdchan_gpu                                        # 1 GPU
make GPU=cuda ARCH=sm_90 MPI=1 MPICXX=mpicxx EXE=build/fdchan_gpu_mpi                # several GPUs
make GPU=cuda ARCH=sm_90 DEFS=-DFD_DEBUG_SYNC EXE=build/fdchan_gpu_dbg               # error check after every kernel
make CXX=g++ EXE=build/fdchan_cpu                                                    # CPU reference (Grace)
```

| setting | value |
|---|---|
| compiler | `nvcc` from the NVIDIA HPC SDK (`-x cu --extended-lambda --expt-relaxed-constexpr -arch=sm_90`) |
| libraries | `-lcufft`; with `MPI=1` also the flags of `mpicxx --showme:compile` / `--showme:link` |
| MPI | must be CUDA-aware: device pointers are passed to `MPI_Sendrecv` and `MPI_Alltoallv` |

If `mpicxx --showme:compile` does not work (not an Open MPI / HPC-X wrapper), set the flags by hand:
`make ... MPI=1 MPI_CFLAGS="-I<mpi>/include" MPI_LFLAGS="-L<mpi>/lib -lmpi"`.

`make test` (CPU, `make CXX=g++ test`) runs the operator and pressure-solver tests in a few seconds; every line of
the pressure test should show differences of 1e-12 or smaller.

Things to look for in the nvcc output, since these constructs have never been through nvcc:

- the nested lambdas passed to `band_solve_piv` inside `solve_impl` (extended `__host__ __device__` lambdas);
- `fabs`, `sqrt` and the `cplx` operators in device code;
- the templates `solve_impl<K>` for K = 4, …, 16 (only K = 4 is used with the default compact y scheme).

Send me the compiler errors as they are; most will be quick fixes.

## 2. Validation on one GPU

Run the tests with `fdchan_gpu_dbg` first. It stops at the first failing kernel with a message `GPU error in ...`.
When they pass, repeat them with `fdchan_gpu`.

### 2.1 Laminar Poiseuille flow (seconds)

```bash
mkdir -p $HOME/fdchan_tests/laminar && cd $HOME/fdchan_tests/laminar
cp $HOME/fdchan/examples/laminar/input.txt .
srun -n 1 --gpus=1 $HOME/fdchan/build/fdchan_gpu_dbg input.txt
```

Expected (the same on the CPU):

```
step      100  t = 5.00000  dt = 5.000e-02  Ub = 1.000000  -dp/dx = 3.000000e-03  Re_tau = 54.77  div = 2.6e-16 ...
step      200  t = 10.00000  dt = 5.000e-02  Ub = 1.000000  -dp/dx = 3.000000e-03  Re_tau = 54.77  div = 2.7e-16 ...
```

−dp/dx must be exactly 3/Re and `div` must stay at round-off (1e-16 to 1e-15).

### 2.2 Tollmien–Schlichting wave (about 10–20 minutes on one GPU)

```bash
mkdir -p $HOME/fdchan_tests/ts && cd $HOME/fdchan_tests/ts
cp $HOME/fdchan/examples/ts_wave/{input.txt,fit.py} .
srun -n 1 --gpus=1 $HOME/fdchan/build/fdchan_gpu input.txt
python3 fit.py data/log.dat 250
```

Expected:

```
omega_i = -2.78324129e-03   abs. error = 3.40e-07   rel. error = 1.22e-04
```

The same input with HiCHAN (ny = 65, `examples/ts_wave`) gives −2.78324103e-3: both codes use the same compact
stencils and the advective form. The GPU result should agree with the CPU value to at least 8 digits. This case
is tiny (4 × 65 × 1), so the run time is set by kernel launches (175 000 steps), not by the GPU's speed; it says
nothing about performance. On the laptop CPU it took 375 s.

### 2.3 Turbulent channel from noise: GPU against CPU (minutes)

The random initial field depends on the C++ library, so compare the GPU build with the CPU build on Arrhenius,
not with numbers from elsewhere.

```bash
mkdir -p $HOME/fdchan_tests/turb_{gpu,cpu} && cd $HOME/fdchan_tests
for d in turb_gpu turb_cpu; do
  sed -e '/^iout_/d' -e '/^tstats/d' -e '/^tmax/d' -e '/^nstep/d' -e '/^timing/d' \
      $HOME/fdchan/examples/turb_coarse/input.txt > $d/input.txt
  printf "nstep = 30\niout_log = 10\ntiming = 1\n" >> $d/input.txt
done
(cd turb_gpu && srun -n 1 --gpus=1 $HOME/fdchan/build/fdchan_gpu input.txt)
(cd turb_cpu && srun -n 1 -c 72 $HOME/fdchan/build/fdchan_cpu input.txt)
diff turb_gpu/data/log.dat turb_cpu/data/log.dat
```

Expected: the two `log.dat` files agree to about 10 or more of the printed digits after 30 steps; only the `max|div|`
column (round-off itself, about 1e-15) differs completely. cuFFT and the CPU FFT round differently, so bitwise
agreement is not expected, and differences grow slowly in time (the flow is chaotic).

## 3. Several GPUs

### 3.1 Same case on 1, 2 and 4 GPUs (one node)

Use the turbulent input of 2.3 (nz = 48, so 4 GPUs get 12 z-planes each) in directories `r1`, `r2`, `r4`:

```bash
srun --mpi=pmix -n 1 --gpus-per-node=1 $HOME/fdchan/build/fdchan_gpu_mpi input.txt   # in r1
srun --mpi=pmix -n 2 --gpus-per-node=2 $HOME/fdchan/build/fdchan_gpu_mpi input.txt   # in r2
srun --mpi=pmix -n 4 --gpus-per-node=4 $HOME/fdchan/build/fdchan_gpu_mpi input.txt   # in r4
```

Expected:

- stdout: `FDCHAN (GPU build, N rank(s), double precision)` and
  `parallel  N rank(s), z-slabs of ... planes, ... x-modes per rank in the pressure solve`;
- the `log.dat` files of r1, r2 and r4 agree to all printed digits except `max|div|` (on the CPU, 1–4 ranks agreed
  to all 11 digits; only the order of the global sums differs, at the 1e-15 level);
- r1 agrees with `fdchan_gpu` of 2.3 to all printed digits (`fdchan_gpu` and `fdchan_gpu_mpi` on one rank do the same
  arithmetic).

On one node, Open MPI's libfabric (OFI) components abort in `MPI_Init` on Arrhenius (`free(): invalid pointer` in
`libfabric.so`, called from `opal_common_ofi_fi_domain`). Disable them before `srun`, as
`examples/arrhenius_mpi.sh` does automatically when the job has one node:

```bash
export OMPI_MCA_btl=^ofi OMPI_MCA_mtl=^ofi
```

If the run crashes in `MPI_Sendrecv` or `MPI_Alltoallv` (segmentation fault, `cuda` errors from UCX), the MPI is not
handling device pointers. Tell me; staging through host memory is a small change.

### 3.2 Restart on a different number of GPUs

In a new directory: run the input of 2.3 with `nstep = 20` on 2 GPUs. Then set
`init_type = restart`, `restart_file = data/restart.bin`, `nstep = 10` and run on 4 GPUs.

Expected: the last line of `data/log.dat` (step 10 of the second run, t identical) agrees with step 30 of `r4`
(or r1) to all printed digits. On the CPU this was exact.

### 3.3 Two nodes

Repeat 3.1 with 8 GPUs on two nodes (`examples/arrhenius_mpi.sh`, `srun --mpi=pmix`). With more than one node the
script keeps libfabric and sets `FI_PROVIDER=cxi` (Slingshot) and `FI_LOG_LEVEL=warn`; this has not been tested yet. nz = 48 gives 6 planes per
GPU, still above the minimum of 3. The same agreement is expected.

## 4. Performance

All runs with `timing = 1` in the input: at the end FDCHAN prints the same kind of timing report as HiCHAN
(also in `data/timing_report.txt`): the time in each part (nonlinear terms, right-hand side, pressure, x- and
z-FFTs, all-to-alls, halo exchanges, output) with min/mean/max over the ranks, and the distribution of the time
per step; `data/timing_steps.dat` has one line per step. Use 200 steps and `iout_log = 50`, `iout_stats = 0`,
`iout_restart = 0`.

`timing_sync = 1` (default) synchronises the GPU around every part; for the final number of a configuration run
once more with `timing = 0`, or with `timing_sync = 0` (the time per step is then exact, the parts less so).

Run B and F1 a second time with `fuse_y = 0`. The default `fuse_y = 1` forms the y-derivatives inside the kernels
that use them; it reads and writes less memory but uses one GPU thread per y-line instead of one per point. Both
give the same `log.dat` to the last digit; keep whichever is faster (expected: `fuse_y = 1` for the large grids).

### 4.1 One GPU, Re_τ = 180

| run | code | grid | input |
|---|---|---|---|
| A | HiCHAN | 128 × 129 × 128 modes | `examples/retau180`, `nscal = 0`, `timing = .true.` |
| B | FDCHAN | 128 × 129 × 128 | `examples/retau180/input.txt` with `nz = 128` |
| C | FDCHAN | 128 × 129 × 192 | `examples/retau180/input.txt` as it is (1.5× points in z for the same accuracy) |

`examples/arrhenius_bench.sh` runs B, C and A in one job (edit the paths at the top).

Read: time per step, and the `dt` column of `log.dat` after the first 100 steps. Both codes use RK3 with the same
CFL rule structure, but compare the time per unit of simulated time, (time per step) / dt, not only the time per step.

### 4.2 Production-size case on 16 GPUs

Your 2560 × 385 × 1920 case, without the scalar (FDCHAN has no scalars yet), 4 nodes × 4 GPUs:

| run | code | grid | notes |
|---|---|---|---|
| P0 | HiCHAN | 2560 × 385 × 1920 modes, `nscal = 0` | your best `&parallel` settings |
| F1 | FDCHAN | 2560 × 385 × 1920 | `mpirun -n 16`, 120 z-planes per GPU |
| F2 | FDCHAN | 2560 × 385 × 2880 | equal z-accuracy (1.5×), 180 planes per GPU |

Memory: FDCHAN prints its array memory on rank 0 at start-up; about 240 bytes per grid point in total, so about
28 GB per GPU for F1 and 43 GB for F2 (96 GB available).

Then strong scaling of F1 on 8, 16 and 32 GPUs (240, 120, 60 planes per GPU).

## 5. Expected performance (estimates, not measurements)

### 5.1 One GPU

From the memory-traffic model in DESIGN.md: at the same grid one FDCHAN stage moves about 0.6× the data of a
HiCHAN stage. If both codes reach a similar fraction of the memory bandwidth, B should be 1.3–1.7× faster per step
than A, and C about as fast as A. This is uncertain in both directions:

- against FDCHAN: its kernels are only partly tuned (the banded LU keeps its rows in local memory; the y-derivatives
  are fused into the kernels that use them and the all-to-all packing uses tiled transposes, but neither has run on a
  GPU yet);
- for FDCHAN: one x-FFT per transform instead of a 2D FFT plus two transposes, on a grid without z-dealiasing.

For this small grid (2–3 million points) kernel-launch overhead may dominate both codes.

### 5.2 Several nodes

Communication per RK stage and GPU for F1: halo exchanges of about 0.24 GB with the two neighbours and two
all-to-alls of about 0.95 GB each, roughly 2 GB in total. HiCHAN moves about 21 GB per stage and GPU for the same
grid without scalar (9 transforms with two transposes each). With an effective 5–15 GB/s per GPU between nodes:

| run | communication per step | computation per step | time per step |
|---|---|---|---|
| P0 (HiCHAN, no scalar) | 4–13 s | 1–3 s | 5–15 s |
| F1 | 0.4–1.3 s | 0.2–0.6 s | 0.6–2 s |
| F2 | 0.6–1.9 s | 0.3–0.9 s | 0.9–3 s |

So when HiCHAN is limited by the network, FDCHAN should be several times faster per step (about 5–10× for F1,
3–7× for F2). Treat these as correct to a factor of two at best; the all-to-all throughput on Arrhenius and the
efficiency of the untuned kernels are the main unknowns. If the `communication` line of FDCHAN's timing report is
small, the kernels are the limit and there is room for tuning.

## 6. What to send back

- compiler errors from step 1, if any;
- 2.1 and 2.2: the last lines of stdout and the `fit.py` result;
- 2.3, 3.1, 3.2: the last line of each `log.dat` (or the `diff`);
- 4 and 5: `timing_report.txt` of both codes, the `dt` column after 100 steps, and the memory line FDCHAN prints
  at start-up.
