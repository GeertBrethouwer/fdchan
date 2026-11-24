#!/bin/bash
#
# Checks of the scalars (nscal > 0) on ONE Arrhenius node (4 GH200), Test.md section 3.5:
#   1. tests/test_scalars on 1 GPU and on 4 GPUs: decay of single modes (Dirichlet/Neumann, theta = 0, 0.5, 1)
#      against the exact decay, steady profiles with Neumann walls and sources, hydrostatic balance, and the three
#      forms of the advection term;
#   2. the solver with examples/turb_scalars (two scalars, one buoyant; 30 steps) on 1 and 2 GPUs: log.dat must
#      agree to all printed digits except max|div|;
#   3. restart: a 1-GPU run of 20 steps, its restart file read on 2 GPUs: the first log line must equal the line of
#      step 20 of the 30-step run (all columns except step, dt, dpdx_applied and max|div|).
# Every step writes <name>.log; a summary is printed at the end.
#
# Build once (login node):
#   module load GPU/buildenv-nvhpc/25.9-cu13.0
#   cd $HOME/fdchan && make GPU=cuda ARCH=sm_90 MPI=1 NCCL=1 MPICXX=mpicxx EXE=build/fdchan_gpu_mpi
#   rm -f build/test_scalars && make GPU=cuda ARCH=sm_90 MPI=1 NCCL=1 MPICXX=mpicxx build/test_scalars
# Submit from an empty directory:  sbatch $HOME/fdchan/examples/arrhenius_scalars_test.sh
#
#SBATCH -J scalars_test
#SBATCH -A naiss20XX-X-XX
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 00:30:00

module load GPU/buildenv-nvhpc/25.9-cu13.0
FD=${FD:-$HOME/fdchan}                  # the checkout to test (FD=$HOME/fdchan_sp for single precision)
cd $SLURM_SUBMIT_DIR
for x in build/test_scalars build/fdchan_gpu_mpi; do
  if [ ! -x $FD/$x ]; then echo "$FD/$x not found: build it first (see the top of this script)"; exit 1; fi
done

# One node (as examples/arrhenius_mpi.sh): Open MPI's libfabric components crash in MPI_Init there, and NCCL needs
# no network plugin (NVLink). The site resets the NCCL variables in every task, so they go on the srun line.
export OMPI_MCA_btl=^ofi OMPI_MCA_mtl=^ofi
export NCCL_DEBUG=WARN
TASKENV="NCCL_NET_PLUGIN=none NCCL_NET=Socket"

step() {  # step <name> <ntasks> <command...>: one job step with ntasks tasks and as many GPUs
  local name=$1 n=$2; shift 2
  echo "== $name ($n GPU(s)): $*"
  srun --mpi=pmix -n $n --gpus=$n --exact --time=15 env $TASKENV "$@" > $name.log 2>&1
  echo "   exit code $?"
}

# ---- 1. the unit tests
step test_1gpu 1 $FD/build/test_scalars
step test_4gpu 4 $FD/build/test_scalars

# ---- 2. the solver, 1 and 2 GPUs
for n in 1 2; do
  d=solver_${n}gpu
  mkdir -p $d
  sed -E '/^(nstep|tmax|iout_[a-z_]+|tstats_start|timing|data_dir) *=/d' \
      $FD/examples/turb_scalars/input.txt > $d/input.txt
  printf "nstep = 30\niout_log = 10\niout_stats = 5\ntstats_start = 0\niout_save_stats = 10\ntiming = 1\n" >> $d/input.txt
  (cd $d && step solver $n $FD/build/fdchan_gpu_mpi input.txt && mv solver.log ../$d.log)
done

# ---- 3. restart: 20 steps on 1 GPU (restart file at the end), then one step on 2 GPUs from it
mkdir -p restart_src restart_2gpu
sed 's/^nstep = 30/nstep = 20/' solver_1gpu/input.txt > restart_src/input.txt
(cd restart_src && step solver 1 $FD/build/fdchan_gpu_mpi input.txt && mv solver.log ../restart_src.log)
sed 's/^nstep = 30/nstep = 1/' solver_1gpu/input.txt > restart_2gpu/input.txt
printf "init_type = restart\nrestart_file = ../restart_src/data/restart.bin\n" >> restart_2gpu/input.txt
(cd restart_2gpu && step solver 2 $FD/build/fdchan_gpu_mpi input.txt && mv solver.log ../restart_2gpu.log)

echo
echo "== summary"
for f in test_1gpu test_4gpu; do
  printf "%-12s %s\n" $f "$(grep -E 'scalars: (OK|FAIL)' $f.log || echo 'no result, see '$f.log)"
  grep -h "WRONG" $f.log | sed 's/^/             /'
done
for n in 1 2; do
  printf "%-12s %s\n" solver_${n}gpu "$(grep -m1 'FDCHAN (GPU' solver_${n}gpu.log)"
  grep -h "^scalars\|fdchan error" solver_${n}gpu.log | head -3 | sed 's/^/             /'
done
if [ -f solver_1gpu/data/log.dat ] && [ -f solver_2gpu/data/log.dat ]; then
  if cmp -s <(awk '{$8=""; print}' solver_1gpu/data/log.dat) <(awk '{$8=""; print}' solver_2gpu/data/log.dat); then
    echo "solver 1 vs 2 GPUs: log.dat agrees in all printed digits except max|div|"
  else
    echo "solver 1 vs 2 GPUs: log.dat DIFFERS (last lines):"
    tail -1 solver_1gpu/data/log.dat; tail -1 solver_2gpu/data/log.dat
  fi
  tail -1 solver_1gpu/data/log.dat | awk '{printf "step 30: max|div| %s, dS1/dy walls %s %s, dS2/dy walls %s %s\n", $8, $11, $12, $13, $14}'
fi
if [ -f restart_2gpu/data/log.dat ]; then
  # without step, dt and dpdx_applied (reset at a restart) and max|div|
  a=$(awk '$1==20 {$1=""; $3=""; $5=""; $8=""; print}' solver_1gpu/data/log.dat)
  b=$(awk '$1==0 {$1=""; $3=""; $5=""; $8=""; print}' restart_2gpu/data/log.dat)
  [ "$a" = "$b" ] && echo "restart 1 -> 2 GPUs: identical to step 20" || { echo "restart 1 -> 2 GPUs: DIFFERS"; echo "$a"; echo "$b"; }
fi
grep -h "of which scalars\|scalars (update" solver_1gpu/data/timing_report.txt 2>/dev/null | sed 's/^/timing (1 GPU): /'
