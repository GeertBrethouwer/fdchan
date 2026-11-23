#!/bin/bash
#
# Checks of the compact z-differences (z_scheme = compact) on ONE Arrhenius node (4 GH200), Test.md section 3.4:
#   1. tests/test_zcompact on 1 GPU and on 4 GPUs: D1z, D2z of order 4, 6, 8 against the exact answer of the
#      periodic scheme (round-off; on 4 GPUs this checks the distributed solve, uneven slabs) and their orders;
#   2. the solver with z_scheme = compact, fd_order_z = 8 (turbulent channel from noise, nz = 100, 30 steps) on
#      1 and 2 GPUs: log.dat must agree to all printed digits except max|div|, which must stay at round-off.
# Every step writes <name>.log; a summary is printed at the end.
#
# Build once (login node):
#   module load GPU/buildenv-nvhpc/25.9-cu13.0
#   cd $HOME/fdchan && make GPU=cuda ARCH=sm_90 MPI=1 NCCL=1 MPICXX=mpicxx EXE=build/fdchan_gpu_mpi
#   make GPU=cuda ARCH=sm_90 MPI=1 NCCL=1 MPICXX=mpicxx build/test_zcompact
# Submit from an empty directory:  sbatch $HOME/fdchan/examples/arrhenius_zcompact_test.sh
#
#SBATCH -J zcompact_test
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
for x in build/test_zcompact build/fdchan_gpu_mpi; do
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
  srun --mpi=pmix -n $n --gpus=$n --exact --time=10 env $TASKENV "$@" > $name.log 2>&1
  echo "   exit code $?"
}

# ---- 1. the derivatives
step test_1gpu 1 $FD/build/test_zcompact
step test_4gpu 4 $FD/build/test_zcompact

# ---- 2. the solver, 1 and 2 GPUs
for n in 1 2; do
  d=solver_${n}gpu
  mkdir -p $d
  sed -E '/^(nz|nstep|tmax|iout_[a-z_]+|tstats_start|timing|z_scheme|fd_order_z|data_dir) *=/d' \
      $FD/examples/turb_coarse/input.txt > $d/input.txt
  printf "nz = 100\nnstep = 30\niout_log = 10\nz_scheme = compact\nfd_order_z = 8\n" >> $d/input.txt
  (cd $d && step solver $n $FD/build/fdchan_gpu_mpi input.txt && mv solver.log ../$d.log)
done

echo
echo "== summary"
for f in test_1gpu test_4gpu; do
  printf "%-12s %s\n" $f "$(grep -E 'compact z: (OK|FAIL)' $f.log || echo 'no result, see '$f.log)"
done
grep -h "planes per rank" test_4gpu.log | sed 's/; vs exact derivative.*//' | head -9
for n in 1 2; do
  printf "%-12s %s\n" solver_${n}gpu "$(grep -m1 'FDCHAN (GPU' solver_${n}gpu.log)"
  grep -h "z compact\|fdchan error" solver_${n}gpu.log | head -2 | sed 's/^/             /'
done
if [ -f solver_1gpu/data/log.dat ] && [ -f solver_2gpu/data/log.dat ]; then
  # all columns except 9 (max|div|)
  if cmp -s <(awk '{$9=""; print}' solver_1gpu/data/log.dat) <(awk '{$9=""; print}' solver_2gpu/data/log.dat); then
    echo "solver 1 vs 2 GPUs: log.dat agrees in all printed digits except max|div|"
  else
    echo "solver 1 vs 2 GPUs: log.dat DIFFERS (last lines):"
    tail -1 solver_1gpu/data/log.dat; tail -1 solver_2gpu/data/log.dat
  fi
  for n in 1 2; do
    awk -v n=$n 'NR>1 && $9>m{m=$9} END{print "solver_" n "gpu max|div| over the run: " m}' solver_${n}gpu/data/log.dat
  done
fi
