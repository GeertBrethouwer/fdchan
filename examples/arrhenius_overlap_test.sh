#!/bin/bash
#
# Test and timing of the overlap options (overlap_scalars, p_chunks) on Arrhenius with a production case: four
# short runs from the same restart file,
#   A: as in the input   B: overlap_scalars = 1   C: p_chunks = 4   D: both
# Checks:
#   - B must give a log.dat identical to A: the scalars only run at a different time, with the same kernels.
#   - C and D: identical to A except possibly max|div| (column 8) and the last digits of other columns. The z-FFTs
#     of the chunks use cuFFT plans with fewer lines, which may choose other algorithms (round-off).
#   A larger difference, or a hang, is a bug: please send me the output.
# The summary gives the time per step of the four runs.
#
# Usage (from any directory; the runs go into ./overlap_test_<jobid>):
#   sbatch -A <your-account> --export=ALL,RUN=/scratch/geert/fdchan_run $HOME/fdchan/examples/arrhenius_overlap_test.sh
# RUN: an FDCHAN run directory with input.txt and data/restart.bin (its data_dir must be data). NSTEP (default 200)
# steps per run, CHUNKS (default 4) for C and D. Nothing in RUN is changed. Use the node count of the production
# run (-N; here 2 nodes, 8 GPUs).
#
#SBATCH -J fd_overlap_test
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 01:30:00
#SBATCH -p gpu

module load GPU/buildenv-nvhpc/25.9-cu13.0
FD=${FD:-$HOME/fdchan}
EXE=${EXE:-$FD/build/fdchan_gpu_mpi}
RUN=${RUN:-/scratch/geert/fdchan_run}
NSTEP=${NSTEP:-200}
CHUNKS=${CHUNKS:-4}
NNODES=${SLURM_JOB_NUM_NODES:-1}
export NCCL_DEBUG=WARN
if [ "$NNODES" -eq 1 ]; then  # as examples/arrhenius_mpi.sh
  export OMPI_MCA_btl=^ofi OMPI_MCA_mtl=^ofi
  unset FI_PROVIDER
  NCCL_TASK_ENV="NCCL_NET_PLUGIN=none NCCL_NET=Socket"
else
  unset OMPI_MCA_btl OMPI_MCA_mtl
  export FI_PROVIDER=cxi
  NCCL_TASK_ENV=""
fi
[ -x $EXE ] || { echo "$EXE not found: build it (README)"; exit 1; }
[ -f $RUN/input.txt ] && [ -f $RUN/data/restart.bin ] || { echo "need $RUN/input.txt and $RUN/data/restart.bin"; exit 1; }
T=${SLURM_SUBMIT_DIR:-$PWD}/overlap_test_${SLURM_JOB_ID:-local}
mkdir -p $T
echo "FDCHAN $(cd $FD && git log -1 --format='%h %s' 2>/dev/null)"

declare -A OPT=( [A]="" [B]="overlap_scalars = 1" [C]="p_chunks = $CHUNKS" [D]="overlap_scalars = 1|p_chunks = $CHUNKS" )
for r in A B C D; do
  d=$T/$r; mkdir -p $d
  # the production input, with these items at the end (a later item overrides an earlier one)
  { cat $RUN/input.txt; echo
    printf "%s\n" "init_type = restart" "restart_file = $RUN/data/restart.bin" "nstep = $NSTEP" "data_dir = data" \
                  "iout_log = 20" "iout_restart = 0" "iout_save_stats = 0" "iout_stats = 0" "timing = 1" \
                  "iout_samples = 0" "tout_samples = 0"
    [ -n "${OPT[$r]}" ] && tr '|' '\n' <<< "${OPT[$r]}"; } > $d/input.txt
  echo "== run $r: ${OPT[$r]:-(as in the input)}"
  (cd $d && srun --mpi=pmix env $NCCL_TASK_ENV $EXE input.txt > out.txt 2>&1)
  echo "   exit code $?"
done

echo
echo "== summary ($NSTEP steps)"
for r in A B C D; do
  tps=$(grep -h "s per step" $T/$r/out.txt | tail -1 | sed -E 's/.*: ([0-9.eE+-]+) s per step.*/\1/')
  printf "  %s  %-40s %s s per step\n" $r "${OPT[$r]:-(as in the input)}" "${tps:-?}"
  grep -h "^overlap\|fdchan error\|NCCL error\|GPU error" $T/$r/out.txt | head -2 | sed 's/^/      /'
done
nodiv() { awk '!/^#/ {$8 = ""; print}' $1; }
cmp -s <(grep -v '^#' $T/A/data/log.dat) <(grep -v '^#' $T/B/data/log.dat) \
  && echo "  B = A: log.dat identical (overlap_scalars correct)" || echo "  B and A: log.dat DIFFERENT -- please send me $T"
for r in C D; do
  if cmp -s <(nodiv $T/A/data/log.dat) <(nodiv $T/$r/data/log.dat); then
    echo "  $r = A in all printed digits except max|div|"
  else
    echo "  $r and A differ (last lines below; round-off in the last digits is fine, more is not):"
    tail -1 $T/A/data/log.dat | cut -c1-130 | sed 's/^/     A /'; tail -1 $T/$r/data/log.dat | cut -c1-130 | sed 's/^/     '$r' /'
  fi
done
echo "  max|div| (column 8) at the end: A $(tail -1 $T/A/data/log.dat | awk '{print $8}'), C $(tail -1 $T/C/data/log.dat | awk '{print $8}')"
