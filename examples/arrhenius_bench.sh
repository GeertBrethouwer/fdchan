#!/bin/bash
#
# Time per step of FDCHAN and HiCHAN on one Arrhenius GH200, for the same channel case.
# Account, partition and module names are placeholders: check them with `sinfo` and `module avail buildenv`.
#
#SBATCH -J fdchan_bench
#SBATCH -A naiss20XX-X-XX
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH -t 01:00:00

module load GPU/buildenv-nvhpc/25.9-cu13.0

FDCHAN=$HOME/fdchan/build/fdchan_gpu      # make GPU=cuda ARCH=sm_90 EXE=build/fdchan_gpu (see Test.md)
HICHAN=$HOME/hichan/build/hichan          # make CONFIG=configs/arrhenius.mk
export LD_LIBRARY_PATH=$HOME/hichan/dependencies/cuDecomp/build/lib:$LD_LIBRARY_PATH
export NCCL_NET_PLUGIN=none NCCL_NET=Socket NCCL_DEBUG=WARN

cd $SLURM_SUBMIT_DIR
NSTEP=500

# FDCHAN: same grid size as HiCHAN (128 x 129 x 128) and the grid of examples/retau180 (1.5x in z)
for size in "128 129 128" "128 129 192"; do
  set -- $size
  d=fd_$1_$2_$3; mkdir -p $d
  sed -e "s/^nx = .*/nx = $1/" -e "s/^ny = .*/ny = $2/" -e "s/^nz = .*/nz = $3/" \
      -e "s/^nstep = .*/nstep = $NSTEP/" -e "s#^iout_.*##" $HOME/fdchan/examples/retau180/input.txt > $d/input.txt
  echo "data_dir = $d/data" >> $d/input.txt
  echo "iout_log = 100" >> $d/input.txt
  srun -n 1 $FDCHAN $d/input.txt | tee $d/out.txt | tail -3
done

# HiCHAN: its own examples/retau180, without scalars, NSTEP steps
d=hichan_retau180; mkdir -p $d
sed -e "s/nstep *= *[0-9]*/nstep = $NSTEP/" -e "s/nscal *= *[0-9]*/nscal = 0/" \
    $HOME/hichan/examples/retau180/input.nml > $d/input.nml
(cd $d && srun -n 1 $HICHAN | tee out.txt | tail -3)

grep -h -E "s per step|per time step" fd_*/out.txt hichan_retau180/out.txt
