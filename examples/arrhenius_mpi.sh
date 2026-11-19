#!/bin/bash
#
# FDCHAN on several Arrhenius GH200 GPUs: one MPI rank per GPU (4 per node), z-slabs.
# Account, partition and module names are placeholders: check them with `sinfo` and `module avail buildenv`.
# Build:  module load GPU/buildenv-nvhpc/25.9-cu13.0 && make GPU=cuda ARCH=sm_90 MPI=1 MPICXX=mpicxx
#
#SBATCH -J fdchan_mpi
#SBATCH -A naiss20XX-X-XX
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 01:00:00

module load GPU/buildenv-nvhpc/25.9-cu13.0
cd $SLURM_SUBMIT_DIR
# each rank selects GPU (node-local rank) mod 4 itself; nz/(number of ranks) should be >= 16
srun --mpi=pmix $HOME/fdchan/build/fdchan_mpi input.txt
