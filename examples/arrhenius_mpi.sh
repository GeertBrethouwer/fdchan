#!/bin/bash
#
# FDCHAN on several Arrhenius GH200 GPUs: one MPI rank per GPU (4 per node), z-slabs.
# Account, partition and module names are placeholders: check them with `sinfo` and `module avail buildenv`.
# Build:  module load GPU/buildenv-nvhpc/25.9-cu13.0
#         make GPU=cuda ARCH=sm_90 MPI=1 MPICXX=mpicxx EXE=build/fdchan_gpu_mpi
#
#SBATCH -J fdchan_mpi
#SBATCH -A naiss20XX-X-XX
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 01:00:00

module load GPU/buildenv-nvhpc/25.9-cu13.0

# MPI transport, chosen from the number of nodes of the job:
# - one node: shared memory only. Open MPI's libfabric (OFI) components abort in MPI_Init on Arrhenius
#   ("free(): invalid pointer" in libfabric.so, called from opal_common_ofi_fi_domain), so disable them;
#   GPU buffers then go through the smcuda BTL.
# - several nodes: libfabric with the Slingshot provider (cxi). Not yet tested: if MPI_Init still fails,
#   the FI_LOG_LEVEL=warn output says why; ask NSC which MPI to use for multi-node GPU jobs.
NNODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
if [ "$NNODES" -eq 1 ]; then
  export OMPI_MCA_btl=^ofi
  export OMPI_MCA_mtl=^ofi
  unset FI_PROVIDER
  echo "single node: Open MPI OFI components disabled"
else
  unset OMPI_MCA_btl OMPI_MCA_mtl
  export FI_PROVIDER=cxi
  export FI_LOG_LEVEL=warn
  echo "$NNODES nodes: libfabric provider $FI_PROVIDER"
fi
env | grep -E '^(OMPI_MCA|FI)_' | sort   # record the settings in the Slurm output

cd $SLURM_SUBMIT_DIR
# each rank selects GPU (node-local rank) mod 4 itself; nz/(number of ranks) should be >= 16
srun --mpi=pmix $HOME/fdchan/build/fdchan_gpu_mpi input.txt
