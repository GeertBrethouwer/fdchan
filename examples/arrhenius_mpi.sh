#!/bin/bash
#
# FDCHAN on several Arrhenius GH200 GPUs: one MPI rank per GPU (4 per node), z-slabs.
# Account, partition and module names are placeholders: check them with `sinfo` and `module avail buildenv`.
# Build:  module load GPU/buildenv-nvhpc/25.9-cu13.0
#         make GPU=cuda ARCH=sm_90 MPI=1 NCCL=1 MPICXX=mpicxx EXE=build/fdchan_gpu_mpi
# With NCCL=1 the halo exchanges and all-to-alls go through NCCL (NVLink inside a node); MPI still starts the
# ranks and does the reductions and restart files. comm_backend = mpi in input.txt switches back to MPI.
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
#
# NCCL: inside a node it uses NVLink and needs no network plugin. Its libfabric (OFI) plugin (AWS_OFI_NCCL) fails
# on Arrhenius ("NET/OFI Couldn't create domain" -> "Failed to initialize any NET plugin" -> ncclCommInitRank
# "invalid usage"), so on one node NCCL_NET_PLUGIN=none and NCCL_NET=Socket (Socket is only NCCL's transport
# between nodes; the GPUs of the node still use NVLink). The site sets NCCL_NET_PLUGIN=.../libnccl-net-ofi.so
# and NCCL_NET="AWS Libfabric" again in every task, after srun has copied this environment, so exporting them
# here is not enough: they are set on the srun command line with env (NCCL_TASK_ENV below).
# Between nodes NCCL needs a working plugin for Slingshot (ask NSC); NCCL_NET=Socket works but is slow,
# otherwise set comm_backend = mpi in input.txt. NCCL_DEBUG=WARN prints the reason of a failure.
NNODES=${SLURM_JOB_NUM_NODES:-${SLURM_NNODES:-1}}
export NCCL_DEBUG=WARN
if [ "$NNODES" -eq 1 ]; then
  export OMPI_MCA_btl=^ofi
  export OMPI_MCA_mtl=^ofi
  unset FI_PROVIDER
  NCCL_TASK_ENV="NCCL_NET_PLUGIN=none NCCL_NET=Socket"
  echo "single node: Open MPI OFI components and the NCCL network plugin disabled"
else
  unset OMPI_MCA_btl OMPI_MCA_mtl
  NCCL_TASK_ENV=""
  export FI_PROVIDER=cxi
  export FI_LOG_LEVEL=warn
  echo "$NNODES nodes: libfabric provider $FI_PROVIDER"
fi
env | grep -E '^(OMPI_MCA|FI|NCCL)_' | sort   # record the settings in the Slurm output
echo "set in every task: ${NCCL_TASK_ENV:-(nothing)}"

cd $SLURM_SUBMIT_DIR
# each rank selects GPU (node-local rank) mod 4 itself; nz/(number of ranks) should be >= 16
# (to check what the tasks see: srun -n 1 env $NCCL_TASK_ENV bash -c 'env | grep ^NCCL_NET')
srun --mpi=pmix env $NCCL_TASK_ENV $HOME/fdchan/build/fdchan_gpu_mpi input.txt
