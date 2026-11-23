#!/bin/bash
#
# Does NCCL work across Arrhenius nodes over Slingshot? Runs tools/nccl_check (all-to-all of all ranks, an
# inter-node ring that uses only the network, and an all-to-all inside each node) with several network
# settings, one job step each, and prints a summary at the end. Each step writes nccl_<name>.log.
#
# Build once (login node):
#   module load GPU/buildenv-nvhpc/25.9-cu13.0
#   cd $HOME/fdchan && make GPU=cuda ARCH=sm_90 MPI=1 NCCL=1 MPICXX=mpicxx build/nccl_check
# Submit from an empty directory:  sbatch $HOME/fdchan/examples/arrhenius_nccl_test.sh
#
#SBATCH -J nccl_test
#SBATCH -A naiss20XX-X-XX
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 00:30:00

module load GPU/buildenv-nvhpc/25.9-cu13.0
EXE=$HOME/fdchan/build/nccl_check
MIB=${MIB:-64}          # MiB per pair of ranks; sbatch --export=ALL,MIB=256 ... for larger messages
cd $SLURM_SUBMIT_DIR
if [ ! -x "$EXE" ]; then echo "$EXE not found: build it first (see the top of this script)"; exit 1; fi

# MPI only starts the ranks and exchanges the NCCL id: keep Open MPI off libfabric (its OFI components abort in
# MPI_Init on Arrhenius); the bootstrap then uses TCP, which is fine for a few bytes.
export OMPI_MCA_btl=^ofi
export OMPI_MCA_mtl=^ofi

echo "== site settings (the job environment; the site sets the NCCL ones again in every task)"
env | grep -E '^(NCCL|FI)_' | sort
echo "== tasks see:"
srun -N 1 -n 1 bash -c 'env | grep -E "^(NCCL_NET|FI_PROVIDER)" | sort'
if command -v fi_info >/dev/null; then
  echo "== libfabric providers on a compute node (fi_info -l):"
  srun -N 1 -n 1 fi_info -l 2>&1 | head -20
fi

# One step per setting. The variables are given with env on the srun line: exported here they would be
# overwritten in the tasks by the site's settings. NCCL_DEBUG=INFO shows the network NCCL chooses.
DBG="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
run() {
  local name=$1; shift
  echo
  echo "== $name: $*"
  srun --mpi=pmix --time=5 env $DBG "$@" $EXE $MIB > nccl_$name.log 2>&1
  echo "   exit code $?"
  grep -E "RESULT|Using network|Selected provider|Couldn't create domain|Failed to initialize|NCCL WARN" nccl_$name.log \
    | sort | uniq -c | sort -rn | head -12
}

# 1. the site's settings as they are (AWS OFI plugin, NCCL_NET="AWS Libfabric", FI_PROVIDER not set)
run site
# 2. the same, but libfabric restricted to the Slingshot provider
run cxi FI_PROVIDER=cxi
# 3. cxi with the settings that HPE and other Slingshot sites recommend for NCCL
run cxi_tuned FI_PROVIDER=cxi FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 \
    FI_CXI_DEFAULT_TX_SIZE=32768 NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1
# 4. reference: no plugin, NCCL's TCP sockets between the nodes (works, but slow)
run socket NCCL_NET_PLUGIN=none NCCL_NET=Socket

echo
echo "== summary (inter-node ring = the network alone; compare with the socket line)"
for f in nccl_site.log nccl_cxi.log nccl_cxi_tuned.log nccl_socket.log; do
  echo "-- $f"
  grep -E "^RESULT" $f || echo "   no RESULT lines: see $f"
done
