#!/bin/bash
#
# Communication across Arrhenius nodes over Slingshot, in two parts:
#   A. MPI alone (tools/mpi_check): does MPI start on two nodes and reach every rank, with which settings?
#   B. NCCL without MPI (tools/nccl_check): does NCCL work across the nodes, with which network settings, and how
#      fast is it (all-to-all of all ranks, inter-node ring = the network alone, all-to-all inside each node)?
# One job step per setting; each writes <name>.log (every line prefixed with its task number), and a summary is
# printed at the end. A step that does not finish within 3 minutes is stopped.
#
# Build once (login node):
#   module load GPU/buildenv-nvhpc/25.9-cu13.0
#   cd $HOME/fdchan && make GPU=cuda ARCH=sm_90 NCCL=1 build/nccl_check
#   make GPU=cuda ARCH=sm_90 MPI=1 MPICXX=mpicxx build/mpi_check
# Submit from an empty directory:  sbatch $HOME/fdchan/examples/arrhenius_nccl_test.sh
#
#SBATCH -J comm_test
#SBATCH -A naiss20XX-X-XX
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=72
#SBATCH -t 00:45:00

module load GPU/buildenv-nvhpc/25.9-cu13.0
B=$HOME/fdchan/build
MIB=${MIB:-64}          # MiB per pair of ranks for the NCCL test; sbatch --export=ALL,MIB=256 ... for more
cd $SLURM_SUBMIT_DIR
for x in nccl_check mpi_check; do
  if [ ! -x $B/$x ]; then echo "$B/$x not found: build it first (see the top of this script)"; exit 1; fi
done

echo "== job environment (the site sets the NCCL ones again in every task)"
env | grep -E '^(NCCL|FI|OMPI_MCA)_' | sort
echo "== tasks see:"
srun -N 1 -n 1 bash -c 'env | grep -E "^(NCCL_NET|FI_PROVIDER|OMPI_MCA)" | sort'
if command -v fi_info >/dev/null; then
  echo "== libfabric providers on a compute node:"
  srun -N 1 -n 1 fi_info -l 2>&1 | grep -v version | tr -d '\n'; echo
fi

# Every setting is given with env on the srun line (exported here, the site would overwrite the NCCL ones in
# the tasks). -u removes a variable. -l prefixes every line with its task number.
step() {  # step <name> <mpi|none> <settings...> -- <program and arguments>
  local name=$1 mpi=$2; shift 2
  local vars=()
  while [ "$1" != "--" ]; do vars+=("$1"); shift; done
  shift
  echo
  echo "== $name: ${vars[*]}"
  srun --mpi=$mpi -l --time=3 --kill-on-bad-exit=0 env "${vars[@]}" "$@" > $name.log 2>&1
  echo "   exit code $?"
  if grep -q "RESULT" $name.log; then
    grep -E "RESULT|Using network|Selected provider|Couldn't create domain|Failed to initialize" $name.log \
      | sed 's/^ *[0-9]*: //' | sort | uniq -c | sort -rn | head -10
  else
    echo "   no RESULT; last stage of each task and the errors:"
    grep -E "stage" $name.log | sort -t: -k1,1n -s | awk -F: '{last[$1]=$0} END {for (t in last) print "   " last[t]}' | sort
    grep -viE "stage|NCCL INFO|^ *[0-9]*: *$" $name.log | sort | uniq -c | sort -rn | head -12
  fi
}

# ---- A. MPI (FDCHAN needs it to start and for its reductions, on any number of nodes)
# A1: Open MPI's libfabric components off (as in the one-node job scripts): TCP between the nodes
step mpi_ofi_off pmix OMPI_MCA_btl=^ofi OMPI_MCA_mtl=^ofi -- $B/mpi_check
# A2: the same, TCP on the Slingshot interface
step mpi_ofi_off_hsn pmix OMPI_MCA_btl=^ofi OMPI_MCA_mtl=^ofi OMPI_MCA_btl_tcp_if_include=hsn0 -- $B/mpi_check
# A3: Open MPI with libfabric, Slingshot provider (crashed in MPI_Init on one node: does it on two?)
step mpi_cxi pmix -u OMPI_MCA_btl -u OMPI_MCA_mtl FI_PROVIDER=cxi -- $B/mpi_check
# A4: the site's settings as they are
step mpi_site pmix -u OMPI_MCA_btl -u OMPI_MCA_mtl -- $B/mpi_check

# ---- B. NCCL without MPI
DBG="NCCL_DEBUG=INFO NCCL_DEBUG_SUBSYS=INIT,NET"
# B1: the site's NCCL settings (AWS OFI plugin, NCCL_NET="AWS Libfabric", FI_PROVIDER not set)
step nccl_site none $DBG -- $B/nccl_check $MIB
# B2: the same, libfabric restricted to the Slingshot provider
step nccl_cxi none $DBG FI_PROVIDER=cxi -- $B/nccl_check $MIB
# B3: cxi with the settings recommended for NCCL on Slingshot
step nccl_cxi_tuned none $DBG FI_PROVIDER=cxi FI_CXI_DISABLE_HOST_REGISTER=1 FI_CXI_DEFAULT_CQ_SIZE=131072 \
     FI_CXI_DEFAULT_TX_SIZE=32768 NCCL_NET_GDR_LEVEL=PHB NCCL_CROSS_NIC=1 -- $B/nccl_check $MIB
# B4: reference: no plugin, NCCL's TCP sockets between the nodes (slow)
step nccl_socket none $DBG NCCL_NET_PLUGIN=none NCCL_NET=Socket -- $B/nccl_check $MIB

echo
echo "== summary"
for name in mpi_ofi_off mpi_ofi_off_hsn mpi_cxi mpi_site nccl_site nccl_cxi nccl_cxi_tuned nccl_socket; do
  r=$(grep -h "RESULT" $name.log 2>/dev/null | sed 's/^ *[0-9]*: //')
  printf "%-16s %s\n" $name "${r:-no RESULT (see $name.log)}" | sed '2,$s/^/                 /'
done
