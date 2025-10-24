#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/env.sh"

HOSTFILE="$1"

if [ -z "$HOSTFILE" ]; then
    echo "ERROR: HOSTFILE is not specified"
    exit 1
fi

HEAD_NODE=$(head -1 "$HOSTFILE")
export HEAD_NODE
export MASTER_ADDR=${HEAD_NODE}

NNODES=$(wc --lines "$HOSTFILE" | awk '{print $1}')
DEV=""
CX7_DEV=""

# Find first available active device
for d in $(ssh $HEAD_NODE "ibstat -l"); do
    state=$(ssh $HEAD_NODE "ibstat $d" | grep 'State:' | awk '{print $2}')
    type=$(ssh $HEAD_NODE "ibstat $d" | grep 'CA type:' | awk '{print $3}')
    if [ $state == 'Active' ]; then
        if [ "$DEV" == "" ]; then
            DEV=$d
        fi
        if [ "$type" == 'MT4129' ]; then
            CX7_DEV=$d
            break
        fi
    fi
done

if [ "x$DEV" == "x" ]; then
    echo "No active devices are found on $HEAD_NODE"
    exit -1
fi

# Verify the IB port is actually Up (not just Active in ibstat)
ib_port_state=$(ssh $HEAD_NODE "ibdev2netdev" | grep "$DEV" | grep -c "Up" || true)
if [ "$ib_port_state" -eq 0 ]; then
    echo "ERROR: InfiniBand device $DEV port is Down on $HEAD_NODE"
    echo "Please bring up the port with: sudo ip link set <interface> up"
    ssh $HEAD_NODE "ibdev2netdev"
    exit 1
fi
echo "INFO: Using InfiniBand device $DEV (port is Up)"

function mpi_params {
    ppn=$1
    nnodes=$2
    if [ "x$nnodes" == "x" ]; then
        nnodes=$NNODES
    fi
    echo "-np $((nnodes*ppn)) --oversubscribe --hostfile ${HOSTFILE} \
--map-by ppr:$ppn:node --bind-to socket  \
-x PATH -x LD_LIBRARY_PATH --mca opal_common_ucx_opal_mem_hooks 1 --mca plm_rsh_args -p12345 \
--mca coll ^ucc,hcoll \
-x UCX_NET_DEVICES=$DEV:1"
}

# # shellcheck disable=SC2086
mpirun $(mpi_params 1) hostname


# # shellcheck disable=SC2086
mpirun $(mpi_params 1) cat /proc/1/cgroup

NGPUS=$(ssh $HEAD_NODE 'nvidia-smi -L | wc -l')
PPN=4
EXE=/opt/nvidia/src/ucc/build/test/mpi/ucc_test_mpi
EXE+=" --inplace 2 --set_device 2 --root random:2 --count_bits 32,64 --displ_bits 32,64"

start=`date +%s`

for MT in "" "-T"; do
    if [ $MT == "-T" ]; then
        TG="--triggered 0"
    else
        # disabled so far, need to fix current issues and re-enable for CI
        TG="--triggered 0"
    fi
    echo "INFO: UCC MPI unit tests (default configuration) ..."
    # shellcheck disable=SC2086
    default_args="-x UCC_TL_NCCL_TUNE=0"
    # disable cuda_ipc transport of UCX in some cases since it's not compatible with UCC TL CUDA
    ucx_tls_no_cuda_ipc="-x UCX_TLS=^cuda_ipc"

    mpirun $(mpi_params $PPN) $default_args $ucx_tls_no_cuda_ipc $EXE $MT $TG --mtypes host,cuda
    echo "INFO: UCC MPI unit tests (default configuration) ... DONE"


    echo "INFO: UCC MPI unit tests (NCCL) ..."
    # shellcheck disable=SC2086
    # Configure NCCL to use the same IB device as UCX for bootstrap communication
    nccl_args=" -x UCC_CLS=basic -x UCC_CL_BASIC_TLS=ucp,nccl -x UCC_TL_NCCL_TUNE=cuda:inf "
    nccl_args+=" -x NCCL_IB_HCA=${DEV} -x NCCL_DEBUG=WARN "
    mpirun $(mpi_params $NGPUS) $nccl_args $EXE $MT $TG --mtypes cuda
    echo "INFO: UCC MPI unit tests (NCCL) ... DONE"


    echo "INFO: UCC MPI unit tests (TL/UCP) ..."
    # shellcheck disable=SC2086
    tlucp_args=" -x UCC_CLS=basic -x UCC_CL_BASIC_TLS=ucp "
    mpirun $(mpi_params $PPN) $tlucp_args $EXE $MT $TG --mtypes host,cuda
    echo "INFO: UCC MPI unit tests (TL/UCP) ... DONE"


    echo "INFO: UCC MPI unit tests (TL/CUDA) ..."
    # shellcheck disable=SC2086
    tlcuda_args=" -x UCC_CLS=basic -x UCC_CL_BASIC_TLS=ucp,cuda -x UCC_TL_CUDA_TUNE=cuda:inf "
    tlcuda_colls="alltoall,alltoallv,allgather,allgatherv,reduce_scatter,reduce_scatterv"
    mpirun $(mpi_params $PPN 1) $ucx_tls_no_cuda_ipc $tlcuda_args $EXE $MT $TG --mtypes cuda -c $tlcuda_colls
    echo "INFO: UCC MPI unit tests (TL/CUDA) ... DONE"

    echo "INFO: UCC MPI unit tests (TL/MLX5) ..."
    # shellcheck disable=SC2086
    if [ "x$CX7_DEV" == "x" ]; then
        echo "WARNING: No active CX7 devices found on ${HEAD_NODE}"
        echo "INFO: UCC MPI unit tests (TL/MLX5) ... SKIPPED"
    elif [ $NNODES -lt 2 ]; then
        echo "WARNING: At least two nodes are required, but only $NNODES are available"
        echo "INFO: UCC MPI unit tests (TL/MLX5) ... SKIPPED"
    else
        tlmlx5_args=" -x UCC_CLS=basic -x UCC_CL_BASIC_TLS=ucp,mlx5 -x UCC_TL_MLX5_NET_DEVICES=$CX7_DEV:1 -x UCC_TL_MLX5_TUNE=inf "
        tlmlx5_colls="alltoall"
        mpirun $(mpi_params $PPN) $tlmlx5_args $EXE $MT $TG --mtypes host,cuda -c $tlmlx5_colls -t world -d uint8 -O 0 -m 1:128
        echo "INFO: UCC MPI unit tests (TL/MLX5) ... DONE"
    fi

    echo "INFO: UCC MPI unit tests (CL/HIER) ..."
    # shellcheck disable=SC2086
    clhier_args=" -x UCC_CLS=basic,hier -x UCC_CL_HIER_TUNE=inf -x UCC_TL_NCCL_TUNE=0 "
    clhier_colls="alltoall,alltoallv,allreduce,barrier"
    mpirun $(mpi_params $PPN) $ucx_tls_no_cuda_ipc $clhier_args $EXE $MT $TG --mtypes host,cuda -c $clhier_colls
    echo "INFO: UCC MPI unit tests (CL/HIER) ... DONE"


    echo "INFO: UCC MPI unit tests (CL/HIER+ucp) ..."
    # shellcheck disable=SC2086
    clhier_args=" -x UCC_CLS=basic,hier -x UCC_CL_HIER_TUNE=inf -x UCC_CL_HIER_TLS=ucp -x UCC_TL_NCCL_TUNE=0 "
    clhier_colls="alltoall,alltoallv,allreduce,barrier"
    mpirun $(mpi_params $PPN) $ucx_tls_no_cuda_ipc $clhier_args $EXE $MT $TG --mtypes host,cuda -c $clhier_colls
    echo "INFO: UCC MPI unit tests (CL/HIER+ucp) ... DONE"


    echo "INFO: UCC MPI unit tests (CL/HIER+rab) ..."
    # shellcheck disable=SC2086
    clhier_args=" -x UCC_CLS=basic,hier -x UCC_CL_HIER_TUNE=allreduce:@rab:inf -x UCC_CL_HIER_TLS=ucp -x UCC_TL_NCCL_TUNE=0 "
    clhier_colls="allreduce"
    mpirun $(mpi_params $PPN) $ucx_tls_no_cuda_ipc $clhier_args $EXE $MT $TG --mtypes host,cuda -c $clhier_colls
    echo "INFO: UCC MPI unit tests (CL/HIER+rab) ... DONE"


    echo "INFO: UCC MPI unit tests (CL/HIER+split_rail) ..."
    # shellcheck disable=SC2086
    clhier_args=" -x UCC_CLS=basic,hier -x UCC_CL_HIER_TUNE=allreduce:@split_rail:inf -x UCC_CL_HIER_TLS=ucp -x UCC_TL_NCCL_TUNE=0 "
    clhier_colls="allreduce"
    mpirun $(mpi_params $PPN) $ucx_tls_no_cuda_ipc $clhier_args $EXE $MT $TG --mtypes host,cuda -c $clhier_colls
    echo "INFO: UCC MPI unit tests (CL/HIER+split_rail) ... DONE"

    echo "INFO: UCC MPI unit tests (CL/HIER+split_rail+pipeline) ..."
    # shellcheck disable=SC2086
    clhier_args=" -x UCC_CLS=basic,hier -x UCC_CL_HIER_TUNE=allreduce:@split_rail:inf -x UCC_CL_HIER_TLS=ucp -x UCC_TL_NCCL_TUNE=0 "
    clhier_args+=" -x UCC_CL_HIER_ALLREDUCE_SPLIT_RAIL_PIPELINE=thresh=0:fragsize=256K "
    clhier_colls="allreduce"
    mpirun $(mpi_params $PPN) $ucx_tls_no_cuda_ipc $clhier_args $EXE $MT $TG --mtypes host,cuda -c $clhier_colls
    echo "INFO: UCC MPI unit tests (CL/HIER+split_rail+pipeline) ... DONE"
done

end=`date +%s`

echo Tests took $((end - start)) seconds
