#!/bin/bash
# Shared helpers for Slurm-native UCC MPI tests.
#
# Sourced by run_tests_ucc_mpi_slurm_<group>.sh wrappers. Each wrapper is the
# script that slurmCI 'run' launches via `srun --ntasks-per-node=N` inside a
# pyxis/enroot container. srun/PMIx provides the MPI ranks, so the test binary
# is invoked directly (no mpirun, no hostfile, no ssh).

UCC_SRC_DIR="/opt/nvidia/src/ucc"
EXE="${UCC_SRC_DIR}/build/test/mpi/ucc_test_mpi"
EXE_ARGS="--inplace 2 --set_device 2 --root random:2 --count_bits 32,64 --displ_bits 32,64"

# DEV is set by mpi_slurm_setup and used by the run_* functions.
DEV=""

mpi_slurm_setup() {
    export UCX_WARN_UNUSED_ENV_VARS=n
    # Disable NCCL by default; the NCCL group re-enables it explicitly.
    export UCC_TL_NCCL_TUNE=0

    echo "=== UCC MPI slurm (job ${SLURM_JOB_ID:-?} rank ${SLURM_PROCID:-0}/${SLURM_NTASKS:-?} nodes ${SLURM_NNODES:-?}) ==="
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>&1 || echo "nvidia-smi failed"
    fi

    # Local IB device discovery (no ssh): first Active device.
    DEV=""
    if command -v ibstat >/dev/null 2>&1; then
        for d in $(ibstat -l 2>/dev/null); do
            state=$(ibstat "$d" 2>/dev/null | awk '/State:/{print $2; exit}')
            if [ "$state" = "Active" ]; then DEV="$d"; break; fi
        done
    fi
    if [ -n "$DEV" ]; then
        export UCX_NET_DEVICES="${DEV}:1"
        echo "INFO: using IB device ${DEV}"
    else
        echo "WARNING: no Active IB device found; UCX will auto-select"
    fi
}

# Bulk group (ppn=4, multi-node): default, TL/UCP, CL/HIER variants, 2-step
# bcast, and TL/MLX5 (self-skips without >=2 nodes + IB device).
mpi_slurm_run_bulk() {
    local MT TG
    for MT in "" "-T"; do
        TG="--triggered 0"

        echo "INFO: default configuration ..."
        # shellcheck disable=SC2086
        UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda
        echo "INFO: default configuration ... DONE"

        echo "INFO: TL/UCP ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic UCC_CL_BASIC_TLS=ucp UCX_LOG_LEVEL=info UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda
        echo "INFO: TL/UCP ... DONE"

        echo "INFO: CL/HIER ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=inf UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c alltoall,alltoallv,allreduce,barrier
        echo "INFO: CL/HIER ... DONE"

        echo "INFO: CL/HIER+ucp ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=inf UCC_CL_HIER_TLS=ucp UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c alltoall,alltoallv,allreduce,barrier
        echo "INFO: CL/HIER+ucp ... DONE"

        echo "INFO: CL/HIER+rab ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=allreduce:@rab:inf UCC_CL_HIER_TLS=ucp UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c allreduce
        echo "INFO: CL/HIER+rab ... DONE"

        echo "INFO: CL/HIER+split_rail ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=allreduce:@split_rail:inf UCC_CL_HIER_TLS=ucp UCC_TL_NCCL_TUNE=0 UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c allreduce
        echo "INFO: CL/HIER+split_rail ... DONE"

        echo "INFO: CL/HIER+split_rail+pipeline ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic,hier UCC_CL_HIER_TUNE=allreduce:@split_rail:inf UCC_CL_HIER_TLS=ucp UCC_TL_NCCL_TUNE=0 \
            UCC_CL_HIER_ALLREDUCE_SPLIT_RAIL_PIPELINE=thresh=0:fragsize=256K UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c allreduce
        echo "INFO: CL/HIER+split_rail+pipeline ... DONE"

        echo "INFO: CL/HIER+2step bcast ..."
        # shellcheck disable=SC2086
        UCC_CLS=all UCC_TLS="^sharp" UCC_CL_HIER_TUNE="bcast:0-inf:@2step" UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c bcast
        echo "INFO: CL/HIER+2step bcast ... DONE"

        if [ "${SLURM_NNODES:-1}" -ge 2 ] && [ -n "$DEV" ]; then
            echo "INFO: TL/MLX5 ..."
            # shellcheck disable=SC2086
            UCC_CLS=basic UCC_CL_BASIC_TLS=ucp,mlx5 UCC_TL_MLX5_NET_DEVICES="${DEV}:1" UCC_TL_MLX5_TUNE=inf \
                $EXE $EXE_ARGS $MT $TG --mtypes host,cuda -c alltoall -t world -d uint8 -O 0 -m 1:128
            echo "INFO: TL/MLX5 ... DONE"
        else
            echo "INFO: TL/MLX5 ... SKIPPED (needs >=2 nodes + Active IB device)"
        fi
    done
}

# NCCL group (ppn = GPUs per node): cuda-only collectives over TL/NCCL.
mpi_slurm_run_nccl() {
    local MT TG
    for MT in "" "-T"; do
        TG="--triggered 0"
        echo "INFO: NCCL ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic UCC_CL_BASIC_TLS=ucp,nccl UCC_TL_NCCL_TUNE=cuda:inf \
            NCCL_IB_HCA="${DEV}" NCCL_DEBUG=WARN \
            $EXE $EXE_ARGS $MT $TG --mtypes cuda
        echo "INFO: NCCL ... DONE"
    done
}

# TL/CUDA group (single node): cuda collectives over TL/CUDA.
mpi_slurm_run_tlcuda() {
    local MT TG
    for MT in "" "-T"; do
        TG="--triggered 0"
        echo "INFO: TL/CUDA ..."
        # shellcheck disable=SC2086
        UCC_CLS=basic UCC_CL_BASIC_TLS=ucp,cuda UCC_TL_CUDA_TUNE=cuda:inf UCX_TLS="^cuda_ipc" \
            $EXE $EXE_ARGS $MT $TG --mtypes cuda \
            -c alltoall,alltoallv,allgather,allgatherv,reduce_scatter,reduce_scatterv
        echo "INFO: TL/CUDA ... DONE"
    done
}
