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

# Minimal end-to-end sanity: confirms srun/PMIx launch + CUDA memtype work.
mpi_slurm_run_smoke() {
    echo "INFO: smoke - barrier + small allreduce on host,cuda ..."
    # shellcheck disable=SC2086
    UCX_TLS="^cuda_ipc" $EXE $EXE_ARGS --mtypes host,cuda -c barrier,allreduce -m 1:1024
    echo "INFO: smoke ... DONE"
}
