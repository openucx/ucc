#!/bin/bash -eEx
set -o pipefail

# Slurm-native DLRM (torch_ucc) test. One rank per node under
# `srun --ntasks-per-node=1`. Maps Slurm env -> torch distributed env, then
# runs the existing dlrm python launcher.

UCC_SRC_DIR="/opt/nvidia/src/ucc"

# Master address = first node of the Slurm allocation.
if command -v scontrol >/dev/null 2>&1 && [ -n "${SLURM_JOB_NODELIST:-}" ]; then
    MASTER_ADDR=$(scontrol show hostnames "${SLURM_JOB_NODELIST}" | head -n1)
else
    MASTER_ADDR=$(hostname -s)
fi
export MASTER_ADDR
export MASTER_PORT="${MASTER_PORT:-12346}"
export RANK="${SLURM_PROCID:-0}"
export WORLD_SIZE="${SLURM_NTASKS:-1}"
export LOCAL_RANK="${SLURM_LOCALID:-0}"
export CPU_GPU_MODE="gpu"

# Same UCC configuration as the bare-metal DLRM path.
export UCC_CLS=basic
export UCC_CL_BASIC_TLS=nccl,ucp

echo "=== DLRM slurm (job ${SLURM_JOB_ID:-?}) RANK=${RANK}/${WORLD_SIZE} LOCAL_RANK=${LOCAL_RANK} MASTER_ADDR=${MASTER_ADDR}:${MASTER_PORT} ==="

exec "${UCC_SRC_DIR}/.ci/scripts/run_dlrm_s_pytorch.sh"
