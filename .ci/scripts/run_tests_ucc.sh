#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

UCC_SRC_DIR="$(cd "${SCRIPT_DIR}/../../" && pwd -P)"
cd "${UCC_SRC_DIR}/build"

export UCX_WARN_UNUSED_ENV_VARS=n
# Disable NCCL
export UCC_TL_NCCL_TUNE=0

# --- Slurm/GPU diagnostics (when running under Slurm) ---
if [ -n "${SLURM_JOB_ID}" ]; then
    echo "=== Slurm/GPU diagnostics (job ${SLURM_JOB_ID}, rank ${SLURM_PROCID:-0}) ==="
    echo "SLURM_JOB_ID=${SLURM_JOB_ID} SLURM_PROCID=${SLURM_PROCID:-0} SLURM_LOCALID=${SLURM_LOCALID:-N/A}"
    echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
    echo "NVIDIA_VISIBLE_DEVICES=${NVIDIA_VISIBLE_DEVICES:-<unset>}"
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader 2>&1 || echo "nvidia-smi failed or no GPUs"
    else
        echo "nvidia-smi not found"
    fi
    echo "=== end diagnostics ==="
fi

make gtest
