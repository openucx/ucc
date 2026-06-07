#!/bin/bash -eEx
set -o pipefail

# Slurm-native DLRM (torch_ucc) test. One rank per node under
# `srun --ntasks-per-node=1`. Maps Slurm env -> torch distributed env, then
# runs the existing dlrm python launcher.

UCC_SRC_DIR="/opt/nvidia/src/ucc"

# Master address = first node of the Slurm allocation. This MUST be identical
# on every rank, so derive it from the allocation node list -- never from each
# rank's own hostname (that makes ranks disagree and torch rendezvous hangs).
# `scontrol` is not present in this container image, so parse SLURM_NODELIST
# directly, handling compact forms like "funk[06,20]" and "funk[06-08]".
first_slurm_node() {
    local nl="${SLURM_JOB_NODELIST:-${SLURM_NODELIST:-}}"
    [ -z "$nl" ] && return 1
    if command -v scontrol >/dev/null 2>&1; then
        scontrol show hostnames "$nl" | head -n1
        return 0
    fi
    if [[ "$nl" == *"["* ]]; then
        local prefix="${nl%%[*}"                      # e.g. funk
        local range="${nl#*[}"; range="${range%%]*}"  # e.g. 06,20 or 06-08
        printf '%s%s\n' "$prefix" "${range%%[,-]*}"    # e.g. funk06
    else
        printf '%s\n' "${nl%%,*}"                      # e.g. funk06
    fi
}

MASTER_ADDR="$(first_slurm_node || hostname -s)"
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
