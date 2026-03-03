#!/bin/bash -eEx
set -o pipefail

UCC_SRC_DIR="${SRC_DIR}/ucc"
cd "${UCC_SRC_DIR}/build"

export UCX_WARN_UNUSED_ENV_VARS=n
# Disable NCCL
export UCC_TL_NCCL_TUNE=0

UCC_GTEST_SHARDS=${UCC_GTEST_SHARDS:-1}

if [ "${UCC_GTEST_SHARDS}" -le 1 ]; then
    make gtest
else
    num_gpus=$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)

    pids=""
    for i in $(seq 0 $((UCC_GTEST_SHARDS - 1))); do
        shard_env="GTEST_TOTAL_SHARDS=${UCC_GTEST_SHARDS} GTEST_SHARD_INDEX=${i}"
        if [ "$num_gpus" -gt 1 ]; then
            gpu_id=$((i % num_gpus))
            shard_env="CUDA_VISIBLE_DEVICES=${gpu_id} ${shard_env}"
        fi
        env $shard_env make gtest 2>&1 | sed -u "s/^/[shard ${i}] /" &
        pids="$pids $!"
    done
    failed=0
    for pid in $pids; do
        wait "$pid" || failed=1
    done
    if [ "$failed" -ne 0 ]; then
        echo "ERROR: One or more gtest shards failed"
        exit 1
    fi
fi
