#!/bin/bash -eEx
set -o pipefail

UCC_SRC_DIR="${SRC_DIR}/ucc"
cd "${UCC_SRC_DIR}/build"

export UCX_WARN_UNUSED_ENV_VARS=n
# Disable NCCL
export UCC_TL_NCCL_TUNE=0

# CI environment: Override UCC_HANDLE_ERRORS to prevent hanging on errors
# In CI we want tests to fail fast, not freeze for debugging
export UCC_HANDLE_ERRORS=${UCC_HANDLE_ERRORS:-bt}

# ==========================================
# Debug configuration for troubleshooting
# ==========================================
# Enable debug mode by setting DEBUG_UCC=1 in environment
if [ "${DEBUG_UCC:-0}" = "1" ]; then
    echo "=== DEBUG MODE ENABLED ==="

    # UCC logging - set to debug/trace for verbose output
    export UCC_LOG_LEVEL=${UCC_LOG_LEVEL:-debug}

    # UCC collective trace - shows collective operations
    export UCC_COLL_TRACE=${UCC_COLL_TRACE:-debug}

    # UCX logging - helpful for communication issues
    export UCX_LOG_LEVEL=${UCX_LOG_LEVEL:-info}

    # CUDA debugging (if GPU is involved)
    export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-1}
    export CUDA_DEVICE_MAX_CONNECTIONS=${CUDA_DEVICE_MAX_CONNECTIONS:-1}

    # UCP transport layer debugging
    export UCC_TL_UCP_LOG_LEVEL=${UCC_TL_UCP_LOG_LEVEL:-debug}

    # Show which GPUs are being used
    if command -v nvidia-smi &> /dev/null; then
        echo "=== GPU Information ==="
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv
    fi

    echo "=== Environment Variables ==="
    env | grep -E "^(UCC|UCX|CUDA|NCCL)" | sort
    echo "=========================="
fi

# Even in normal mode, print test times to help identify hangs
export GTEST_EXTRA_ARGS="--gtest_print_time=1 ${GTEST_EXTRA_ARGS:-}"

# ==========================================
# Potential hang mitigations
# ==========================================
# These can be uncommented to work around specific issues:

# Option 1: Reduce preconnect threshold (helps with team creation deadlocks)
# export UCC_TL_UCP_PRECONNECT=${UCC_TL_UCP_PRECONNECT:-8}

# Option 2: Disable async progress (may help with progress engine issues)
# export UCX_USE_MT_MUTEX=${UCX_USE_MT_MUTEX:-n}

# Option 3: Force GPU memory pinning (can help with GPU-related hangs)
# export UCX_IB_GPU_DIRECT_RDMA=${UCX_IB_GPU_DIRECT_RDMA:-no}

# Option 4: Limit CUDA visible devices to avoid multi-GPU issues
# export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

# Option 5: Increase UCX request timeouts
# export UCX_UNIFIED_MODE=${UCX_UNIFIED_MODE:-y}


make gtest
