#!/bin/bash -eEx
set -o pipefail

UCC_SRC_DIR="${SRC_DIR}/ucc"
cd "${UCC_SRC_DIR}/build"

export UCX_WARN_UNUSED_ENV_VARS=n
# Disable NCCL
export UCC_TL_NCCL_TUNE=0

make gtest
