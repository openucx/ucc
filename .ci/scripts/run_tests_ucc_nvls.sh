#!/bin/bash -eEx


SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/env.sh"

export OMPI_MCA_coll=^hcoll
export OMPI_MCA_coll_ucc_enable=0
export UCC_TLS=cuda,ucp
export UCC_LOG_LEVEL=info
export UCC_TL_CUDA_NVLS_SM_COUNT=20
export UCC_TL_CUDA_TUNE=allreduce:cuda:@0

/opt/nvidia/bin/ucc/build/bin/ucc_perftest -c allreduce -F -m cuda -b 1k -e 32M -d bfloat16 -o sum
