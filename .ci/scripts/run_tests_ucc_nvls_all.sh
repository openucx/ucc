#!/bin/bash -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

if [ "${SLURM_LOCALID:-0}" = "0" ]; then
    "${SCRIPT_DIR}/check_nvls_fabric.sh"
fi

export OMPI_MCA_coll=^hcoll
export OMPI_MCA_coll_ucc_enable=0
export UCC_LOG_LEVEL=info
export UCC_TL_CUDA_NVLS_SM_COUNT=4
export UCC_TLS=cuda,ucp

PERFTEST=/opt/nvidia/bin/ucc/build/bin/ucc_perftest

echo "INFO: NVLS perftest (allreduce) ..."
UCC_TL_CUDA_TUNE=allreduce:cuda:@0 $PERFTEST -c allreduce -F -m cuda -b 1k -e 32M -d bfloat16 -o sum
echo "INFO: NVLS perftest (allreduce) ... DONE"

# Disabled: reduce_scatter NVLS is tested via MPI tests instead.
#echo "INFO: NVLS perftest (reduce_scatter) ..."
#UCC_TL_CUDA_TUNE=reduce_scatter:cuda:@3 $PERFTEST -c reduce_scatter -F -m cuda -b 1k -e 32M -d bfloat16 -o sum
#echo "INFO: NVLS perftest (reduce_scatter) ... DONE"
