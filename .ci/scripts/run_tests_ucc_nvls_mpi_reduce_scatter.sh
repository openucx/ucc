#!/bin/bash -xe
# NVLS reduce_scatter only. Run as a separate srun step (separate MPI job).
# -m 1024:33554432:4 keeps per-rank counts NVLS-aligned.

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

if [ "${SLURM_LOCALID:-0}" = "0" ]; then
    "${SCRIPT_DIR}/check_nvls_fabric.sh"
fi

export OMPI_MCA_coll=^hcoll
export OMPI_MCA_coll_ucc_enable=0
export UCC_TLS=cuda,ucp
export UCC_LOG_LEVEL=info
export UCC_TL_CUDA_NVLS_SM_COUNT=4

EXE="/opt/nvidia/src/ucc/build/test/mpi/ucc_test_mpi"
EXE+=" --set_device 2 --mtypes cuda"
DTYPES="float32,int32,uint32,int64,uint64"

echo "INFO: NVLS MPI tests (reduce_scatter) ..."
UCC_TL_CUDA_TUNE="reduce_scatter:cuda:@3" $EXE -c reduce_scatter -d ${DTYPES} -o sum -m 1024:33554432:4
echo "INFO: NVLS MPI tests (reduce_scatter) ... DONE"
