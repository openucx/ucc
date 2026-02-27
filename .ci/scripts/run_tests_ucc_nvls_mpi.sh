#!/bin/bash -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

export OMPI_MCA_coll=^hcoll
export OMPI_MCA_coll_ucc_enable=0
export UCC_TLS=cuda,ucp
export UCC_LOG_LEVEL=info
export UCC_TL_CUDA_NVLS_SM_COUNT=4

EXE="/opt/nvidia/src/ucc/build/test/mpi/ucc_test_mpi"
EXE+=" --set_device 2 --mtypes cuda"

echo "INFO: NVLS MPI tests (allreduce) ..."
UCC_TL_CUDA_TUNE="allreduce:cuda:@0" $EXE -c allreduce -d float32 -o sum -m 1024:33554432
echo "INFO: NVLS MPI tests (allreduce) ... DONE"

# echo "INFO: NVLS MPI tests (reduce_scatter) ..."
# UCC_TL_CUDA_TUNE="reduce_scatter:cuda:@3" $EXE -c reduce_scatter -d float32 -o sum -m 1024:33554432
# echo "INFO: NVLS MPI tests (reduce_scatter) ... DONE"
