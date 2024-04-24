source run/env.sh

nnodes=1; ppn=8; mpirun -np $((nnodes*ppn)) -x UCC_TL_NCCL_TUNE=0 -x UCC_MC_CUDA_MPOOL_MAX_ELEMS=128 -x UCC_MC_CUDA_MPOOL_ELEM_SIZE=1048576 --bind-to core ./build-original/bin/ucc_perftest -c alltoallv -b 128 -e 1048576
