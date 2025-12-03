#!/bin/bash
set -xvEe -o pipefail

# NOTE: script is preprocessed by envsubst
# ensure all variables to be set are in stand alone and simple format
# complex bash string operations are not supported in envsubst

srun --job-name=${SLM_JOB_NAME} \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --mpi=pmix \
	--cpu-bind=verbose \
    --container-image=${UCC_ENROOT_IMAGE_NAME} \
     bash -lc '
        OMPI_MCA_coll=^hcoll \
        OMPI_MCA_coll_ucc_enable=0 \
        UCC_TLS=cuda,ucp UCC_LOG_LEVEL=info UCC_TL_CUDA_NVLS_SM_COUNT=20 UCC_TL_CUDA_TUNE=allreduce:cuda:@0 \
        /opt/nvidia/bin/ucc/build/bin/ucc_perftest -c allreduce -F -m cuda -b 1k -e 32M -d bfloat16 -o sum \
     '
