#!/bin/bash
set -xvEe -o pipefail

# NOTE: script is preprocessed by envsubst
# ensure all variables to be set are in stand alone and simple format
# complex bash string operations are not supported in envsubst

# Note2: docker image name format should be converted to enroot format (replace first / with #)
# Example: harbor.mellanox.com/torch-ucc/ucc/1.0.0/x86_64/centos8/cuda12.9 -> harbor.mellanox.com#torch-ucc/ucc/1.0.0/x86_64/centos8/cuda12.9

srun --job-name=${SLM_JOB_NAME} --nodes=${SLM_NODES} --partition=${SLM_PARTITION} \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --mpi=pmix \
	--cpu-bind=verbose \
    --container-image=$(echo "${UCC_DOCKER_IMAGE_NAME}" |sed 's/\//#/'):${BUILD_NUMBER} \
     bash -lc '
          OMPI_MCA_coll=^hcoll \
          OMPI_MCA_coll_ucc_enable=0 \
          UCC_TLS=cuda,ucp UCC_LOG_LEVEL=info UCC_TL_CUDA_NVLS_SM_COUNT=20 UCC_TL_CUDA_TUNE=allreduce:cuda:@0 \
          /opt/nvidia/bin/ucc/build/bin/ucc_perftest -c allreduce -F -m cuda -b 1k -e 32M -d bfloat16 -o sum \
        '