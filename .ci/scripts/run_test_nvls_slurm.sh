#! /bin/bash

# NOTE: script is preprocessed by envsubst
# ensure all variables to be set are in stand alone and simple format
# complex bash string operations are not supported in envsubst

srun \
    --jobid=${SLM_JOB_ID} \
    --nodes=${SLM_NODES} \
    --ntasks-per-node=1 \
    --gpus-per-node=1 \
    --mpi=pmix \
	--cpu-bind=verbose \
    --container-image=${UCC_ENROOT_IMAGE_NAME} \
       bash -l /opt/nvidia/src/ucc/.ci/scripts/run_test_nvls.sh
