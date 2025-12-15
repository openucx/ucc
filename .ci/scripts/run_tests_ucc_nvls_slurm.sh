#!/bin/bash -eEx


SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/env.sh"

# slurm_command_prefix="scctl client connect --"
scctl client connect -- srun --jobid=${JOB_ID} --nodes=2 --mpi=pmix --cpu-bind=verbose --ntasks-per-node=1 --gpus-per-node=1 --container-image=${DOCKER_IMAGE_NAME} /opt/nvidia/src/ucc/.ci/scripts/run_tests_ucc_nvls.sh
