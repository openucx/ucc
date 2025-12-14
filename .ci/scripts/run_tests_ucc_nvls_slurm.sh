#!/bin/bash -xe


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

readonly SLURM_COMMAND="srun --jobid=${SLURM_JOB_ID} --nodes=${SLURM_NODES} --mpi=pmix --ntasks-per-node=1 --container-image=${DOCKER_IMAGE_NAME} '/opt/nvidia/src/ucc/.ci/scripts/run_tests_ucc_nvls.sh'"

# Validate SLURM_HEAD_NODE is set
if [ -z "${SLURM_HEAD_NODE}" ]; then
    echo "ERROR: SLURM_HEAD_NODE is not set or empty"
    exit 1
fi

# Execute based on head node type
case "${SLURM_HEAD_NODE}" in
    scctl)
        echo "Using scctl client to connect and execute slurm command"
        scctl client connect -- "${SLURM_COMMAND}"
        ;;
    "")
        echo "ERROR: Invalid SLURM_HEAD_NODE value: ${SLURM_HEAD_NODE}"
        exit 1
        ;;
    *)
        echo "Connecting to SLURM head node: ${SLURM_HEAD_NODE}"
        eval "${SSH_CMD} ${SLURM_HEAD_NODE} ${SLURM_COMMAND}"
        ;;
esac
