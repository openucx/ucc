#!/bin/bash -xe

# Generic NVLS Slurm test runner.
# Usage: run_nvls_slurm.sh <container_script> [ntasks_per_node]

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

CONTAINER_SCRIPT=${1:?"Usage: run_nvls_slurm.sh <container_script> [ntasks_per_node]"}
NTASKS_PER_NODE=${2:-1}

readonly SLURM_COMMAND="srun --jobid=${SLURM_JOB_ID} --nodes=${SLURM_NODES} --mpi=pmix --ntasks-per-node=${NTASKS_PER_NODE} --container-image=${DOCKER_IMAGE_NAME} '${CONTAINER_SCRIPT}'"

if [ -z "${SLURM_HEAD_NODE}" ]; then
    echo "ERROR: SLURM_HEAD_NODE is not set or empty"
    exit 1
fi

case "${SLURM_HEAD_NODE}" in
    scctl)
        echo "Using scctl client to connect and execute slurm command"
        scctl client connect -- "${SLURM_COMMAND}"
        ;;
    dlcluster*)
        echo "Connecting to SLURM head node: ${SLURM_HEAD_NODE}"
        SLURM_COMMAND_ESCAPED="${SLURM_COMMAND//\#/\\#}"
        eval "${SSH_CMD} ${SLURM_HEAD_NODE} \"${SLURM_COMMAND_ESCAPED}\""
        ;;
    *)
        echo "Connecting to SLURM head node: ${SLURM_HEAD_NODE}"
        eval "${SSH_CMD} ${SLURM_HEAD_NODE} \"${SLURM_COMMAND}\""
        ;;
esac
