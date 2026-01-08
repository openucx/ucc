#!/bin/bash
set -x

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

if [[ -z ${SLURM_JOB_ID} ]]; then
    echo "ERROR: SLURM_JOB_ID is not set, trying to get it from ${WORKSPACE}/job_id.txt"
    if [[ -f ${WORKSPACE}/job_id.txt ]]; then
        SLURM_JOB_ID=$(cat ${WORKSPACE}/job_id.txt)
    else
        echo "ERROR: ${WORKSPACE}/job_id.txt does not exist"
        exit 1
    fi
fi
[[ -z ${SLURM_HEAD_NODE} ]] && { echo "ERROR: SLURM_HEAD_NODE is not set"; exit 1; }

readonly SLURM_STOP_ALLOCATION_CMD="scancel ${SLURM_JOB_ID}"

if [ "${SLURM_HEAD_NODE}" == "scctl" ]; then
    echo "INFO: Using scctl client to stop Slurm resources"
    scctl --raw-errors client connect -- "${SLURM_STOP_ALLOCATION_CMD}"
else
    echo "INFO: Connecting to SLURM head node via SSH: ${SLURM_HEAD_NODE}"
    eval "${SSH_CMD} ${SLURM_HEAD_NODE} ${SLURM_STOP_ALLOCATION_CMD}"
fi
