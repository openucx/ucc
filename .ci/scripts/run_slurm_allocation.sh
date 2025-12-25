#!/bin/bash
set -xe

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

[[ -z ${SLURM_JOB_NAME} ]] && { echo "ERROR: SLURM_JOB_NAME is not set"; exit 1; }
[[ -z ${SLURM_PARTITION} ]] && { echo "ERROR: SLURM_PARTITION is not set"; exit 1; }
[[ -z ${SLURM_NODES} ]] && { echo "ERROR: SLURM_NODES is not set"; exit 1; }
[[ -z ${SLURM_HEAD_NODE} ]] && { echo "ERROR: SLURM_HEAD_NODE is not set"; exit 1; }
[[ -z ${SLURM_JOB_TIMEOUT} ]] && { echo "ERROR: SLURM_JOB_TIMEOUT is not set"; exit 1; }

readonly SLURM_IMMEDIATE_TIMEOUT=${SLURM_IMMEDIATE_TIMEOUT:-600} # time to wait for resource allocation to be granted
readonly SLURM_ACCOUNT=${SLURM_ACCOUNT:+"--account=${SLURM_ACCOUNT}"}
readonly SLURM_ALLOCATION_CMD="salloc ${SLURM_ACCOUNT} -N ${SLURM_NODES} -p ${SLURM_PARTITION} --job-name=${SLURM_JOB_NAME} --immediate=${SLURM_IMMEDIATE_TIMEOUT} --time=${SLURM_JOB_TIMEOUT} --no-shell"
readonly SLURM_GET_JOB_ID_CMD="squeue --noheader --name=${SLURM_JOB_NAME} --format=%i"

case "${SLURM_HEAD_NODE}" in
    scctl)
        echo "INFO: Using scctl client to connect and allocate Slurm resources"
        export SCCTL_USER=${SERVICE_USER_USERNAME}
        export SCCTL_PASSWORD=${SERVICE_USER_PASSWORD}
        scctl -v
        scctl --raw-errors upgrade
        scctl --raw-errors login
        result=$(scctl --raw-errors client exists)
        if [ "$result" == "client does not exist" ]; then
            echo "INFO: Creating scctl client"
            scctl --raw-errors client create
        fi
        echo "INFO: Setting enroot credentials"
        "${SCRIPT_DIR}/enroot_setup.sh"
        echo "INFO: Allocating Slurm resources via scctl"
        scctl --raw-errors client connect -- "${SLURM_ALLOCATION_CMD}"
        JOB_ID=$(scctl --raw-errors client connect -- "${SLURM_GET_JOB_ID_CMD}")
        ;;
    dlcluster*)
        echo "INFO: Connecting to SLURM head node via SSH: ${SLURM_HEAD_NODE}"
        echo "INFO: Setting enroot credentials"
        eval "${SSH_CMD} ${SLURM_HEAD_NODE} 'env ENROOT_USERNAME=${ENROOT_USERNAME} ENROOT_PASSWORD=${ENROOT_PASSWORD} ENROOT_REGISTRY=${ENROOT_REGISTRY} bash -s' < $SCRIPT_DIR/enroot_setup.sh"
        echo "INFO: Allocating Slurm resources via SSH"
        SALLOC_OUTPUT=$(eval "${SSH_CMD} ${SLURM_HEAD_NODE} ${SLURM_ALLOCATION_CMD}" 2>&1)
        echo "${SALLOC_OUTPUT}"
        # Extract job ID from salloc output (looks for "Granted job allocation XXXXXX")
        # seems like on dlcluster the squeue method does not return the job ID, so we need to extract it from the output
        JOB_ID=$(echo "${SALLOC_OUTPUT}" | grep -oP "Granted job allocation \K[0-9]+")
        ;;
    "")
        echo "ERROR: Invalid SLURM_HEAD_NODE value: ${SLURM_HEAD_NODE}"
        exit 1
        ;;
    *)
        echo "INFO: Setting enroot credentials"
        "${SCRIPT_DIR}/enroot_setup.sh"
        echo "INFO: Connecting to SLURM head node via SSH: ${SLURM_HEAD_NODE}"
        echo "INFO: Allocating Slurm resources via SSH"
        eval "${SSH_CMD} ${SLURM_HEAD_NODE} ${SLURM_ALLOCATION_CMD}"
        JOB_ID=$(eval "${SSH_CMD} ${SLURM_HEAD_NODE} ${SLURM_GET_JOB_ID_CMD}")
        ;;
esac

[[ -z $JOB_ID ]] && { echo "ERROR: Failed to get job ID"; exit 1; }

echo "INFO: Job ID: ${JOB_ID} to be saved in ${WORKSPACE}/job_id.txt"
echo "${JOB_ID}" > "${WORKSPACE}/job_id.txt"
echo "INFO: Allocate Slurm resources"
