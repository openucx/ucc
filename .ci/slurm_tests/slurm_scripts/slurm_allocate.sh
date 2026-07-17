#!/bin/bash
set -xvEe -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/common.sh"

: "${SLM_JOB_NAME:?SLM_JOB_NAME is not set}"
: "${SLM_PARTITION:?SLM_PARTITION is not set}"
: "${SLM_NODES:?SLM_NODES is not set}"
: "${SLM_HEAD_NODE:?SLM_HEAD_NODE is not set}"

slurm_cmd="salloc -N ${SLM_NODES} -p ${SLM_PARTITION} --job-name=${SLM_JOB_NAME} --immediate=120 --time=00:30:00 --no-shell"

if [ "${SLM_HEAD_NODE}" == "scctl" ]; then
    export RANCHER_USER=${SERVICE_USER_USERNAME}
    export RANCHER_PASSWORD=${SERVICE_USER_PASSWORD}

    scctl -v
    scctl --raw-errors login
    result=$(scctl --raw-errors client exists)
    if [ "$result" == "client does not exist" ]; then
        scctl --raw-errors client create
    fi
    scctl --raw-errors client connect -s "${SCRIPT_DIR}/fix_enroot.sh"
    scctl --raw-errors client connect "${slurm_cmd}"
else
    "${SCRIPT_DIR}/fix_ssh_key.sh"
    ${ssh_cmd} "${SLM_HEAD_NODE}" "${slurm_cmd}"
fi

echo "INFO: Allocate Slurm resources"
