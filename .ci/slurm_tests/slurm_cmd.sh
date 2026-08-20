#!/bin/bash
set -xvEe -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

script_cmd="${1}"
slurm_via=${SLURM_VIA:-"scctl"}
slurm_cmd=${SLURM_CMD}

echo "INFO: Allocate Slurm resources"

if [ "${slurm_via}" == "scctl" ]; then
    if [ "${script_cmd}" == "init" ]; then
        export RANCHER_USER=${SERVICE_USER_USERNAME}
        export RANCHER_PASSWORD=${SERVICE_USER_PASSWORD}

        scctl -v
        scctl --raw-errors login
        result=$(scctl --raw-errors client exists)
        if [ "$result" == "client does not exist" ]; then
            scctl --raw-errors client create
        fi
        scctl --raw-errors client connect -s "${SCRIPT_DIR}/fix_enroot.sh"
    elif [ "${script_cmd}" == "exec" ]; then
        scctl --raw-errors client connect "${slurm_cmd}"
    elif [ "${script_cmd}" == "exec_file" ]; then
        scctl --raw-errors client connect -s "${slurm_cmd}"
    else
        echo "ERROR: invalid script command: ${script_cmd}"
        exit 1
    fi
elif [ "${slurm_via}" == "ssh" ]; then
    if [ "${script_cmd}" == "init" ]; then
        : # TODO: implement ssh allocation, run fix_enroot.sh
    elif [ "${script_cmd}" == "exec" ]; then
        : # TODO: implement ssh allocation, run slurm_cmd
    else
        echo "ERROR: invalid script command: ${script_cmd}"
        exit 1
    fi
else
    echo "ERROR: invalid allocation via: ${slurm_via}"
    exit 1
fi