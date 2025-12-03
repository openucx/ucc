#!/bin/bash
set -xvEe -o pipefail

: "${SLM_JOB_NAME:?SLM_JOB_NAME is not set}"

slurm_headnode_list="scctl hpchead"
for slurm_headnode in ${slurm_headnode_list}; do
    slurm_cmd="scancel --name=${SLM_JOB_NAME}"
    if [ "${slurm_headnode}" == "scctl" ]; then
        export RANCHER_USER=${SERVICE_USER_USERNAME}
        export RANCHER_PASSWORD=${SERVICE_USER_PASSWORD}

        scctl -v
        scctl --raw-errors login
        result=$(scctl --raw-errors client exists)
        if [ "$result" == "client does not exist" ]; then
            scctl --raw-errors client create
        fi
        scctl --raw-errors client connect "${slurm_cmd}"
    else
        ssh "${slurm_headnode}" "${slurm_cmd}"
    fi
done
