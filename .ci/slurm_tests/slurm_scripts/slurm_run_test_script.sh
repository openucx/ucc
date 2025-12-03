#!/bin/bash
set -xvEe -o pipefail

test_script="${1}"
: "${test_script:?test_script is not set}"

: "${SLM_JOB_NAME:?SLM_JOB_NAME is not set}"
: "${SLM_HEAD_NODE:?SLM_HEAD_NODE is not set}"
: "${UCC_ENROOT_IMAGE_NAME:?UCC_ENROOT_IMAGE_NAME is not set}"

if [ "${SLM_HEAD_NODE}" == "scctl" ]; then
      slurm_test_script="${WORKSPACE}"/slurm_test.sh
      envsubst < "${test_script}" > "${slurm_test_script}"
      cat  "${slurm_test_script}"
      scctl --raw-errors client connect -s "${slurm_test_script}"
else
    : # TODO: implement ssh script execution, use heredoc to eliminate need for envsubst
fi
