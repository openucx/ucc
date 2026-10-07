#!/bin/bash
set -xvEe -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/common.sh"

test_script="${1}"
: "${test_script:?test_script is not set}"

: "${SLM_JOB_NAME:?SLM_JOB_NAME is not set}"
: "${SLM_HEAD_NODE:?SLM_HEAD_NODE is not set}"
: "${UCC_ENROOT_IMAGE_NAME:?UCC_ENROOT_IMAGE_NAME is not set}"

if [ "${SLM_HEAD_NODE}" == "scctl" ]; then
      slurm_test_script="${WORKSPACE}"/slurm_test.sh
      SLM_JOB_ID=$(scctl client connect "squeue --noheader --name=${SLM_JOB_NAME} -o '%i'")
      export SLM_JOB_ID
      envsubst < "${test_script}" > "${slurm_test_script}"
      cat  "${slurm_test_script}"
      scctl --raw-errors client connect -s "${slurm_test_script}"
else
    SLM_JOB_ID=$(${ssh_cmd} "${SLM_HEAD_NODE}" "squeue --noheader --name=${SLM_JOB_NAME} -o '%i'")
    export SLM_JOB_ID
#shellcheck disable=SC2087
${ssh_cmd} "${SLM_HEAD_NODE}" << EOF
$(envsubst < "${test_script}")
EOF

fi
