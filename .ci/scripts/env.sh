#!/bin/bash -eEx

export PATH="/opt/hpcx/ompi/bin:$PATH"
export LD_LIBRARY_PATH="/opt/hpcx/ompi/lib:${LD_LIBRARY_PATH}"
export OPAL_PREFIX=/opt/hpcx/ompi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

# shellcheck disable=SC2034
#DLRM_MODEL="big"
DLRM_MODEL="small"

HOSTNAME=$(hostname -s)
export HOSTNAME
SRC_ROOT_DIR=$(cd "${SCRIPT_DIR}/../../" && pwd -P)
export CONFIGS_DIR="${SRC_ROOT_DIR}/.ci/configs"

# DLRM MASTER_PORT
export MASTER_PORT="12346"
export DOCKER_SSH_PORT="12345"

export SSH_CMD="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null"
