#!/bin/bash -eEx

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
