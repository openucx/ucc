#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/env.sh"

export HOSTFILE=${HOSTFILE:-${CONFIGS_DIR}/$HOSTNAME/hostfile.txt}

if [ ! -f "${HOSTFILE}" ]; then
    echo "ERROR: ${HOSTFILE} does not exist"
    exit 1
fi

HEAD_NODE=$(head -1 "$HOSTFILE")
export HEAD_NODE

ssh -p "${DOCKER_SSH_PORT}" "${HEAD_NODE}" /opt/nvidia/src/ucc/.ci/scripts/run_tests_ucc_mpi.sh "/opt/nvidia/src/ucc/.ci/configs/${HEAD_NODE}/hostfile.txt"
