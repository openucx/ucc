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

DOCKER_CONTAINER_NAME="torch_ucc_${BUILD_NUMBER}"

# shellcheck disable=SC2002
HOST_LIST="$(cat "$HOSTFILE" | xargs hostlist)"

echo "INFO: stop docker container on ..."
pdsh -w "${HOST_LIST}" -R ssh docker stop "${DOCKER_CONTAINER_NAME}"
echo "INFO: stop docker container on ... DONE"
