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

# shellcheck disable=SC2002
HOST_LIST="$(cat "$HOSTFILE" | xargs)"

echo "INFO: remove docker container on ..."
pdsh -w "${HOST_LIST}" -R ssh docker ps -a -q -f "name=torch_ucc_${BUILD_NUMBER}" -f "name=ucc_tests_${BUILD_NUMBER}" | xargs -r docker rm -f
echo "INFO: remove docker container on ... DONE"

echo "INFO: remove docker image on ..."
pdsh -w "${HOST_LIST}" -R ssh docker rmi -f "${UCC_DOCKER_IMAGE_NAME}:${BUILD_NUMBER}"
echo "INFO: remove docker image on ... DONE"
