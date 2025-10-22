#!/bin/bash -eEx
set -o pipefail

function err_report () {
    echo "Exited with ERROR in line $1"
    exit 1
}
trap 'err_report $LINENO' ERR

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
HOSTS=$(cat "$HOSTFILE" | xargs | tr ' ' ',')
export HOSTS
HEAD_NODE=$(head -1 "$HOSTFILE")
export HEAD_NODE

DOCKER_CONTAINER_NAME="torch_ucc_${BUILD_ID}"
DOCKER_IMAGE_NAME="${UCC_DOCKER_IMAGE_NAME}:${BUILD_ID}"

DOCKER_RUN_ARGS="\
--pull always \
--network=host \
--uts=host \
--ipc=host \
--ulimit stack=67108864 \
--ulimit memlock=-1 \
--security-opt seccomp=unconfined \
--cap-add=SYS_ADMIN \
--device=/dev/infiniband/ \
--gpus all \
-it \
-d \
--rm \
--name=${DOCKER_CONTAINER_NAME} \
-v /labhome/swx-jenkins:/labhome/swx-jenkins \
"

# shellcheck disable=SC2013
for HOST in $(cat "$HOSTFILE"); do
    echo "INFO: HOST = $HOST"

    STALE_DOCKER_CONTAINER_LIST=$(ssh -n "$HOST" "docker ps -a -q -f name=${DOCKER_CONTAINER_NAME}")
    if [ -n "${STALE_DOCKER_CONTAINER_LIST}" ]; then
        echo "WARNING: stale docker container (name: ${DOCKER_CONTAINER_NAME}) is detected on ${HOST} (to be stopped)"
        echo "INFO: Stopping stale docker container (name: ${DOCKER_CONTAINER_NAME}) on ${HOST}..."
        ssh "${HOST}" docker stop "$STALE_DOCKER_CONTAINER_LIST"
        echo "INFO: Stopping stale docker container (name: ${DOCKER_CONTAINER_NAME}) on ${HOST}... DONE"
    fi
done

# shellcheck disable=SC2002
HOST_LIST="$(cat "$HOSTFILE" | xargs hostlist)"

pdsh -w "${HOST_LIST}" -R ssh hostname

pdsh -w "${HOST_LIST}" -R ssh docker pull "${DOCKER_IMAGE_NAME}"

# shellcheck disable=SC2013
for HOST in $(cat "$HOSTFILE"); do
    echo "INFO: start docker container on $HOST ..."
    # shellcheck disable=SC2029
    ssh "$HOST" "docker run \
        ${DOCKER_RUN_ARGS} \
        ${DOCKER_IMAGE_NAME} \
        sudo /usr/sbin/sshd -D -p ${DOCKER_SSH_PORT}"
    echo "INFO: start docker container on $HOST ... DONE"

    sleep 15

    echo "INFO: verify docker container on $HOST ..."
    ssh -p "${DOCKER_SSH_PORT}" "$HOST" hostname
    ssh -p "${DOCKER_SSH_PORT}" "$HOST" cat /proc/1/cgroup
    echo "INFO: verify docker container on $HOST ... DONE"
done
