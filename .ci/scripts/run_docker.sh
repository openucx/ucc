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

DOCKER_CONTAINER_NAME="torch_ucc"
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
--user root \
-it \
-d \
--rm \
--name=${DOCKER_CONTAINER_NAME} \
-v /labhome:/labhome \
-v /root/.ssh:/root/.ssh \
"

# shellcheck disable=SC2013
for HOST in $(cat "$HOSTFILE"); do
    echo "INFO: HOST = $HOST"

    STALE_DOCKER_CONTAINER_LIST=$(ssh -n "$HOST" "docker ps -a -q -f name=${DOCKER_CONTAINER_NAME}")
    if [ -n "${STALE_DOCKER_CONTAINER_LIST}" ]; then
        echo "WARNING: stale docker container (name: ${DOCKER_CONTAINER_NAME}) is detected on ${HOST} (to be stopped)"
        echo "INFO: Stopping stale docker container (name: ${DOCKER_CONTAINER_NAME}) on ${HOST}..."
        ssh "${HOST}" docker stop ${DOCKER_CONTAINER_NAME}
        echo "INFO: Stopping stale docker container (name: ${DOCKER_CONTAINER_NAME}) on ${HOST}... DONE"
    fi
done

# shellcheck disable=SC2002
HOST_LIST="$(cat "$HOSTFILE" | xargs hostlist)"

pdsh -w "${HOST_LIST}" -R ssh hostname

echo "INFO: clean up docker artefacts on ..."
pdsh -w "${HOST_LIST}" -R ssh docker system prune --all --volumes --force
echo "INFO: clean up docker artefacts on ... DONE"

pdsh -w "${HOST_LIST}" -R ssh docker pull "${DOCKER_IMAGE_NAME}"

# shellcheck disable=SC2013
for HOST in $(cat "$HOSTFILE"); do
    echo "INFO: start docker container on $HOST ..."
    # shellcheck disable=SC2029
    sudo ssh "$HOST" "docker run \
        ${DOCKER_RUN_ARGS} \
        ${DOCKER_IMAGE_NAME} \
        bash -c '/usr/sbin/sshd -p ${DOCKER_SSH_PORT}; sleep infinity'"
    echo "INFO: start docker container on $HOST ... DONE"

    sleep 5

    echo "INFO: verify docker container on $HOST ..."
    sudo ssh -p "${DOCKER_SSH_PORT}" "$HOST" hostname
    sudo ssh -p "${DOCKER_SSH_PORT}" "$HOST" cat /proc/1/cgroup
    echo "INFO: verify docker container on $HOST ... DONE"
done
