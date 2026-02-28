# shellcheck shell=bash

function trap_error() {
    local lineno=$1
    local msg=$2
    echo "Error at line $lineno: $msg"
    if [ "${DEBUG}" == "9" ]; then
        echo "Debug mode, sleeping for 3600 seconds to allow for debugging of the pod"
        sleep 3600
    fi
    exit 1
}

trap 'trap_error $LINENO "Error in script"' ERR

export ssh_cmd="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -l svcnbu-swx-hpcx"
