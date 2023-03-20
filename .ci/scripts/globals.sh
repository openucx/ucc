#!/bin/bash
#
# Copyright (c) 2001-2018 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# See file LICENSE for terms.
#

WORKSPACE=${WORKSPACE:=$PWD}
if [ -z "$BUILD_NUMBER" ]; then
    echo Running interactive
    BUILD_NUMBER=1
    WS_URL=file://$WORKSPACE
    JENKINS_RUN_TESTS=yes
else
    echo Running under jenkins
    WS_URL=$JOB_URL/ws
fi

export mpi_module="hpcx-ga-gcc"
export ics_module="intel/ics-15.0.1"
export SD_PORT=${SD_PORT:="6127"}
export AM_PORT=${AM_PORT:="6126"}
export LOCKFILE=/tmp/sharp_jenkins.lock
export SD_PID_FILE=${SD_PID_FILE:-/var/run/sharpd.pid}
export AM_PID_FILE=${AM_PID_FILE:-/var/run/sharp_am.pid}
export SMX_TESTER_PID_FILE=${AM_PID_FILE:-/var/run/sharp_smx_test.pid}
export IB_DEV="mlx5_0:1"
# The env variable above is for sharp_mpi_test
# A variable for SD (if necessary) should be defined separately
# If defining one - be aware that 'sharpd_start' cmd runs SD with sudo,
# which will affect the env variables

#export ENABLE_SHARP_COLL=1
export MAX_PAYLOAD_SIZE=256
export SHARP_COLL_LOG_LEVEL=3

export prefix=jenkins

nproc=$(grep processor /proc/cpuinfo|wc -l)
make_opt="-j$(($nproc / 2 + 1))"


#
# Test if an environment module exists and load it if yes.
# Otherwise, return error code.
#
module_load() {
	set +x

	local module=$1

	if [ -n "$(module avail $module 2>&1)" ]
	then
		module load $module
		set -x
		return 0
	else
		set -x
		return 1
	fi
}


function do_github_status()
{
    echo "Calling: github $1"
    eval "local $1"

    local token=""
    if [ -z "$tokenfile" ]; then
        tokenfile="$HOME/.mellanox-github"
    fi

    if [ -r "$tokenfile" ]; then
        token="$(cat $tokenfile)"
    else
        echo Error: Unable to read tokenfile: $tokenfile
        return
    fi

    curl \
    -X POST \
    -H "Content-Type: application/json" \
    -d "{\"state\": \"$state\", \"context\": \"$context\",\"description\": \"$info\", \"target_url\": \"$target_url\"}" \
    "https://api.github.com/repos/$repo/statuses/${sha1}?access_token=$token"
}

function check_filter()
{
    local msg=$1
    local filter=$2

    if [ -n "$filter" -a "$filter" == "on" ]; then
        if [ -z "$JENKINS_RUN_TESTS" -o "$JENKINS_RUN_TESTS" == "no" ]; then
            echo "$msg [SKIP]"
            exit 0
        fi
    fi

    echo "$msg [OK]"
}

