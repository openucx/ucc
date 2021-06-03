#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR="$(
    cd "$(dirname "$0")"
    pwd -P
)"
cd "${SCRIPT_DIR}"
. "${SCRIPT_DIR}/env.sh"

HOSTFILE="$1"

if [ -z "$HOSTFILE" ]; then
    echo "ERROR: HOSTFILE is not specified"
    exit 1
fi

export PATH="/usr/lib64/openmpi/bin:$PATH"
export LD_LIBRARY_PATH="/usr/lib64/openmpi/lib:${LD_LIBRARY_PATH}"

HEAD_NODE=$(head -1 "$HOSTFILE")
export HEAD_NODE
export MASTER_ADDR=${HEAD_NODE}

NP=$(wc --lines "$HOSTFILE" | awk '{print $1}')

# shellcheck disable=SC2086
mpirun \
    -np $NP \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    hostname

# shellcheck disable=SC2086
mpirun \
    -np $NP \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    cat /proc/1/cgroup

if [ "${ENABLE_VALGRIND}" = "yes" ]; then
    echo "INFO: UCC MPI unit tests (CPU) with Valgrind..."
    command -v valgrind
    valgrind --version
    MPI_APP="/opt/nvidia/torch-ucc/src/ucc/build/test/mpi/.libs/ucc_test_mpi --mtypes host --inplace 2 --root random:2 --count_bits 32,64 --displ_bits 32,64"
    VALGRIND_ARGS="\
         --tool=memcheck \
         --leak-check=full \
         --track-origins=yes \
         --fair-sched=try \
         --num-callers=25 \
         --error-exitcode=1 \
         --child-silent-after-fork=yes \
         --suppressions=/opt/nvidia/torch-ucc/src/ucc/.ci/configs/valgrind/mpirun_valgrind.supp \
         ${VALGRIND_EXTRA_ARGS}"
    # https://github.com/openucx/ucx/issues/4866
    # shellcheck disable=SC2086
    mpirun \
        --hostfile ${HOSTFILE} \
        --map-by node \
        --allow-run-as-root \
        --mca plm_rsh_args '-p 12345' \
        -x PATH \
        -x LD_LIBRARY_PATH \
        --mca opal_common_ucx_opal_mem_hooks 1 \
        -np 1 valgrind ${VALGRIND_ARGS} ${MPI_APP} : -np $((NP - 1)) ${MPI_APP}
    echo "INFO: UCC MPI unit tests (CPU) with Valgrind... DONE"
else
    echo "INFO: UCC MPI unit tests (CPU/GPU with NCCL) ..."
    ## shellcheck disable=SC2086
    #mpirun \
    #    -np $NP \
    #    --hostfile ${HOSTFILE} \
    #    --map-by node \
    #    --allow-run-as-root \
    #    --mca plm_rsh_args '-p 12345' \
    #    -x PATH \
    #    -x LD_LIBRARY_PATH \
    #    /opt/nvidia/torch-ucc/src/ucc/build/test/mpi/ucc_test_mpi --mtypes host,cuda --inplace 2 --set_device 1 --root random:2 --count_bits 32,64 --displ_bits 32,64
    #echo "INFO: UCC MPI unit tests (CPU/GPU with NCCL) ... DONE"
    #
    #echo "INFO: UCC MPI unit tests (GPU without NCCL) ..."
    ## shellcheck disable=SC2086
    #mpirun \
    #    -np $NP \
    #    --hostfile ${HOSTFILE} \
    #    --map-by node \
    #    --allow-run-as-root \
    #    --mca plm_rsh_args '-p 12345' \
    #    -x PATH \
    #    -x LD_LIBRARY_PATH \
    #    -x UCC_TL_NCCL_TUNE=0 \
    #    /opt/nvidia/torch-ucc/src/ucc/build/test/mpi/ucc_test_mpi --mtypes cuda --inplace 2 --set_device 1 --root random:2 --count_bits 32,64 --displ_bits 32,64
    #echo "INFO: UCC MPI unit tests (GPU without NCCL) ... DONE"
fi
