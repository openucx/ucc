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

N_HOSTS=$(wc --lines "$HOSTFILE" | awk '{print $1}')
NP_HOSTS=${N_HOSTS}
N_CPU=$(nproc)
NP_CPU=$((N_CPU * N_HOSTS))
N_GPU=$(nvidia-smi --list-gpus | wc --lines)
NP_GPU=$((N_GPU * N_HOSTS))

# shellcheck disable=SC2086
mpirun \
    -np ${NP_HOSTS} \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    --report-bindings \
    hostname

# shellcheck disable=SC2086
mpirun \
    -np ${NP_HOSTS} \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    --report-bindings \
    cat /proc/1/cgroup

echo "INFO: UCC MPI unit tests (CPU) ..."
# shellcheck disable=SC2086
mpirun \
    -np ${NP_CPU} \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    --report-bindings \
    /opt/nvidia/torch-ucc/src/ucc/build/test/mpi/ucc_test_mpi --mtypes host --inplace 2 --set_device 1 --root random:2 --count_bits 32,64 --displ_bits 32,64
echo "INFO: UCC MPI unit tests (CPU) ... DONE"

echo "INFO: UCC MPI unit tests (GPU with NCCL) ..."
# shellcheck disable=SC2086
mpirun \
    -np ${NP_GPU} \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    --report-bindings \
    /opt/nvidia/torch-ucc/src/ucc/build/test/mpi/ucc_test_mpi --mtypes cuda --inplace 2 --set_device 1 --root random:2 --count_bits 32,64 --displ_bits 32,64
echo "INFO: UCC MPI unit tests (GPU with NCCL) ... DONE"

echo "INFO: UCC MPI unit tests (GPU without NCCL) ..."
# shellcheck disable=SC2086
mpirun \
    -np ${NP_GPU} \
    --hostfile ${HOSTFILE} \
    --map-by node \
    --allow-run-as-root \
    --mca plm_rsh_args '-p 12345' \
    -x PATH \
    -x LD_LIBRARY_PATH \
    -x UCC_TL_NCCL_TUNE=0 \
    --report-bindings \
    /opt/nvidia/torch-ucc/src/ucc/build/test/mpi/ucc_test_mpi --mtypes cuda --inplace 2 --set_device 1 --root random:2 --count_bits 32,64 --displ_bits 32,64
echo "INFO: UCC MPI unit tests (GPU without NCCL) ... DONE"
