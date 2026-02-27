#!/bin/bash -xe


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

readonly SRUN_BASE="srun --jobid=${SLURM_JOB_ID} --nodes=${SLURM_NODES} --mpi=pmix --ntasks-per-node=${NVLS_MPI_PPN:-4} --container-image=${DOCKER_IMAGE_NAME}"
readonly SRUN_PERFTEST="${SRUN_BASE} '/opt/nvidia/src/ucc/.ci/scripts/run_tests_ucc_nvls_all.sh'"
readonly SRUN_MPI="${SRUN_BASE} '/opt/nvidia/src/ucc/.ci/scripts/run_tests_ucc_nvls_mpi.sh'"

# Validate SLURM_HEAD_NODE is set
if [ -z "${SLURM_HEAD_NODE}" ]; then
    echo "ERROR: SLURM_HEAD_NODE is not set or empty"
    exit 1
fi

run_srun() {
    local SLURM_COMMAND="$1"
    case "${SLURM_HEAD_NODE}" in
        scctl)
            echo "Using scctl client to connect and execute slurm command"
            scctl client connect -- "${SLURM_COMMAND}"
            ;;
        dlcluster*)
            echo "Connecting to SLURM head node: ${SLURM_HEAD_NODE}"
            local SLURM_COMMAND_ESCAPED="${SLURM_COMMAND//\#/\\#}"
            eval "${SSH_CMD} ${SLURM_HEAD_NODE} \"${SLURM_COMMAND_ESCAPED}\""
            ;;
        *)
            echo "Connecting to SLURM head node: ${SLURM_HEAD_NODE}"
            eval "${SSH_CMD} ${SLURM_HEAD_NODE} ${SLURM_COMMAND}"
            ;;
    esac
}

echo "INFO: Running NVLS smoke test + perf tests ..."
run_srun "${SRUN_PERFTEST}"
echo "INFO: Running NVLS smoke test + perf tests ... DONE"

echo "INFO: Running NVLS MPI tests ..."
run_srun "${SRUN_MPI}"
echo "INFO: Running NVLS MPI tests ... DONE"
