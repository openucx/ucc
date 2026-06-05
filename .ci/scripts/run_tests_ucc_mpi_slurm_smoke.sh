#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
# shellcheck source=/dev/null
. "${SCRIPT_DIR}/mpi_slurm_common.sh"

mpi_slurm_setup
mpi_slurm_run_smoke
