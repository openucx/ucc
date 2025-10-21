#!/bin/bash -eE

set -o pipefail

handle_error() {
    err_code=$?
    local script_name="${0##*/}"
    local last_command="${BASH_COMMAND}"
    local error_line="${BASH_LINENO[${BASH_LINENO[@]} - 1]}"
    echo "Error: ${script_name} - Command '${last_command}' failed at line ${error_line}."
    exit $err_code
}
trap 'handle_error' ERR

#
# This script runs static code analysis using Coverity.
# The script needs to be run from the main directory.
#
export UCC_PASSWORD=$UCC_PASSWORD
export UCC_USERNAME=$UCC_USERNAME
topdir=$(git rev-parse --show-toplevel)
cd "$topdir" || exit 1
module load hpcx-gcc
module load dev/cuda12.9.0
module load dev/nccl_2.26.5-1_cuda12.9.0
module load tools/cov-2021.12
./autogen.sh
./configure --with-nccl --with-tls=cuda,nccl,self,sharp,shm,ucp,mlx5 --with-ucx="${HPCX_UCX_DIR}" --with-sharp="${HPCX_SHARP_DIR}" --with-nvcc-gencode="-gencode arch=compute_86,code=sm_86"
make_opt="-j$(($(nproc) / 2 + 1))"
COV_BUILD_DIR=$(dirname "$0")/cov-build
mkdir -p "$COV_BUILD_DIR"
COV_ANALYSE_OPTIONS+=" --all"
COV_ANALYSE_OPTIONS+=" --enable-fnptr"
COV_ANALYSE_OPTIONS+=" --fnptr-models"
COV_ANALYSE_OPTIONS+=" --checker-option INFINITE_LOOP:report_bound_type_mismatch:true"
COV_ANALYSE_OPTIONS+=" --checker-option RESOURCE_LEAK:allow_unimpl:true"
COV_ANALYSE_OPTIONS+=" --aggressiveness-level medium"

function show_usage() {
    echo ""
    echo "Usage $(basename "$0") [options]"
    echo ""
    echo "Run static code analysis using Coverity, generate a defects report, and compare"
    echo "the current report with the known issues filter file."
    echo ""
    echo "Script should be run from the main directory after makefiles have been generated."
    echo ""
    echo "Options:"
    echo "--cov-analyze-args <args>  Additional arguments for cov-analyze."
    echo "                           For more information, see cov-analyze manual."
    echo ""
    echo "--skip-coverity            Skip the Coverity stage."
    echo "--skip-cov-build           Skip cov-build in the Coverity stage."
    echo "                           filters out at least one issue."
    echo ""
    echo "-h, --help                 Show usage."
    echo ""
}

# Build and run static code analysis with Coverity
function build_with_coverity() {
    echo "Building target with Coverity..."

    if [ ! -f Makefile ]; then
        echo "ERROR: Make file not found"
        echo "Need to run from a directory with Make files"
        return 1
    fi

    make clean >/dev/null
    # Run cov-build
    cov-build --dir "${COV_BUILD_DIR}" make $make_opt all >/dev/null
    err_code=$?
    return $err_code
}

# Run Coverity analysis
function run_coverity_analysis() {
    echo "Running Coverity analysis..."

    # Run cov-analyze
    # shellcheck disable=SC2086
    cov-analyze --dir "${COV_BUILD_DIR}" "$@" --jobs auto $COV_ANALYSE_OPTIONS
    err_code=$?
    return $err_code
}

# Main function
function main() {
    # Parse command-line options
    while [[ $# -gt 0 ]]; do
        case "$1" in
        --cov-analyze-args)
            shift
            COV_ANALYZE_ARGS="$1"
            ;;
        --skip-coverity)
            SKIP_COVERITY=true
            ;;
        --skip-cov-build)
            SKIP_COV_BUILD=true
            ;;
        -h | --help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unrecognized option: $1"
            show_usage
            exit 1
            ;;
        esac
        shift
    done

    # Build with Coverity
    if [ -z "$SKIP_COVERITY" ] && [ -z "$SKIP_COV_BUILD" ]; then
        build_with_coverity
    fi

    # Run Coverity analysis
    if [ -z "$SKIP_COVERITY" ]; then
    # shellcheck disable=SC2086
        if ! run_coverity_analysis $COV_ANALYZE_ARGS; then
            echo "Coverity analysis failed"
            return 1
        fi
    fi

    echo "Uploading to synopsys main coverity server"
    cov-commit-defects --ssl --on-new-cert trust --host coverity.mellanox.com --port 8443 \
    --user "${UCC_USERNAME}" --password "${UCC_PASSWORD}" --dir "$COV_BUILD_DIR" --stream ucc_master
    return 0
}

# Call the main function
main "$@"
