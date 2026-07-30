#!/bin/bash
# run_cache_equivalence.sh - cache enabled-vs-disabled equivalence test.
#
# Runs ucc_test_mpi over the same team/collective set with the cache on (pass A)
# and off (pass B), using its built-in per-collective correctness checks as the
# equivalence oracle (no cross-run diffing). Exits non-zero if any run fails.
#
# Usage: bash test/mpi/run_cache_equivalence.sh [NP]
#
# NP defaults to 8; --oversubscribe keeps it usable on a single-host container.
# Pass a larger NP when a bigger cluster is available.

set -eE

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"

MPIRUN="${MPIRUN:-$(command -v mpirun)}"
# EXE defaults to the ucc_test_mpi built next to this script, but may be
# overridden (e.g. by CI, where the build tree differs from the source tree).
EXE="${EXE:-${SCRIPT_DIR}/ucc_test_mpi}"
NP="${1:-8}"

TEAMS="world,half,odd_even,reverse"
COLLS="barrier,allreduce,bcast,alltoall,allgather"
MTYPES="host"

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

PASS_COUNT=0
FAIL_COUNT=0

emit_result() {
    local state="$1"   # CACHE_ON | CACHE_OFF
    local team="$2"    # team label or "all"
    local result="$3"  # PASS | FAIL
    local detail="${4:-}"
    if [ -n "${detail}" ]; then
        echo "CACHE_EQUIV_RESULT state=${state} team=${team} result=${result} (${detail})"
    else
        echo "CACHE_EQUIV_RESULT state=${state} team=${team} result=${result}"
    fi
    if [ "${result}" = "PASS" ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
}

# run_mpi_pass <state_label> <cache_enable_val> <teams_arg>
# Runs ucc_test_mpi over the collective suite AND the team-cache correctness suite
# (UCC_TEAM_CACHE_CORRECTNESS_TESTS=1). The collective suite proves on-vs-off
# equivalence; the correctness suite exercises actual reuse/derivation. For the
# cache-on pass we also assert the correctness suite did NOT skip itself (which it
# does when the cache is disabled), so a silently-inert cache cannot masquerade as
# "equivalent" by never touching the cache at all. The suite MPI_Aborts on a real
# cache failure, so a non-zero exit is caught below.
run_mpi_pass() {
    local label="$1" enable_val="$2" teams_arg="$3"

    local rc=0
    local log
    log="$(mktemp)"
    "${MPIRUN}" \
        --allow-run-as-root \
        -np "${NP}" \
        --oversubscribe \
        -x "UCC_TEAM_CACHE_ENABLE=${enable_val}" \
        -x "UCC_TEAM_CACHE_CORRECTNESS_TESTS=1" \
        "${EXE}" \
        -t "${teams_arg}" \
        -c "${COLLS}" \
        --mtypes "${MTYPES}" \
        > "${log}" 2>&1 || rc=$?
    cat "${log}"

    local t result="PASS"
    [ "${rc}" -eq 0 ] || result="FAIL"

    if [ "${label}" = "CACHE_ON" ] && [ "${result}" = "PASS" ]; then
        if grep -qa "SKIP all team-cache tests" "${log}"; then
            echo "CACHE_EQUIV: ERROR - cache-on pass did not exercise the cache" \
                 "(correctness suite skipped itself); the cache may be inert"
            result="FAIL"
        fi
    fi
    rm -f "${log}"

    IFS=',' read -ra team_list <<< "${teams_arg}"
    for t in "${team_list[@]}"; do
        emit_result "${label}" "${t}" "${result}"
    done

    return "${rc}"
}

# -----------------------------------------------------------------------------
# Pre-flight
# -----------------------------------------------------------------------------

if [ ! -x "${EXE}" ]; then
    echo "ERROR: ucc_test_mpi not found or not executable at ${EXE}"
    exit 1
fi

echo "========================================================================"
echo "  cache equivalence test  NP=${NP}  teams=${TEAMS}  colls=${COLLS}"
echo "========================================================================"

# -----------------------------------------------------------------------------
# Pass A - cache ENABLED
# -----------------------------------------------------------------------------

echo ""
echo "--- Pass A: UCC_TEAM_CACHE_ENABLE=1 ---"
pass_a_rc=0
run_mpi_pass "CACHE_ON" "1" "${TEAMS}" || pass_a_rc=$?

# -----------------------------------------------------------------------------
# Pass B - cache DISABLED
# -----------------------------------------------------------------------------

echo ""
echo "--- Pass B: UCC_TEAM_CACHE_ENABLE=0 ---"
pass_b_rc=0
run_mpi_pass "CACHE_OFF" "0" "${TEAMS}" || pass_b_rc=$?

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
echo "========================================================================"
echo "  SUMMARY  PASS=${PASS_COUNT}  FAIL=${FAIL_COUNT}"
echo "========================================================================"

if [ "${FAIL_COUNT}" -gt 0 ] || [ "${pass_a_rc}" -ne 0 ] || [ "${pass_b_rc}" -ne 0 ]; then
    echo "CACHE_EQUIV_OVERALL: FAIL"
    exit 1
else
    echo "CACHE_EQUIV_OVERALL: PASS"
    exit 0
fi
