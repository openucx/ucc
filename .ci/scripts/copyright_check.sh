#!/bin/bash -x
set -o pipefail

[[ -z $WORKSPACE ]] && { echo "ERROR: WORKSPACE must be set"; exit 1; }
[[ -z $GITHUB_TOKEN ]] && { echo "ERROR: GITHUB_TOKEN must be set"; exit 1; }

/opt/nvidia/header_check.py \
    --revs HEAD \
    --config "${WORKSPACE}/.ci/copyright-check-map.yaml" \
    --git-repo "${WORKSPACE}" | tee copyrights.log
exit_code=$?

if grep -q ERROR copyrights.log; then
    echo "Copyright check FAILED. Please fix headers"
    exit 1
fi

# No ERROR logged, but the tool exited non-zero.
[[ $exit_code -eq 0 ]] || { echo "ERROR: header_check.py exited with ${exit_code}"; exit 1; }

echo "Copyright check PASSED."
