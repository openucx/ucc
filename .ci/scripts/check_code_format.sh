#!/bin/bash -eEx
set -o pipefail

TMP_DIR="/tmp/$(basename "$0" .sh)_$$"
rm -rf "${TMP_DIR}"
mkdir -p "${TMP_DIR}"

UCC_SRC_DIR="/opt/nvidia/src/ucc"
cd "${UCC_SRC_DIR}"

git log -1 HEAD
git log -1 HEAD^
echo "INFO: Checking code format on diff HEAD^..HEAD"
#git-clang-format --binary=clang-format-11 --style=file --diff HEAD^ HEAD >"${TMP_DIR}/check_code_format.patch"
git-clang-format --diff HEAD^ HEAD >"${TMP_DIR}/check_code_format.patch"
echo "INFO: Generated patch file:"
cat "${TMP_DIR}/check_code_format.patch"
if [ "$(cat "${TMP_DIR}/check_code_format.patch")" = "no modified files to format" ]; then
  echo "INFO: code format is OK"
  echo "PASS"
  exit 0
fi

git apply "${TMP_DIR}/check_code_format.patch"
if ! git diff --quiet --exit-code; then
  echo "WARNING: Code is not formatted according to the code style, see https://github.com/openucx/ucx/wiki/Code-style-checking for more info"
  echo "FAIL"
fi

rm -rf "${TMP_DIR}"
