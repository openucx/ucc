#!/bin/bash -eEx
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
. "${SCRIPT_DIR}/common.sh"

echo "INFO: Build UCX"
cd "${SRC_DIR}/ucx"
"${SRC_DIR}/ucx/autogen.sh"
mkdir -p "${SRC_DIR}/ucx/build-${UCX_BUILD_TYPE}"
cd "${SRC_DIR}/ucx/build-${UCX_BUILD_TYPE}"
"${SRC_DIR}/ucx/contrib/configure-release-mt" --with-cuda="${CUDA_HOME}" --prefix="${UCX_INSTALL_DIR}"
make "-j${NPROC}" install
echo "${UCX_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/ucx.conf
ldconfig
ldconfig -p | grep -i ucx
