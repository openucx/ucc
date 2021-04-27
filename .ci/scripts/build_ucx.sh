#!/bin/bash -eEx
set -o pipefail

echo "INFO: Build UCX"
cd "${TORCH_UCC_SRC_DIR}/ucx"
"${TORCH_UCC_SRC_DIR}/ucx/autogen.sh"
mkdir -p "${TORCH_UCC_SRC_DIR}/ucx/build-${UCX_BUILD_TYPE}"
cd "${TORCH_UCC_SRC_DIR}/ucx/build-${UCX_BUILD_TYPE}"
"${TORCH_UCC_SRC_DIR}/ucx/contrib/configure-release-mt" --with-cuda="${CUDA_HOME}" --prefix="${UCX_INSTALL_DIR}"
make -j install
echo "${UCX_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/ucx.conf
ldconfig
ldconfig -p | grep -i ucx
