#!/bin/bash -eEx
set -o pipefail

echo "INFO: Build UCC"
UCC_SRC_DIR="${TORCH_UCC_SRC_DIR}/ucc"
cd "${UCC_SRC_DIR}"
"${UCC_SRC_DIR}/autogen.sh"
mkdir -p "${UCC_SRC_DIR}/build"
cd "${UCC_SRC_DIR}/build"
"${UCC_SRC_DIR}/configure" --with-ucx="${UCX_INSTALL_DIR}" --with-cuda="${CUDA_HOME}" \
    --prefix="${UCC_INSTALL_DIR}" --enable-gtest
make -j install
echo "${UCC_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/ucc.conf
ldconfig
ldconfig -p | grep -i libucc

# Build MPI tests
cd "${UCC_SRC_DIR}/build/test/mpi"
make -j
