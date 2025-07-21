#!/bin/bash -eEx
set -o pipefail
export CFLAGS="-Wno-error=maybe-uninitialized"
export CXXFLAGS="-Wno-error=maybe-uninitialized"

echo "INFO: Build UCC"
UCC_SRC_DIR="${SRC_DIR}/ucc"
cd "${UCC_SRC_DIR}"
"${UCC_SRC_DIR}/autogen.sh"
mkdir -p "${UCC_SRC_DIR}/build"
cd "${UCC_SRC_DIR}/build"
"${UCC_SRC_DIR}/configure" --with-ucx="${UCX_INSTALL_DIR}" --with-cuda="${CUDA_HOME}" \
    --prefix="${UCC_INSTALL_DIR}" --enable-gtest --with-mpi \
    --with-tls=cuda,nccl,self,sharp,shm,ucp,mlx5 \
    --with-nvcc-gencode="-gencode arch=compute_86,code=sm_86"
make -j install
echo "${UCC_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/ucc.conf
ldconfig
ldconfig -p | grep -i libucc
