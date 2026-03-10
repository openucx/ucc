#!/bin/bash -eEx
set -o pipefail
export CFLAGS="-Wno-error=maybe-uninitialized"
export CXXFLAGS="-Wno-error=maybe-uninitialized"

export UCC_ENABLE_GTEST=${UCC_ENABLE_GTEST:-yes}
export UCC_ENABLE_NVLS=${UCC_ENABLE_NVLS:-no}
export UCC_BUILD_TLS=${UCC_BUILD_TLS:-cuda,nccl,self,sharp,shm,ucp,mlx5}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd -P)"
. "${SCRIPT_DIR}/common.sh"

echo "INFO: Build UCC"
UCC_SRC_DIR="${SRC_DIR}/ucc"
cd "${UCC_SRC_DIR}"
"${UCC_SRC_DIR}/autogen.sh"
mkdir -p "${UCC_SRC_DIR}/build"
cd "${UCC_SRC_DIR}/build"

# Build base configure flags
CONFIGURE_FLAGS="--with-ucx=${UCX_INSTALL_DIR} --with-cuda=${CUDA_HOME} \
    --prefix=${UCC_INSTALL_DIR} --with-mpi \
    --with-tls=${UCC_BUILD_TLS}"

# Add NVLS support if enabled
if [ "${UCC_ENABLE_NVLS}" = "yes" ] || [ "${UCC_ENABLE_NVLS}" = "true" ] || [ "${UCC_ENABLE_NVLS}" = "1" ]; then
    echo "INFO: Enabling NVLS support for GB300NVL72"
    CONFIGURE_FLAGS="${CONFIGURE_FLAGS} --with-nvls --with-nvcc-gencode=\"-gencode=arch=compute_100,code=sm_100\""
else
    CONFIGURE_FLAGS="${CONFIGURE_FLAGS} --with-nvcc-gencode=\"-gencode=arch=compute_86,code=sm_86\""
fi
if [ "${UCC_ENABLE_GTEST}" = "yes" ] || [ "${UCC_ENABLE_GTEST}" = "true" ] || [ "${UCC_ENABLE_GTEST}" = "1" ]; then
    echo "INFO: Enabling gtest"
    CONFIGURE_FLAGS="${CONFIGURE_FLAGS} --enable-gtest"
fi

echo "INFO: Configure flags: ${CONFIGURE_FLAGS}"
eval "${UCC_SRC_DIR}/configure ${CONFIGURE_FLAGS}"

# Skip libtool relinking during install: the relink produces identical RUNPATH
# and adds ~5s of pure overhead per build.
if ! grep -q 'need_relink=yes' libtool; then
    echo "WARN: libtool relinking patch had no effect (need_relink=yes not found)"
fi
sed -i 's/need_relink=yes/need_relink=no/g' libtool

make "-j${NPROC}" install
echo "${UCC_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/ucc.conf
ldconfig
ldconfig -p | grep -i libucc
