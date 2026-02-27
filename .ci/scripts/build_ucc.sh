#!/bin/bash -eEx
set -o pipefail
export CFLAGS="-Wno-error=maybe-uninitialized"
export CXXFLAGS="-Wno-error=maybe-uninitialized"

export UCC_ENABLE_GTEST=${UCC_ENABLE_GTEST:-yes}
export UCC_ENABLE_NVLS=${UCC_ENABLE_NVLS:-no}

# In containers, calculate based on memory limits to avoid OOM
# Determine number of parallel build jobs based on available system memory if running inside a container/Kubernetes
if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || [ -n "${KUBERNETES_SERVICE_HOST}" ]; then
    # Prefer cgroupv1 path, fall back to cgroupv2 or static default if not found
    if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    elif [ -f /sys/fs/cgroup/memory.max ]; then
        limit=$(cat /sys/fs/cgroup/memory.max)
        # If cgroupv2 limit is "max", meaning unlimited, set to 4GB to avoid OOM
        [ "$limit" = "max" ] && limit=$((4 * 1024 * 1024 * 1024))
    else
        # Default to 4GB if no limit is found
        limit=$((4 * 1024 * 1024 * 1024))
    fi

    # Use 1 build process per GB of memory, clamp in [1,16]
    nproc=$((limit / (1024 * 1024 * 1024)))
    [ "$nproc" -gt 16 ] && nproc=16
    [ "$nproc" -lt 1 ] && nproc=1
else
    nproc=$(nproc --all)
fi

echo "INFO: Build UCC"
UCC_SRC_DIR="${SRC_DIR}/ucc"
cd "${UCC_SRC_DIR}"
"${UCC_SRC_DIR}/autogen.sh"
mkdir -p "${UCC_SRC_DIR}/build"
cd "${UCC_SRC_DIR}/build"

# Build base configure flags
CONFIGURE_FLAGS="--with-ucx=${UCX_INSTALL_DIR} --with-cuda=${CUDA_HOME} \
    --prefix=${UCC_INSTALL_DIR} --with-mpi \
    --with-tls=cuda,nccl,self,sharp,shm,ucp,mlx5"

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
sed -i 's/need_relink=yes/need_relink=no/g' libtool

make "-j${nproc}" install
echo "${UCC_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/ucc.conf
ldconfig
ldconfig -p | grep -i libucc
