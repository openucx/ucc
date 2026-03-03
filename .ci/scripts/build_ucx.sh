#!/bin/bash -eEx
set -o pipefail

# Determine number of parallel build jobs based on available memory (container-aware)
if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || [ -n "${KUBERNETES_SERVICE_HOST}" ]; then
    if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    elif [ -f /sys/fs/cgroup/memory.max ]; then
        limit=$(cat /sys/fs/cgroup/memory.max)
        [ "$limit" = "max" ] && limit=$((4 * 1024 * 1024 * 1024))
    else
        limit=$((4 * 1024 * 1024 * 1024))
    fi
    nproc=$((limit / (1024 * 1024 * 1024)))
    [ "$nproc" -gt 16 ] && nproc=16
    [ "$nproc" -lt 1 ] && nproc=1
else
    nproc=$(nproc --all)
fi

echo "INFO: Build UCX"
cd "${SRC_DIR}/ucx"
"${SRC_DIR}/ucx/autogen.sh"
mkdir -p "${SRC_DIR}/ucx/build-${UCX_BUILD_TYPE}"
cd "${SRC_DIR}/ucx/build-${UCX_BUILD_TYPE}"
"${SRC_DIR}/ucx/contrib/configure-release-mt" --with-cuda="${CUDA_HOME}" --prefix="${UCX_INSTALL_DIR}"
make "-j${nproc}" install
echo "${UCX_INSTALL_DIR}/lib" > /etc/ld.so.conf.d/ucx.conf
ldconfig
ldconfig -p | grep -i ucx
