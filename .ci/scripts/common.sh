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
    export NPROC=$nproc
fi

