# Determine number of parallel build jobs based on available system memory.
# In containers/Kubernetes, use cgroup memory limits to avoid OOM.
if [ -f /.dockerenv ] || [ -f /run/.containerenv ] || [ -n "${KUBERNETES_SERVICE_HOST}" ]; then
    if [ -f /sys/fs/cgroup/memory/memory.limit_in_bytes ]; then
        limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)
    elif [ -f /sys/fs/cgroup/memory.max ]; then
        limit=$(cat /sys/fs/cgroup/memory.max)
        [ "$limit" = "max" ] && limit=$((4 * 1024 * 1024 * 1024))
    else
        limit=$((4 * 1024 * 1024 * 1024))
    fi
    NPROC=$((limit / (1024 * 1024 * 1024)))
    [ "$NPROC" -gt 16 ] && NPROC=16
    [ "$NPROC" -lt 1 ] && NPROC=1
else
    NPROC=$(nproc --all)
fi
export NPROC

