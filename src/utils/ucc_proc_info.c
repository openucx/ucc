/**
* Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* See file LICENSE for terms.
*/
#include "ucc_proc_info.h"
#include "ucc_log.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_math.h"
#include <errno.h>
#include <sched.h>
#include <limits.h>
#include <stdint.h>
#include "config.h"
#ifdef HAVE_UCS_GET_SYSTEM_ID
#include <ucs/sys/uid.h>
#endif
#include <dlfcn.h>

ucc_proc_info_t ucc_local_proc;
static char ucc_local_hostname[HOST_NAME_MAX];

const char*  ucc_hostname()
{
    return ucc_local_hostname;
}

uint64_t ucc_get_system_id()
{
#ifdef HAVE_UCS_GET_SYSTEM_ID
    return ucs_get_system_id();
#else
    return ucc_str_hash_djb2(ucc_local_hostname);
#endif
}

typedef unsigned long int cpu_mask_t;
#define NCPUBITS (8 * sizeof(cpu_mask_t))

#define CPUELT(cpu) ((cpu) / NCPUBITS)
#define CPUMASK(cpu) ((cpu_mask_t)1 << ((cpu) % NCPUBITS))

#define SBGP_CPU_ISSET(cpu, setsize, cpusetp)                                  \
    ({                                                                         \
        size_t __cpu = (cpu);                                                  \
        __cpu < 8 * (setsize)                                                  \
            ? ((((const cpu_mask_t *)((cpusetp)->__bits))[__CPUELT(__cpu)] &   \
                CPUMASK(__cpu))) != 0                                          \
            : 0;                                                               \
    })

static int parse_cpuset_file(FILE *file, int *nr_psbl_cpus)
{
    unsigned long start, stop;
    while (fscanf(file, "%lu", &start) == 1) {
        int c = fgetc(file);
        stop  = start;
        if (c == '-') {
            if (fscanf(file, "%lu", &stop) != 1) {
                /* Range is usually <int>-<int> */
                errno = EINVAL;
                return -1;
            }
            c = fgetc(file);
        }

        if (c == EOF || c == '\n') {
            *nr_psbl_cpus = (int)stop + 1;
            break;
        }

        if (c != ',') {
            /* Wrong terminating char */
            errno = EINVAL;
            return -1;
        }
    }
    return 0;
}

static ucc_status_t ucc_get_bound_socket_id(ucc_socket_id_t *socketid)
{
    cpu_set_t *cpuset = NULL;
    int        sockid = -1, sockid2 = -1;
    int        try, i, n_sockets, cpu, nr_cpus, nr_psbl_cpus = 0;
    size_t     setsize;
    FILE *     fptr, *possible;
    char       str[1024];
    int *      socket_ids, tmpid;

    /* Get the number of total procs and online procs */
    nr_cpus = sysconf(_SC_NPROCESSORS_CONF);

    /* Need to make sure nr_cpus !< possible_cpus+1 */
    possible = fopen("/sys/devices/system/cpu/possible", "r");
    if (possible) {
        if (parse_cpuset_file(possible, &nr_psbl_cpus) == 0) {
            if (nr_cpus < nr_psbl_cpus + 1)
                nr_cpus = nr_psbl_cpus;
        }
        fclose(possible);
    }

    if (!nr_cpus) {
        return UCC_ERR_NO_MESSAGE;
    }

    /* The cpuset size on some kernels needs to be bigger than
     * the number of nr_cpus, hwloc gets around this
     * by blocking on a loop and increasing nr_cpus.
     * We will try 1000 (arbitrary) attempts, and revert to hwloc
     * if all fail */
    setsize = ((nr_cpus + NCPUBITS - 1) / NCPUBITS) * sizeof(cpu_mask_t);
    cpuset  = __sched_cpualloc(nr_cpus);
    if (NULL == cpuset) {
        return UCC_ERR_NO_MESSAGE;
    }
    try = 1000;
    while (0 < sched_getaffinity(0, setsize, cpuset) && try > 0) {
        __sched_cpufree(cpuset);
        try--;
        nr_cpus *= 2;
        cpuset = __sched_cpualloc(nr_cpus);
        if (NULL == cpuset) {
            try = 0;
            break;
        }
        setsize = ((nr_cpus + NCPUBITS - 1) / NCPUBITS) * sizeof(cpu_mask_t);
    }

    /* If after all tries we're still not getting it, error out
     * let hwloc take over */
    if (try == 0) {
        ucc_warn("Error when manually trying to discover socket_id using "
                 "sched_getaffinity()");
        __sched_cpufree(cpuset);
        return UCC_ERR_NO_MESSAGE;
    }

    socket_ids = ucc_malloc(nr_cpus * sizeof(int), "socket_ids");
    if (!socket_ids) {
        ucc_error("failed to allocate %zd bytes for socket_ids array",
                  nr_cpus * sizeof(int));
        __sched_cpufree(cpuset);
        return UCC_ERR_NO_MEMORY;
    }
    /* Loop through all cpus, and check if I'm bound to the socket */
    for (cpu = 0; cpu < nr_cpus; cpu++) {
        socket_ids[cpu] = -1;
        sprintf(str, "/sys/bus/cpu/devices/cpu%d/topology/physical_package_id",
                cpu);
        fptr = fopen(str, "r");
        if (!fptr) {
            /* Do nothing just skip */
            continue;
        }
        /* Read socket id from file */
        if ((1 == fscanf(fptr, "%d", &tmpid)) && (tmpid >= 0)) {
            socket_ids[cpu] = tmpid;
            if (SBGP_CPU_ISSET(cpu, setsize, cpuset)) {
                if (sockid == -1) {
                    sockid = tmpid;
                } else if (tmpid != sockid && sockid2 == -1) {
                    sockid2 = tmpid;
                }
            }
        }
        fclose(fptr);
    }

    /* Check that a process is bound to 1 and only 1 socket */
    if ((sockid != -1) && (sockid2 == -1)) {
        /* Some archs (eg. POWER) seem to have non-linear socket_ids.
          * Convert to logical index by findig first occurence of tmpid in
          * the global socket_ids array. */
        n_sockets = ucc_sort_uniq(socket_ids, nr_cpus, 0);
        for (i = 0; i < n_sockets; i++) {
            if (socket_ids[i] == sockid) {
                if (i > (int)UCC_MAX_SOCKET_ID) {
                    ucc_debug("too large socket id %d", i);
                    __sched_cpufree(cpuset);
                    return UCC_ERR_NOT_SUPPORTED;
                }
                *socketid = i;
                break;
            }
        }
        ucc_assert(((*socketid) >= 0) && ((*socketid) < nr_cpus));
    }
    __sched_cpufree(cpuset);
    ucc_free(socket_ids);
    return UCC_OK;
}

#define LOAD_NUMA_SYM(_sym)                                                    \
    ({                                                                         \
        void *h = dlsym(handle, _sym);                                         \
        if ((error = dlerror()) != NULL) {                                     \
            ucc_debug("%s", error);                                            \
            status = UCC_ERR_NOT_FOUND;                                        \
            goto error;                                                        \
        }                                                                      \
        h;                                                                     \
    })

static ucc_status_t ucc_get_bound_numa_id(ucc_numa_id_t *numaid)
{
    ucc_status_t status = UCC_OK;
    char *       error;
    void *       handle, *cpumask;
    int          i, numa_node, n_cfg_cpus, nn;

    handle = dlopen("libnuma.so", RTLD_LAZY);
    if (!handle) {
        ucc_debug("%s", dlerror());
        return UCC_ERR_NOT_FOUND;
    }

    int (*ucc_numa_available)(void) =
        LOAD_NUMA_SYM("numa_available");
    int (*ucc_numa_num_configured_cpus)(void) =
        LOAD_NUMA_SYM("numa_num_configured_cpus");
    void *(*ucc_numa_allocate_cpumask)(void) =
        LOAD_NUMA_SYM("numa_allocate_cpumask");
    void *(*ucc_numa_sched_getaffinity)(int, void *) =
        LOAD_NUMA_SYM("numa_sched_getaffinity");
    int (*ucc_numa_bitmask_isbitset)(void *, int) =
        LOAD_NUMA_SYM("numa_bitmask_isbitset");
    int (*ucc_numa_node_of_cpu)(int)     = LOAD_NUMA_SYM("numa_node_of_cpu");
    int (*ucc_numa_bitmask_free)(void *) = LOAD_NUMA_SYM("numa_bitmask_free");

    if (-1 == ucc_numa_available()) {
        ucc_debug("libnuma is not available");
        status = UCC_ERR_NO_MESSAGE;
        goto error;
    }
    /* Load the cpumask which a process is bound to, then loop through cpus
       from that mask and check the numa nodes those cpus belong to. If there are
       more than 1 numa nodes detected return -1, i.e. not bound to a numa. */
    cpumask = ucc_numa_allocate_cpumask();
    if (!cpumask) {
        ucc_error("numa_allocate_cpumask failed");
        status = UCC_ERR_NO_MESSAGE;
        goto error;
    }
    ucc_numa_sched_getaffinity(getpid(), cpumask);
    numa_node  = -1;
    n_cfg_cpus = ucc_numa_num_configured_cpus();
    for (i = 0; i < n_cfg_cpus; i++) {
        if (ucc_numa_bitmask_isbitset(cpumask, i)) {
            nn = ucc_numa_node_of_cpu(i);
            if (numa_node == -1) {
                numa_node = nn;
            } else if (numa_node != nn && numa_node >= 0) {
                /* At least 2 different numa nodes detected for a given cpu set.
                   set numa_node to -1, which means not bound to a numa. */
                numa_node = -1;
                break;
            }
        }
    }
    ucc_numa_bitmask_free(cpumask);
    if (numa_node >= 0) {
        if (numa_node > (int)UCC_MAX_NUMA_ID) {
            ucc_debug("too large numa id %d", numa_node);
            status = UCC_ERR_NOT_SUPPORTED;
            goto error;
        }
        *numaid = numa_node;
    }
error:
    dlclose(handle);
    return status;
}

ucc_status_t ucc_local_proc_info_init()
{
    ucc_local_proc.host_hash = gethostid();
    if (gethostname(ucc_local_hostname, sizeof(ucc_local_hostname))) {
        ucc_warn("couldn't get local hostname");
        ucc_local_hostname[0] = '\0';
    } else {
        strtok(ucc_local_hostname, ".");
        ucc_assert(sizeof(ucc_host_id_t) >= sizeof(unsigned long));
        ucc_local_proc.host_hash = ucc_get_system_id();
    }
    ucc_local_proc.pid       = getpid();
    ucc_local_proc.socket_id = UCC_SOCKET_ID_INVALID;
    ucc_local_proc.numa_id   = UCC_NUMA_ID_INVALID;

    if (UCC_OK != ucc_get_bound_socket_id(&ucc_local_proc.socket_id)) {
        ucc_debug("failed to get bound socket id");
    }

    if (UCC_OK != ucc_get_bound_numa_id(&ucc_local_proc.numa_id)) {
        ucc_debug("failed to get bound numa id");
    }

    ucc_debug("proc pid %d, host %s, host_hash %lu, sockid %d, numaid %d",
              ucc_local_proc.pid, ucc_local_hostname, ucc_local_proc.host_hash,
              (int)ucc_local_proc.socket_id, (int)ucc_local_proc.numa_id);

    return UCC_OK;
}
