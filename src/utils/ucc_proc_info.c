/**
* Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
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

ucc_status_t ucc_get_bound_socket_id(int *socketid)
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
        ucc_error("Error when manually trying to discover socket_id using "
                  "sched_getaffinity()");
        __sched_cpufree(cpuset);
        return UCC_ERR_NO_MESSAGE;
    }

    socket_ids = ucc_malloc(nr_cpus * sizeof(int), "socket_ids");
    if (!socket_ids) {
        ucc_error("failed to allocate %zd bytes for socket_ids array",
                  nr_cpus * sizeof(int));
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
                *socketid = i;
                break;
            }
        }
        ucc_assert(((*socketid) >= 0) && ((*socketid) < nr_cpus));
    }
    ucc_free(socket_ids);
    return UCC_OK;
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
    ucc_local_proc.socket_id = -1;

    ucc_debug("proc pid %d, host %s, host_hash %lu",
              ucc_local_proc.pid, ucc_local_hostname, ucc_local_proc.host_hash);

    if (UCC_OK != ucc_get_bound_socket_id(&ucc_local_proc.socket_id)) {
        ucc_debug("failed to get bound socket id");
    }

    return UCC_OK;
}
