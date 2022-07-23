/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_PROC_INFO_H_
#define UCC_PROC_INFO_H_

#include "config.h"
#include "ucc/api/ucc.h"
#include <unistd.h>

typedef uint64_t ucc_host_id_t;
typedef uint8_t  ucc_socket_id_t;
typedef uint8_t  ucc_numa_id_t;

#define UCC_SOCKET_ID_INVALID ((ucc_socket_id_t)-1)
#define UCC_NUMA_ID_INVALID   ((ucc_numa_id_t)-1)

#define UCC_MAX_SOCKET_ID (UCC_SOCKET_ID_INVALID - 1)
#define UCC_MAX_NUMA_ID   (UCC_NUMA_ID_INVALID - 1)

typedef struct ucc_proc_info {
    ucc_host_id_t   host_hash;
    ucc_socket_id_t socket_id;
    ucc_numa_id_t   numa_id;
    ucc_host_id_t   host_id;
    pid_t           pid;
} ucc_proc_info_t;

extern ucc_proc_info_t ucc_local_proc;

#define UCC_PROC_INFO_EQUAL(_pi1, _pi2)                                        \
    (((_pi1).host_hash == (_pi2).host_hash) &&                                 \
     ((_pi1).pid == (_pi2).pid)) //TODO maybe need tid ?

ucc_status_t ucc_local_proc_info_init();

uint64_t ucc_get_system_id();

const char*  ucc_hostname();

#endif
