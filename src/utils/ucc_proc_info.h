/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_PROC_INFO_H_
#define UCC_PROC_INFO_H_

#include "config.h"
#include "ucc/api/ucc.h"
#include <unistd.h>

typedef uint64_t ucc_host_id_t;
typedef uint32_t ucc_socket_id_t;

typedef struct ucc_proc_info {
    ucc_host_id_t   host_hash;
    ucc_socket_id_t socket_id;
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
