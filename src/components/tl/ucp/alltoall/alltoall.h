/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALL_H_
#define ALLTOALL_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

ucc_status_t ucc_tl_ucp_alltoall_init(ucc_tl_ucp_task_t *task);

#endif
