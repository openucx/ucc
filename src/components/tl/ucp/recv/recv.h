/**
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef RECV_H_
#define RECV_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

ucc_status_t ucc_tl_ucp_recv_init(ucc_tl_ucp_task_t *task);
ucc_status_t ucc_tl_ucp_recv_start(ucc_coll_task_t *coll_task);
void ucc_tl_ucp_recv_progress(ucc_coll_task_t *coll_task);

#endif
