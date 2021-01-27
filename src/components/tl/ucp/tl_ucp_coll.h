/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_COLL_H_
#define UCC_TL_UCP_COLL_H_
#include "tl_ucp.h"
#include "schedule/ucc_schedule.h"
typedef struct ucc_tl_ucp_task {
    ucc_coll_task_t     super;
    ucc_coll_op_args_t  args;
    ucc_tl_ucp_team_t  *team;
    uint32_t            send_posted;
    uint32_t            send_completed;
    uint32_t            recv_posted;
    uint32_t            recv_completed;
} ucc_tl_ucp_task_t;

static inline ucc_tl_ucp_task_t *ucc_tl_ucp_get_task(ucc_tl_ucp_context_t *ctx)
{
    ucc_tl_ucp_task_t *task;
    task                     = ucc_mpool_get(&ctx->req_mp);
    task->super.super.status = UCC_OPERATION_INITIALIZED;
    task->send_posted        = 0;
    task->send_completed     = 0;
    task->recv_posted        = 0;
    task->recv_completed     = 0;
    return task;
}

static inline void ucc_tl_ucp_put_task(ucc_tl_ucp_task_t *task)
{
    ucc_mpool_put(task);
}

#define UCC_TL_UCP_TASK_P2P_COMPLETE(_task)               \
    (((_task)->send_posted == (_task)->send_completed) && \
     ((_task)->recv_posted == (_task)->recv_completed))

#endif
