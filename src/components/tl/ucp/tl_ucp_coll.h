/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_COLL_H_
#define UCC_TL_UCP_COLL_H_

#include "tl_ucp.h"
#include "schedule/ucc_schedule.h"
#include "coll_patterns/recursive_knomial.h"

typedef struct ucc_tl_ucp_task {
    ucc_coll_task_t      super;
    ucc_coll_args_t      args;
    ucc_tl_ucp_team_t   *team;
    uint32_t             send_posted;
    uint32_t             send_completed;
    uint32_t             recv_posted;
    uint32_t             recv_completed;
    uint32_t             tag;
    uint32_t             n_polls;
    ucc_tl_team_subset_t subset;
    union {
        struct {
            int                   phase;
            ucc_knomial_pattern_t p;
        } barrier;
        struct {
            int                   phase;
            ucc_knomial_pattern_t p;
            void                 *scratch;
        } allreduce_kn;
        struct {
            ucc_rank_t            dist;
            uint32_t              radix;
        } bcast_kn;
    };
} ucc_tl_ucp_task_t;

static inline ucc_tl_ucp_task_t *ucc_tl_ucp_get_task(ucc_tl_ucp_team_t *team)
{
    ucc_tl_ucp_context_t *ctx = UCC_TL_UCP_TEAM_CTX(team);
    ucc_tl_ucp_task_t *task;
    task                     = ucc_mpool_get(&ctx->req_mp);
    task->super.super.status = UCC_OPERATION_INITIALIZED;
    task->super.flags        = 0;
    task->send_posted        = 0;
    task->send_completed     = 0;
    task->recv_posted        = 0;
    task->recv_completed     = 0;
    task->n_polls            = ctx->cfg.n_polls;
    task->team               = team;
    task->subset.map.type    = UCC_EP_MAP_FULL;
    task->subset.map.ep_num  = team->size;
    task->subset.myrank      = team->rank;
    return task;
}

static inline void ucc_tl_ucp_put_task(ucc_tl_ucp_task_t *task)
{
    ucc_mpool_put(task);
}

#define UCC_TL_UCP_TASK_P2P_COMPLETE(_task)                                    \
    (((_task)->send_posted == (_task)->send_completed) &&                      \
     ((_task)->recv_posted == (_task)->recv_completed))

static inline ucc_status_t ucc_tl_ucp_test(ucc_tl_ucp_task_t *task)
{
    int polls = 0;
    if (UCC_TL_UCP_TASK_P2P_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_P2P_COMPLETE(task)) {
            return UCC_OK;
        }
        ucp_worker_progress(UCC_TL_UCP_TEAM_CTX(task->team)->ucp_worker);
    }
    return UCC_INPROGRESS;
}
ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t      *team,
                                  ucc_coll_task_t     **task_h);

#endif
