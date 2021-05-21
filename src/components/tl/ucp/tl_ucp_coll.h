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
#include "components/mc/base/ucc_mc_base.h"
#include "tl_ucp_tag.h"
#include "utils/profile/ucc_profile.h"

#define UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR 1
extern const char
    *ucc_tl_ucp_default_alg_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR];

typedef struct ucc_tl_ucp_task {
    ucc_coll_task_t      super;
    ucc_coll_args_t      args;
    ucc_tl_ucp_team_t *  team;
    uint32_t             send_posted;
    uint32_t             send_completed;
    uint32_t             recv_posted;
    uint32_t             recv_completed;
    uint32_t             tag;
    uint32_t             n_polls;
    ucc_tl_team_subset_t subset;
    union {
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
        } barrier;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            void                   *scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
        } allreduce_kn;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            void                   *scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
        } reduce_scatter_kn;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
        } allgather_kn;
        struct {
            ucc_rank_t              dist;
            uint32_t                radix;
        } bcast_kn;
    };
} ucc_tl_ucp_task_t;

static inline ucc_tl_ucp_task_t *ucc_tl_ucp_get_task(ucc_tl_ucp_team_t *team)
{
    ucc_tl_ucp_context_t *ctx  = UCC_TL_UCP_TEAM_CTX(team);
    ucc_tl_ucp_task_t    *task = ucc_mpool_get(&ctx->req_mp);;

    UCC_PROFILE_REQUEST_NEW(task, "tl_ucp_task", 0);
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
    UCC_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

static inline ucc_schedule_t *ucc_tl_ucp_get_schedule(ucc_tl_ucp_team_t *team)
{
    ucc_tl_ucp_context_t *ctx      = UCC_TL_UCP_TEAM_CTX(team);
    ucc_schedule_t       *schedule = ucc_mpool_get(&ctx->req_mp);

    UCC_PROFILE_REQUEST_NEW(schedule, "tl_ucp_task", 0);
    ucc_schedule_init(schedule, UCC_TL_UCP_TEAM_CORE_CTX(team));

    return schedule;
}

static inline void ucc_tl_ucp_put_schedule(ucc_schedule_t *schedule)
{
    UCC_PROFILE_REQUEST_FREE(schedule);
    ucc_mpool_put(schedule);
}

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *     team,
                                  ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_ucp_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                       ucc_coll_task_t *coll_task);

static inline ucc_tl_ucp_task_t *
ucc_tl_ucp_init_task(ucc_base_coll_args_t *coll_args, ucc_base_team_t *team)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task    = ucc_tl_ucp_get_task(tl_team);

    ucc_coll_task_init(&task->super);
    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));
    task->team           = tl_team;
    task->tag            = tl_team->seq_num;
    tl_team->seq_num     = (tl_team->seq_num + 1) % UCC_TL_UCP_MAX_COLL_TAG;
    task->super.finalize = ucc_tl_ucp_coll_finalize;
    task->super.triggered_post = ucc_tl_ucp_triggered_post;
    return task;
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

ucc_status_t ucc_tl_ucp_alg_id_to_init(int alg_id, const char *alg_id_str,
                                       ucc_coll_type_t          coll_type,
                                       ucc_memory_type_t        mem_type,
                                       ucc_base_coll_init_fn_t *init);
#endif
