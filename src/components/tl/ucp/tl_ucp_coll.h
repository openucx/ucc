/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_COLL_H_
#define UCC_TL_UCP_COLL_H_

#include "tl_ucp.h"
#include "schedule/ucc_schedule_pipelined.h"
#include "coll_patterns/recursive_knomial.h"
#include "components/mc/base/ucc_mc_base.h"
#include "components/ec/ucc_ec.h"
#include "tl_ucp_tag.h"

#define UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR 5
extern const char
    *ucc_tl_ucp_default_alg_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR];

#define CALC_KN_TREE_DIST(_size, _radix, _dist)                               \
    do {                                                                      \
        _dist = 1;                                                            \
        while (_dist * _radix < _size) {                                      \
            _dist *= _radix;                                                  \
        }                                                                     \
    } while (0)

#define VRANK(_rank, _root, _team_size)                                       \
    (((_rank) - (_root) + (_team_size)) % (_team_size))

#define INV_VRANK(_rank, _root, _team_size)                                   \
    (((_rank) + (_root)) % (_team_size))

#define EXEC_TASK_TEST(_phase, _errmsg, _etask) do {                           \
    if (_etask != NULL) {                                                      \
        status = ucc_ee_executor_task_test(_etask);                            \
        if (status > 0) {                                                      \
            task->super.status = UCC_INPROGRESS;                               \
            SAVE_STATE(_phase);                                                \
            return;                                                            \
        }                                                                      \
        ucc_ee_executor_task_finalize(_etask);                                 \
        if (ucc_unlikely(status < 0)) {                                        \
            tl_error(UCC_TASK_LIB(task), _errmsg);                             \
            task->super.status = status;                                       \
            return;                                                            \
        }                                                                      \
    }                                                                          \
} while(0)


typedef struct ucc_tl_ucp_task {
    ucc_coll_task_t super;
    union {
        struct {
            uint32_t        send_posted;
            uint32_t        send_completed;
            uint32_t        recv_posted;
            uint32_t        recv_completed;
            uint32_t        tag;
        } tagged;
        struct {
            uint32_t        put_posted;
            uint32_t        put_completed;
            uint32_t        get_posted;
            uint32_t        get_completed;
        } onesided;
    };
    uint32_t        n_polls;
    ucc_subset_t    subset;
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
            ucc_ee_executor_task_t *etask;
        } allreduce_kn;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            void                   *scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
            ucc_ee_executor_task_t *etask;
        } reduce_scatter_kn;
        struct {
            void                   *scratch;
            size_t                  max_block_count;
            ucc_ep_map_t            inv_map;
            int                     n_frags;
            int                     frag;
            char                    s_scratch_busy[2];
        } reduce_scatter_ring;
        struct {
            void                   *scratch;
            size_t                  max_block_count;
            ucc_ep_map_t            inv_map;
            int                     n_frags;
            int                     frag;
            char                    s_scratch_busy[2];
        } reduce_scatterv_ring;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            ucc_rank_t              recv_dist;
            ptrdiff_t               send_offset;
        } scatter_kn;
        struct {
            int                     phase;
            ucc_knomial_pattern_t   p;
            void                   *sbuf;
            ucc_ee_executor_task_t *etask;
        } allgather_kn;
        struct {
            ucc_rank_t              dist;
            uint32_t                radix;
        } bcast_kn;
        struct {
            ucc_rank_t              dist;
            ucc_rank_t              max_dist;
            int                     children_per_cycle;
            uint32_t                radix;
            int                     phase;
            void                   *scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
        } reduce_kn;
        struct {
            ucc_rank_t              dist;
            ucc_rank_t              max_dist;
            uint32_t                radix;
            int                     phase;
            void *                  scratch;
            ucc_mc_buffer_header_t *scratch_mc_header;
        } gather_kn;
    };
} ucc_tl_ucp_task_t;

typedef struct ucc_tl_ucp_schedule {
    ucc_schedule_pipelined_t super;
    ucc_mc_buffer_header_t  *scratch_mc_header;
} ucc_tl_ucp_schedule_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_ucp_team_t))
#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_ucp_context_t))
#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_ucp_lib_t))
#define TASK_ARGS(_task) (_task)->super.bargs.args

static inline void ucc_tl_ucp_task_reset(ucc_tl_ucp_task_t *task,
                                         ucc_status_t status)
{
    task->tagged.send_posted    = 0;
    task->tagged.send_completed = 0;
    task->tagged.recv_posted    = 0;
    task->tagged.recv_completed = 0;
    task->super.status          = status;
}

static inline ucc_tl_ucp_task_t *ucc_tl_ucp_get_task(ucc_tl_ucp_team_t *team)
{
    ucc_tl_ucp_context_t *ctx  = UCC_TL_UCP_TEAM_CTX(team);
    ucc_tl_ucp_task_t    *task = ucc_mpool_get(&ctx->req_mp);;

    UCC_TL_UCP_PROFILE_REQUEST_NEW(task, "tl_ucp_task", 0);
    task->super.flags       = 0;
    task->n_polls           = ctx->cfg.n_polls;
    task->super.team        = &team->super.super;
    task->subset.map.type   = UCC_EP_MAP_FULL;
    task->subset.map.ep_num = UCC_TL_TEAM_SIZE(team);
    task->subset.myrank     = UCC_TL_TEAM_RANK(team);
    ucc_tl_ucp_task_reset(task, UCC_OPERATION_INITIALIZED);
    return task;
}

static inline void ucc_tl_ucp_put_task(ucc_tl_ucp_task_t *task)
{
    UCC_TL_UCP_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

static inline
ucc_tl_ucp_schedule_t *ucc_tl_ucp_get_schedule(ucc_tl_ucp_team_t *team,
                                               ucc_base_coll_args_t *args)
{
    ucc_tl_ucp_context_t  *ctx      = UCC_TL_UCP_TEAM_CTX(team);
    ucc_tl_ucp_schedule_t *schedule = ucc_mpool_get(&ctx->req_mp);

    UCC_TL_UCP_PROFILE_REQUEST_NEW(schedule, "tl_ucp_sched", 0);
    ucc_schedule_init(&schedule->super.super, args, &team->super.super);
    return schedule;
}

static inline void ucc_tl_ucp_put_schedule(ucc_schedule_t *schedule)
{
    UCC_TL_UCP_PROFILE_REQUEST_FREE(schedule);
    ucc_mpool_put(schedule);
}


ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *     team,
                                  ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task);

static inline ucc_tl_ucp_task_t *
ucc_tl_ucp_init_task(ucc_base_coll_args_t *coll_args, ucc_base_team_t *team)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task    = ucc_tl_ucp_get_task(tl_team);

    if (ucc_unlikely(!task)) {
        return NULL;
    }

    ucc_coll_task_init(&task->super, coll_args, team);

    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
        task->tagged.tag = (coll_args->mask & UCC_COLL_ARGS_FIELD_TAG)
            ? coll_args->args.tag : UCC_TL_UCP_ACTIVE_SET_TAG;
        task->subset.map    = ucc_active_set_to_ep_map(&coll_args->args);
        task->subset.myrank =
            ucc_ep_map_local_rank(task->subset.map,
                                  UCC_TL_TEAM_RANK(tl_team));
        ucc_assert(coll_args->args.coll_type == UCC_COLL_TYPE_BCAST);
        /* root value in args corresponds to the  original team ranks,
           need to convert to subset local value */
        TASK_ARGS(task).root = ucc_ep_map_local_rank(task->subset.map,
                                                     coll_args->args.root);
    } else {
        if (coll_args->mask & UCC_COLL_ARGS_FIELD_TAG) {
            task->tagged.tag = coll_args->args.tag;
        } else {
            tl_team->seq_num = (tl_team->seq_num + 1) % UCC_TL_UCP_MAX_COLL_TAG;
            task->tagged.tag = tl_team->seq_num;
        }
    }

    task->super.finalize       = ucc_tl_ucp_coll_finalize;
    task->super.triggered_post = ucc_triggered_post;
    return task;
}

#define UCC_TL_UCP_TASK_P2P_COMPLETE(_task)                                    \
    (((_task)->tagged.send_posted == (_task)->tagged.send_completed) &&        \
     ((_task)->tagged.recv_posted == (_task)->tagged.recv_completed))

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
        ucp_worker_progress(TASK_CTX(task)->ucp_worker);
    }
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_ucp_alg_id_to_init(int alg_id, const char *alg_id_str,
                                       ucc_coll_type_t          coll_type,
                                       ucc_memory_type_t        mem_type,
                                       ucc_base_coll_init_fn_t *init);


#endif
