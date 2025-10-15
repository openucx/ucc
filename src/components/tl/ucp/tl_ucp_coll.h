/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_COLL_H_
#define UCC_TL_UCP_COLL_H_

#include "tl_ucp.h"
#include "tl_ucp_task.h"
#include "coll_patterns/recursive_knomial.h"

#define UCC_UUNITS_AUTO_RADIX 4
#define UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR 9

ucc_status_t ucc_tl_ucp_team_default_score_str_alloc(ucc_tl_ucp_team_t *team,
    char *default_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR]);

void ucc_tl_ucp_team_default_score_str_free(
    char *default_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR]);

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
        _etask = NULL;                                                         \
        if (ucc_unlikely(status < 0)) {                                        \
            tl_error(UCC_TASK_LIB(task), _errmsg);                             \
            task->super.status = status;                                       \
            return;                                                            \
        }                                                                      \
    }                                                                          \
} while(0)

#define EXEC_TASK_WAIT(_etask, ...)                                            \
    do {                                                                       \
        if (_etask != NULL) {                                                  \
            do {                                                               \
                status = ucc_ee_executor_task_test(_etask);                    \
            } while (status > 0);                                              \
            if (status < 0) {                                                  \
                tl_error(UCC_TASK_LIB(task), "failure in ee task ee task");    \
                task->super.status = status;                                   \
                return __VA_ARGS__;                                            \
            }                                                                  \
            ucc_ee_executor_task_finalize(_etask);                             \
            if (ucc_unlikely(status < 0)) {                                    \
                tl_error(UCC_TASK_LIB(task), "failed to finalize ee task");    \
                task->super.status = status;                                   \
                return __VA_ARGS__;                                            \
            }                                                                  \
        }                                                                      \
    } while (0)

typedef char* (*ucc_tl_ucp_score_str_get_fn_t)(ucc_tl_ucp_team_t *team);
typedef struct ucc_tl_ucp_default_alg_desc {
    char                          *select_str;
    ucc_tl_ucp_score_str_get_fn_t  str_get_fn;
} ucc_tl_ucp_default_alg_desc_t;

#define AVG_ALPHA(_task) (1.0 / (double)UCC_TL_TEAM_SIZE(TASK_TEAM(_task)))

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *     team,
                                  ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task);

static inline ucc_tl_ucp_task_t *ucc_tl_ucp_init_task(ucc_base_coll_args_t *coll_args,
                                                      ucc_base_team_t *team)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task    = ucc_tl_ucp_get_task(tl_team);

    if (ucc_unlikely(!task)) {
        return NULL;
    }

    ucc_coll_task_init(&task->super, coll_args, team);

    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
        task->tagged.tag = (coll_args->args.mask & UCC_COLL_ARGS_FIELD_TAG)
            ? coll_args->args.tag : UCC_TL_UCP_ACTIVE_SET_TAG;
        task->flags        |= UCC_TL_UCP_TASK_FLAG_SUBSET;
        task->subset.map    = ucc_active_set_to_ep_map(&coll_args->args);
        task->subset.myrank =
            ucc_ep_map_local_rank(task->subset.map,
                                  UCC_TL_TEAM_RANK(tl_team));
        ucc_assert(coll_args->args.coll_type == UCC_COLL_TYPE_BCAST);
    } else {
        if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_TAG) {
            task->tagged.tag = coll_args->args.tag;
        } else {
            tl_team->seq_num = (tl_team->seq_num + 1) % UCC_TL_UCP_MAX_COLL_TAG;
            task->tagged.tag = tl_team->seq_num;
        }
    }

    task->super.finalize       = ucc_tl_ucp_coll_finalize;
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
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    }
    return UCC_INPROGRESS;
}

#define UCC_TL_UCP_TASK_RECV_COMPLETE(_task)                                   \
    (((_task)->tagged.recv_posted == (_task)->tagged.recv_completed))

#define UCC_TL_UCP_TASK_SEND_COMPLETE(_task)                                   \
    (((_task)->tagged.send_posted == (_task)->tagged.send_completed))

static inline ucc_status_t ucc_tl_ucp_test_recv(ucc_tl_ucp_task_t *task)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_RECV_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_RECV_COMPLETE(task)) {
            return UCC_OK;
        }
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    }
    return UCC_INPROGRESS;
}

static inline ucc_status_t ucc_tl_ucp_test_send(ucc_tl_ucp_task_t *task)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_SEND_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_SEND_COMPLETE(task)) {
            return UCC_OK;
        }
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    }
    return UCC_INPROGRESS;
}

#define UCC_TL_UCP_TASK_RING_P2P_COMPLETE(_task)                               \
    ((((_task)->tagged.send_posted - (_task)->tagged.send_completed) <= 1) &&  \
     ((_task)->tagged.recv_posted == (_task)->tagged.recv_completed))

static inline ucc_status_t ucc_tl_ucp_test_ring(ucc_tl_ucp_task_t *task)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_RING_P2P_COMPLETE(task)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_RING_P2P_COMPLETE(task)) {
            return UCC_OK;
        }
        ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
    }
    return UCC_INPROGRESS;
}

#define UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(_task)                           \
    (((_task)->onesided.put_posted == (_task)->onesided.put_completed) &&      \
     ((_task)->onesided.get_posted == (_task)->onesided.get_completed) &&      \
     ((_task)->flush_posted == (_task)->flush_completed))

#define UCC_TL_UCP_TASK_ONESIDED_SYNC_COMPLETE(_task, _end)                    \
    (*((long *)(TASK_ARGS(_task).global_work_buffer)) == _end)

static inline ucc_status_t ucc_tl_ucp_test_onesided(ucc_tl_ucp_task_t *task,
                                                    int                sync_end)
{
    int polls = 0;

    if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task) &&
        UCC_TL_UCP_TASK_ONESIDED_SYNC_COMPLETE(task, sync_end)) {
        return UCC_OK;
    }
    while (polls++ < task->n_polls) {
        if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task) &&
            UCC_TL_UCP_TASK_ONESIDED_SYNC_COMPLETE(task, sync_end)) {
            return UCC_OK;
        }
        ucp_worker_progress(UCC_TL_UCP_TASK_TEAM(task)->worker->ucp_worker);
    }
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_ucp_alg_id_to_init(int alg_id, const char *alg_id_str,
                                       ucc_coll_type_t          coll_type,
                                       ucc_memory_type_t        mem_type,
                                       ucc_base_coll_init_fn_t *init);

static inline unsigned
ucc_tl_ucp_get_radix_from_range(ucc_tl_ucp_team_t *team,
                                size_t             msgsize,
                                ucc_memory_type_t  mem_type,
                                ucc_mrange_uint_t *p,
                                ucc_rank_t         default_value)
{
    unsigned radix;

    radix = ucc_mrange_uint_get(p, msgsize, mem_type);

    if (UCC_UUNITS_AUTO == radix) {
        return default_value;
    }
    return radix;
}

/*
 * Get the radix for knomial patterns.
 * If need_scratch is true, the radix is the minimum radix that can be used to fit into scratch buffer.
 * Otherwise, the radix is the minimum radix that can be used to fit into team size.
 */
static inline unsigned ucc_tl_ucp_get_knomial_radix(ucc_tl_ucp_team_t *team,
                                                    size_t             count,
                                                    ucc_datatype_t     dtype,
                                                    ucc_memory_type_t  mem_type,
                                                    ucc_mrange_uint_t *p,
                                                    int need_scratch)
{
    size_t msgsize = count * ucc_dt_size(dtype);
    unsigned opt_radix, cfg_radix, radix;

    opt_radix = (mem_type == UCC_MEMORY_TYPE_HOST) ? team->opt_radix_host :
                                                    team->opt_radix;
    cfg_radix = ucc_tl_ucp_get_radix_from_range(team, msgsize, mem_type, p,
                                                opt_radix);
    if (need_scratch) {
        radix = ucc_knomial_pattern_get_min_radix(cfg_radix, UCC_TL_TEAM_SIZE(team), count);
    } else {
        radix = ucc_min(cfg_radix, UCC_TL_TEAM_SIZE(team));

    }
    return radix;
}

#endif
