/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MHBA_COLL_H_
#define UCC_TL_MHBA_COLL_H_

#include "tl_mhba.h"
#include "schedule/ucc_schedule.h"

#define TMP_TRANSPOSE_PREALLOC 256

typedef struct ucc_tl_mhba_task {
    ucc_coll_task_t super;
} ucc_tl_mhba_task_t;

typedef struct ucc_tl_mhba_schedule {
    ucc_schedule_t         super;
    int                 seq_num;
    int                 seq_index;
    int                 num_of_blocks_columns;
    int                 block_size;
    int                 started;
    ucc_tl_mhba_op_t   *op;
    ucc_tl_mhba_reg_t * send_rcache_region_p;
    ucc_tl_mhba_reg_t * recv_rcache_region_p;
    struct ibv_mr *     transpose_buf_mr;
    void *              tmp_transpose_buf;
    size_t              msg_size;
    ucc_service_coll_req_t *barrier_req;
    int                 barrier_scratch[2];
} ucc_tl_mhba_schedule_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.super.team, ucc_tl_mhba_team_t))

#define TASK_CTX(_task)                                                 \
    (ucc_derived_of((_task)->super.super.team->context, ucc_tl_mhba_context_t))

#define TASK_LIB(_task)                                                 \
    (ucc_derived_of((_task)->super.super.team->context->lib, ucc_tl_mhba_lib_t))

#define TASK_ARGS(_task) (_task)->super.super.bargs.args

#define TASK_SCHEDULE(_task)                                                 \
    (ucc_derived_of((_task)->schedule, ucc_tl_mhba_schedule_t))

static inline ucc_tl_mhba_task_t *ucc_tl_mhba_get_task(ucc_base_coll_args_t *coll_args,
                                                       ucc_base_team_t *team)
{
    ucc_tl_mhba_team_t *tl_team = ucc_derived_of(team, ucc_tl_mhba_team_t);
    ucc_tl_mhba_context_t *ctx  = UCC_TL_MHBA_TEAM_CTX(tl_team);
    ucc_tl_mhba_task_t *   task = ucc_mpool_get(&ctx->req_mp);

    UCC_TL_MHBA_PROFILE_REQUEST_NEW(task, "tl_mhba_task", 0);
    ucc_coll_task_init(&task->super, coll_args, team);
    return task;
}

static inline void ucc_tl_mhba_put_task(ucc_tl_mhba_task_t *task)
{
    UCC_TL_MHBA_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

static inline ucc_tl_mhba_schedule_t *
ucc_tl_mhba_get_schedule(ucc_tl_mhba_team_t *team,
                         ucc_base_coll_args_t *coll_args)
{
    ucc_tl_mhba_context_t * ctx      = UCC_TL_MHBA_TEAM_CTX(team);
    ucc_tl_mhba_schedule_t *schedule = ucc_mpool_get(&ctx->req_mp);

    UCC_TL_MHBA_PROFILE_REQUEST_NEW(schedule, "tl_mhba_sched", 0);
    ucc_schedule_init(&schedule->super, coll_args, &team->super.super);
    return schedule;
}

static inline void ucc_tl_mhba_put_schedule(ucc_tl_mhba_schedule_t *schedule)
{
    UCC_TL_MHBA_PROFILE_REQUEST_FREE(schedule);
    ucc_mpool_put(schedule);
}

ucc_status_t ucc_tl_mhba_alltoall_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *     team,
                                       ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_mhba_task_finalize(ucc_coll_task_t *coll_task);

#endif
