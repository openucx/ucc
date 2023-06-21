/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_COLL_H_
#define UCC_TL_MLX5_COLL_H_

#include "tl_mlx5.h"
#include "schedule/ucc_schedule.h"
#include "alltoall/alltoall.h"

typedef struct ucc_tl_mlx5_task {
    ucc_coll_task_t super;
    union {
        struct {
            ucc_tl_mlx5_mcast_coll_req_t *req_handle;
        } coll_mcast;
    };
} ucc_tl_mlx5_task_t;

typedef struct ucc_tl_mlx5_schedule {
    ucc_schedule_t super;
    union {
        struct {
            int                          seq_num;
            int                          seq_index;
            int                          num_of_blocks_columns;
            int                          block_height;
            int                          block_width;
            int                          started;
            int                          send_blocks_enqueued;
            int                          blocks_sent;
            int                          blocks_completed;
            ucc_tl_mlx5_alltoall_op_t   *op;
            ucc_tl_mlx5_rcache_region_t *send_rcache_region_p;
            ucc_tl_mlx5_rcache_region_t *recv_rcache_region_p;
            size_t                       msg_size;
            ucc_service_coll_req_t      *barrier_req;
            int                          barrier_scratch[2];
            int                          wait_wc;
        } alltoall;
    };
} ucc_tl_mlx5_schedule_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_mlx5_team_t))

#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_mlx5_context_t))

#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_mlx5_lib_t))

#define TASK_ARGS(_task) (_task)->super.bargs.args

#define TASK_SCHEDULE(_task)                                                   \
    (ucc_derived_of((_task)->schedule, ucc_tl_mlx5_schedule_t))

#define SCHEDULE_TEAM(_schedule)                                               \
    (ucc_derived_of((_schedule)->super.super.team, ucc_tl_mlx5_team_t))

#define SCHEDULE_CTX(_schedule)                                                \
    (ucc_derived_of((_schedule)->super.super.team->context,                    \
                    ucc_tl_mlx5_context_t))

#define SCHEDULE_LIB(_schedule)                                                \
    (ucc_derived_of((_schedule)->super.super.team->context->lib,               \
                    ucc_tl_mlx5_lib_t))

#define SCHEDULE_ARGS(_schedule) (_schedule)->super.super.bargs.args

static inline ucc_tl_mlx5_task_t*
ucc_tl_mlx5_get_task(ucc_base_coll_args_t *coll_args, ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t *   tl_team = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_context_t *ctx     = UCC_TL_MLX5_TEAM_CTX(tl_team);
    ucc_tl_mlx5_task_t *   task    = ucc_mpool_get(&ctx->req_mp);

    UCC_TL_MLX5_PROFILE_REQUEST_NEW(task, "tl_mlx5_task", 0);
    ucc_coll_task_init(&task->super, coll_args, team);
    task->coll_mcast.req_handle = NULL;
    return task;
}

static inline void ucc_tl_mlx5_put_task(ucc_tl_mlx5_task_t *task)
{
    UCC_TL_MLX5_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

static inline ucc_status_t
ucc_tl_mlx5_get_schedule(ucc_tl_mlx5_team_t      *team,
                         ucc_base_coll_args_t    *coll_args,
                         ucc_tl_mlx5_schedule_t **schedule)
{
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);

    *schedule = ucc_mpool_get(&ctx->req_mp);

    if (ucc_unlikely(!(*schedule))) {
        return UCC_ERR_NO_MEMORY;
    }
    UCC_TL_MLX5_PROFILE_REQUEST_NEW(schedule, "tl_mlx5_sched", 0);

    return ucc_schedule_init(&((*schedule)->super), coll_args,
                             &team->super.super);
}

static inline void ucc_tl_mlx5_put_schedule(ucc_tl_mlx5_schedule_t *schedule)
{
    UCC_TL_MLX5_PROFILE_REQUEST_FREE(schedule);
    ucc_mpool_put(schedule);
}

ucc_status_t ucc_tl_mlx5_coll_mcast_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t      *team,
                                         ucc_coll_task_t     **task_h);

ucc_status_t ucc_tl_mlx5_task_finalize(ucc_coll_task_t *coll_task);

ucc_tl_mlx5_task_t* ucc_tl_mlx5_init_task(ucc_base_coll_args_t *coll_args,
                                          ucc_base_team_t      *team,
                                          ucc_schedule_t       *schedule);

ucc_status_t ucc_tl_mlx5_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task_h);

#endif
