/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_COLL_H_
#define UCC_TL_MLX5_COLL_H_

#include "tl_mlx5.h"
#include "schedule/ucc_schedule.h"

typedef struct ucc_tl_mlx5_task {
    ucc_coll_task_t super;
} ucc_tl_mlx5_task_t;

typedef struct ucc_tl_mlx5_schedule {
    ucc_schedule_t super;
    union {
        struct {
        } alltoall;
    };
} ucc_tl_mlx5_schedule_t;

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.super.team, ucc_tl_mlx5_team_t))

#define TASK_CTX(_task)                                                        \
    (ucc_derived_of((_task)->super.super.team->context, ucc_tl_mlx5_context_t))

#define TASK_LIB(_task)                                                        \
    (ucc_derived_of((_task)->super.super.team->context->lib, ucc_tl_mlx5_lib_t))

#define TASK_ARGS(_task) (_task)->super.super.bargs.args

#define TASK_SCHEDULE(_task)                                                   \
    (ucc_derived_of((_task)->schedule, ucc_tl_mlx5_schedule_t))

static inline ucc_tl_mlx5_task_t *
ucc_tl_mlx5_get_task(ucc_base_coll_args_t *coll_args, ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t *   tl_team = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_context_t *ctx     = UCC_TL_MLX5_TEAM_CTX(tl_team);
    ucc_tl_mlx5_task_t *   task    = ucc_mpool_get(&ctx->req_mp);

    UCC_TL_MLX5_PROFILE_REQUEST_NEW(task, "tl_mlx5_task", 0);
    ucc_coll_task_init(&task->super, coll_args, team);
    return task;
}

static inline void ucc_tl_mlx5_put_task(ucc_tl_mlx5_task_t *task)
{
    UCC_TL_MLX5_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

static inline ucc_tl_mlx5_schedule_t *
ucc_tl_mlx5_get_schedule(ucc_tl_mlx5_team_t *  team,
                         ucc_base_coll_args_t *coll_args)
{
    ucc_tl_mlx5_context_t * ctx      = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_tl_mlx5_schedule_t *schedule = ucc_mpool_get(&ctx->req_mp);

    UCC_TL_MLX5_PROFILE_REQUEST_NEW(schedule, "tl_mlx5_sched", 0);
    ucc_schedule_init(&schedule->super, coll_args, &team->super.super);
    return schedule;
}

static inline void ucc_tl_mlx5_put_schedule(ucc_tl_mlx5_schedule_t *schedule)
{
    UCC_TL_MLX5_PROFILE_REQUEST_FREE(schedule);
    ucc_mpool_put(schedule);
}

#endif
