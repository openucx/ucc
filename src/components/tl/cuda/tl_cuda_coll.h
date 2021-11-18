/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_COLL_H_
#define UCC_TL_CUDA_COLL_H_

#include "tl_cuda.h"
#include "core/ucc_mc.h"

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_cuda_team_t))

#define TASK_ARGS(_task) (_task)->super.bargs.args

#define TASK_SYNC(_task, _rank)                                                \
    ({                                                                         \
        ucc_tl_cuda_team_t *_team = TASK_TEAM(_task);                          \
        UCC_TL_CUDA_TEAM_SYNC(_team, _rank, (_task)->coll_id);                 \
    })

static inline void ucc_tl_cuda_task_reset(ucc_tl_cuda_task_t *task)
{
    task->super.super.status = UCC_INPROGRESS;
}

static inline ucc_tl_cuda_task_t *ucc_tl_cuda_task_get(ucc_tl_cuda_team_t *team)
{
    ucc_tl_cuda_context_t *ctx  = UCC_TL_CUDA_TEAM_CTX(team);
    ucc_tl_cuda_task_t    *task = ucc_mpool_get(&ctx->req_mp);

    UCC_TL_CUDA_PROFILE_REQUEST_NEW(task, "tl_cuda_task", 0);
    task->super.super.status = UCC_OPERATION_INITIALIZED;
    task->super.flags        = 0;
    task->super.team         = &team->super.super;
    ucc_tl_cuda_task_reset(task);
    return task;
}

static inline void ucc_tl_cuda_task_put(ucc_tl_cuda_task_t *task)
{
    UCC_TL_CUDA_PROFILE_REQUEST_FREE(task);
    ucc_mpool_put(task);
}

static inline ucc_tl_cuda_task_t *
ucc_tl_cuda_task_init(ucc_base_coll_args_t *coll_args,
                      ucc_tl_cuda_team_t *team)
{
    ucc_tl_cuda_task_t *task = ucc_tl_cuda_task_get(team);
    uint32_t max_concurrent;

    max_concurrent = UCC_TL_CUDA_TEAM_LIB(team)->cfg.max_concurrent;
    ucc_coll_task_init(&task->super, coll_args, &team->super.super);
    task->seq_num = team->seq_num++;
    task->coll_id = task->seq_num % max_concurrent;
    return task;
}

ucc_status_t ucc_tl_cuda_mem_info_get(void *ptr, size_t length,
                                      ucc_tl_cuda_team_t *team,
                                      ucc_tl_cuda_mem_info_t *mi);

ucc_status_t ucc_tl_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_cuda_coll_finalize(ucc_coll_task_t *coll_task);

#endif
