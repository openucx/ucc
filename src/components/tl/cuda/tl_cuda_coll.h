/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_COLL_H_
#define UCC_TL_CUDA_COLL_H_

#include "tl_cuda.h"
#include "components/mc/ucc_mc.h"

#define UCC_TL_CUDA_N_DEFAULT_ALG_SELECT_STR 2
extern const char
    *ucc_tl_cuda_default_alg_select_str[UCC_TL_CUDA_N_DEFAULT_ALG_SELECT_STR];

#define TASK_TEAM(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_cuda_team_t))

#define TASK_ARGS(_task) (_task)->super.bargs.args

#define TASK_SYNC(_task, _rank)                                                \
    ({                                                                         \
        ucc_tl_cuda_team_t *_team = TASK_TEAM(_task);                          \
        UCC_TL_CUDA_TEAM_SYNC(_team, _rank, (_task)->coll_id);                 \
    })

#define TASK_BAR(_task)                                                        \
    ({                                                                         \
        ucc_tl_cuda_team_t *_team = TASK_TEAM(_task);                          \
        UCC_TL_CUDA_TEAM_BARRIER(_team, (_task)->coll_id);                     \
    })

#define TASK_SCRATCH(_task, _rank)                                             \
    ({                                                                         \
        ucc_tl_cuda_team_t *_team = TASK_TEAM(_task);                          \
        size_t _scratch_size = UCC_TL_CUDA_TEAM_LIB(_team)->cfg.scratch_size;  \
        void *_scratch;                                                        \
        if (_rank == UCC_TL_TEAM_RANK(_team)) {                                \
            _scratch = _team->scratch.loc;                                     \
        } else {                                                               \
            _scratch = PTR_OFFSET(_team->scratch.rem[_rank],                   \
                                  _team->scratch.rem_info[_rank].offset);      \
        }                                                                      \
        (PTR_OFFSET(_scratch, (_task)->coll_id * _scratch_size));              \
    })

static inline void ucc_tl_cuda_task_reset(ucc_tl_cuda_task_t *task)
{
    task->super.status = UCC_INPROGRESS;
}

static inline ucc_tl_cuda_task_t *ucc_tl_cuda_task_get(ucc_tl_cuda_team_t *team)
{
    ucc_tl_cuda_context_t *ctx  = UCC_TL_CUDA_TEAM_CTX(team);
    ucc_tl_cuda_task_t    *task = ucc_mpool_get(&ctx->req_mp);

    UCC_TL_CUDA_PROFILE_REQUEST_NEW(task, "tl_cuda_task", 0);
    task->super.status = UCC_OPERATION_INITIALIZED;
    task->super.flags  = 0;
    task->super.team   = &team->super.super;
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

static inline ucc_status_t ucc_tl_cuda_get_sync(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t       *team  = TASK_TEAM(task);
    ucc_tl_cuda_sync_state_t *state = &team->sync_state[task->coll_id];

    if ((UCC_TL_TEAM_RANK(team) == 0) && (*state == 0)) {
        *state = task->seq_num;
    }
    if ((*state != task->seq_num) ||
        (task->bar->state[UCC_TL_TEAM_RANK(team)] != UCC_OK)) {
        return UCC_INPROGRESS;
    }
    return UCC_OK;
}

static inline void ucc_tl_cuda_put_sync(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t       *team  = TASK_TEAM(task);
    ucc_tl_cuda_sync_state_t *state = &team->sync_state[task->coll_id];

    if (UCC_TL_TEAM_RANK(team) == 0) {
        ucc_assert(*state == task->seq_num);
        *state = 0;
    }
}

ucc_status_t ucc_tl_cuda_mem_info_get(void *ptr, size_t length,
                                      ucc_tl_cuda_mem_info_t *mi);

ucc_status_t ucc_tl_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_cuda_coll_finalize(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_cuda_shm_barrier_init(ucc_rank_t size, ucc_rank_t rank,
                                          ucc_tl_cuda_shm_barrier_t *barrier);

ucc_status_t ucc_tl_cuda_shm_barrier_start(ucc_rank_t rank,
                                           ucc_tl_cuda_shm_barrier_t *barrier);

ucc_status_t ucc_tl_cuda_shm_barrier_test(ucc_rank_t rank,
                                          ucc_tl_cuda_shm_barrier_t *barrier);

ucc_status_t ucc_tl_cuda_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t          coll_type,
                                        ucc_memory_type_t        mem_type,
                                        ucc_base_coll_init_fn_t *init);

#endif
