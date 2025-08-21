/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_COLL_H_
#define UCC_TL_CUDA_COLL_H_

#include "tl_cuda.h"
#include "components/mc/ucc_mc.h"

#define UCC_TL_CUDA_N_DEFAULT_ALG_SELECT_STR 6
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

#define NVLS_CONTROL_SIZE 1024

#define TASK_SYMMETRIC_MC(_task)                                                   \
    ({                                                                             \
        ucc_tl_cuda_team_t *_team = TASK_TEAM(_task);                              \
        size_t _symm_size = UCC_TL_CUDA_TEAM_LIB(_team)->cfg.nvls_symmetric_size + NVLS_CONTROL_SIZE;  \
        (PTR_OFFSET(_team->nvls.mc_va, (_task)->coll_id * _symm_size));            \
    })

#define TASK_SYMMETRIC_UC(_task)                                                   \
    ({                                                                             \
        ucc_tl_cuda_team_t *_team = TASK_TEAM(_task);                              \
        size_t _symm_size = UCC_TL_CUDA_TEAM_LIB(_team)->cfg.nvls_symmetric_size + NVLS_CONTROL_SIZE;  \
        (PTR_OFFSET(_team->nvls.uc_va, (_task)->coll_id * _symm_size));            \
    })

#define TASK_NVLS_CONTROL_MC(_task)                                                 \
    ({                                                                             \
        ucc_tl_cuda_team_t *_team = TASK_TEAM(_task);                              \
        size_t _symm_payload_size = UCC_TL_CUDA_TEAM_LIB(_team)->cfg.nvls_symmetric_size;  \
        size_t _symm_size = _symm_payload_size + NVLS_CONTROL_SIZE;  \
        ((CUdeviceptr) PTR_OFFSET(_team->nvls.mc_va, (_task)->coll_id * _symm_size + _symm_payload_size));            \
    })

#define TASK_NVLS_CONTROL_UC(_task)                                                 \
    ({                                                                             \
        ucc_tl_cuda_team_t *_team = TASK_TEAM(_task);                              \
        size_t _symm_payload_size = UCC_TL_CUDA_TEAM_LIB(_team)->cfg.nvls_symmetric_size;  \
        size_t _symm_size = _symm_payload_size + NVLS_CONTROL_SIZE;  \
        ((CUdeviceptr) PTR_OFFSET(_team->nvls.uc_va, (_task)->coll_id * _symm_size + _symm_payload_size));            \
    })

static inline void ucc_tl_cuda_task_reset(ucc_tl_cuda_task_t *task)
{
    task->super.status = UCC_INPROGRESS;
}

ucc_status_t ucc_tl_cuda_shm_barrier_init_root(ucc_rank_t size, ucc_rank_t rank, ucc_rank_t root,
                                          ucc_tl_cuda_shm_barrier_t *barrier);

ucc_status_t ucc_tl_cuda_shm_barrier_init(ucc_rank_t size, ucc_rank_t rank,
                                          ucc_tl_cuda_shm_barrier_t *barrier);

ucc_status_t ucc_tl_cuda_shm_barrier_start(ucc_rank_t rank,
                                           ucc_tl_cuda_shm_barrier_t *barrier);

ucc_status_t ucc_tl_cuda_shm_barrier_test(ucc_rank_t rank,
                                          ucc_tl_cuda_shm_barrier_t *barrier);


static inline ucc_tl_cuda_task_t *ucc_tl_cuda_task_get(ucc_tl_cuda_team_t *team)
{
    ucc_tl_cuda_context_t *ctx  = UCC_TL_CUDA_TEAM_CTX(team);
    ucc_tl_cuda_task_t    *task = ucc_mpool_get(&ctx->req_mp);

    if (ucc_unlikely(!task)) {
        tl_error(UCC_TL_CUDA_TEAM_LIB(team), "failed to get task from mpool");
        return NULL;
    }

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

static inline uint64_t compute_key(ucc_rank_t root, ucc_rank_t peer, uint16_t tag)
{
    assert(peer < (1 << 24));
    assert(root < (1 << 24));
    return (uint64_t)tag << 48 | root << 24 | peer;
}

static inline
ucc_status_t ucc_tl_cuda_task_init(ucc_base_coll_args_t *coll_args,
                                   ucc_tl_cuda_team_t *team,
                                   ucc_tl_cuda_task_t **task_h)
{
    ucc_rank_t          trank          = UCC_TL_TEAM_RANK(team);
    ucc_tl_cuda_lib_t  *lib            = UCC_TL_CUDA_TEAM_LIB(team);
    uint32_t            max_concurrent = lib->cfg.max_concurrent;
    ucc_rank_t          peer;
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (!ucc_coll_args_is_predefined_dt(&coll_args->args, trank)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_cuda_task_get(team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_coll_task_init(&task->super, coll_args, &team->super.super);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_cuda_task_put(task);
        return status;
    }

    /* active set */
    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
        ucc_assert(coll_args->args.coll_type == UCC_COLL_TYPE_BCAST);
        task->subset.map    = ucc_active_set_to_ep_map(&coll_args->args);
        task->subset.myrank = UCC_TL_TEAM_RANK(team);
        // currently we support only active set bacst with 2 ranks
        // so root rank should remap phys rank of peer with rank 1
        peer = (task->subset.myrank == coll_args->args.root) ? ucc_ep_map_eval(task->subset.map, 1) : task->subset.myrank;
        task->bcast_linear.key = compute_key(coll_args->args.root, peer, coll_args->args.tag);
        task->seq_num = team->seq_num_active_set++;
    } else {
        task->seq_num = team->seq_num++;
        task->coll_id = task->seq_num % max_concurrent;
        task->bar     = TASK_BAR(task);
    }

    *task_h = task;
    return UCC_OK;
}

// check if segment for current task is available and barrier is available (completed from prev iteration)
// and possibly mark the segment as occupied by updating the state counter to the current seq_num
static inline ucc_status_t ucc_tl_cuda_get_sync_root(ucc_tl_cuda_task_t *task, ucc_rank_t root)
{
    ucc_tl_cuda_team_t                *team  = TASK_TEAM(task);
    volatile ucc_tl_cuda_sync_state_t *state = &team->sync_state[task->coll_id];

    if ((UCC_TL_TEAM_RANK(team) == root) && (*state == 0)) {
        *state = task->seq_num;
    }
    if ((*state != task->seq_num) ||
        (task->bar->state[UCC_TL_TEAM_RANK(team)] != UCC_OK)) {
        return UCC_INPROGRESS;
    }
    return UCC_OK;
}

static inline void ucc_tl_cuda_put_sync_root(ucc_tl_cuda_task_t *task, ucc_rank_t root)
{
    ucc_tl_cuda_team_t       *team  = TASK_TEAM(task);
    ucc_tl_cuda_sync_state_t *state = &team->sync_state[task->coll_id];

    if (UCC_TL_TEAM_RANK(team) == root) {
        ucc_assert(*state == task->seq_num);
        *state = 0;
    }
}

static inline ucc_status_t ucc_tl_cuda_get_sync(ucc_tl_cuda_task_t *task)
{
    return ucc_tl_cuda_get_sync_root(task, 0);
}

static inline void ucc_tl_cuda_put_sync(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_put_sync_root(task, 0);
}

ucc_status_t ucc_tl_cuda_mem_info_get(void *ptr, size_t length,
                                      ucc_tl_cuda_mem_info_t *mi);

ucc_status_t ucc_tl_cuda_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_cuda_coll_finalize(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_cuda_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t          coll_type,
                                        ucc_memory_type_t        mem_type,
                                        ucc_base_coll_init_fn_t *init);

// common utils function for collectives:
static inline int get_rank_step(ucc_tl_cuda_task_t *task, ucc_rank_t rank,
                                int step_id)
{
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);

    return sync->seq_num[step_id];
}

static inline void set_rank_step(ucc_tl_cuda_task_t *task, ucc_rank_t rank,
                                 int step, int step_id)
{
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);

    sync->seq_num[step_id] = step;
}

#endif
