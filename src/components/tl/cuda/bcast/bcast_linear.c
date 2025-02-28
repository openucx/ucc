/**
 * Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast.h"

enum {
    // Barrier setup stages
    STAGE_INIT_BAR_ROOT,          // Initial stage for the root rank to identify and claim a free barrier
    STAGE_FIND_BAR_PEER,          // Stage where peer ranks wait while the root rank identifies a free barrier

    STAGE_SYNC,                   // Initialize the barrier and synchronize the segment required for the current task
    STAGE_SETUP,                  // Verify that all ranks are aligned and have reached the barrier
    // Stages specific to the root rank
    STAGE_COPY,                   // Post copy task: copy data block from src to a scratch buffer
    STAGE_WAIT_COPY,              // The root waits for the completion of its copy operation
    STAGE_WAIT_ALL,               // The root rank waits until all other ranks have reached the same operational step
    STAGE_WAIT_COMPLETION,        // The root rank waits for all other ranks to complete the broadcast operation
    // non-root
    STAGE_WAIT_ROOT,              // Wait while the root rank writes data to its scratch buffer
    STAGE_CLIENT_COPY,            // Initiate their own copy tasks after the root's operations
    STAGE_CLIENT_COPY_WAIT,       // Wait for the completion of the copy operation from the root's scratch buffer
    STAGE_CLIENT_WAIT_COMPLETION, // Wait for the completion of algorithm on all ranks, global sync with root
};

static inline ucc_status_t
ucc_tl_cuda_bcast_linear_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);

    set_rank_step(task, trank, 0, 0); // Initialize rank step tracking
    ucc_memory_cpu_store_fence();
    // initiate barrier wait while all ranks set theirs steps to 0
    return ucc_tl_cuda_shm_barrier_start(trank, task->bar);
}

// Tests if setup is complete for a linear broadcast task
static inline ucc_status_t
ucc_tl_cuda_bcast_linear_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

// Returns the size of the scratch buffer used for data transfers
static inline size_t get_raw_scratch_size(ucc_tl_cuda_team_t *team)
{
    return UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
}

// Posts a copy task to the CUDA executor
static inline ucc_status_t ecopy(void *dst, void *src, size_t size,
                                 ucc_ee_executor_t       *exec,
                                 ucc_ee_executor_task_t **etask)
{
    ucc_ee_executor_task_args_t exec_args = {0};

    exec_args.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    exec_args.copy.dst  = dst;
    exec_args.copy.src  = src;
    exec_args.copy.len  = size;
    return ucc_ee_executor_task_post(exec, &exec_args, etask);
}

// Root rank searches for and claims a free barrier
static inline ucc_status_t root_find_free_barrier(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t        *team            = TASK_TEAM(task);
    uint32_t                   max_concurrent  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.max_concurrent;
    ucc_tl_cuda_shm_barrier_t *curr_bar;
    int                        i;
    ucc_status_t               st;

    // Iterate over available barriers in active set pool to find a free one
    for (i = 0; i < max_concurrent; ++i) {
        curr_bar = UCC_TL_CUDA_TEAM_BARRIER(team, max_concurrent + i);
        // try to set user specified tag to mark that this barrier is used by this task
        if (ucc_atomic_cswap64(&curr_bar->tag, UCC_TL_CUDA_TAG_FREE,
                               task->bcast_linear.key) == UCC_TL_CUDA_TAG_FREE) {
            ucc_debug("Acquire barrier: %p idx: %d marked with tag: %ld",
                      curr_bar, i, curr_bar->tag);
            task->bar = curr_bar;
            st        = ucc_tl_cuda_shm_barrier_init_root(
                task->subset.map.ep_num, task->subset.myrank,
                task->bcast_linear.root, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to init root barrier");
                return UCC_ERR_NO_RESOURCE;
            }
            // Assign a collective ID (index of barrier)
            task->coll_id = i + max_concurrent;
            return UCC_OK;
        }
    }
    // try next time
    return UCC_ERR_NOT_FOUND;
}

// Peer rank searches for a barrier claimed by the root
static inline ucc_status_t peer_find_free_barrier(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    uint32_t max_concurrent  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.max_concurrent;
    ucc_tl_cuda_shm_barrier_t *curr_bar;
    int                        i;
    ucc_status_t               st;

    for (i = 0; i < max_concurrent; ++i) {
        curr_bar = UCC_TL_CUDA_TEAM_BARRIER(team, max_concurrent + i);
        // Check if the barrier is claimed by the task's root
        if (curr_bar->tag == task->bcast_linear.key) {
            task->bar = curr_bar;
            st        = ucc_tl_cuda_shm_barrier_init_root(
                task->subset.map.ep_num, task->subset.myrank,
                task->bcast_linear.root, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to init peer barrier");
                return UCC_ERR_NO_RESOURCE;
            }
            task->coll_id = i + max_concurrent;
            return UCC_OK;
        }
    }
    // try next time
    return UCC_ERR_NOT_FOUND;
}

static ucc_status_t
ucc_tl_cuda_bcast_linear_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

static void ucc_tl_cuda_bcast_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task              = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team              = TASK_TEAM(task);
    ucc_rank_t          trank             = UCC_TL_TEAM_RANK(team);
    size_t              half_scratch_size = get_raw_scratch_size(team) / 2;
    ucc_rank_t          tsize             = UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))
                                            ? (ucc_rank_t)task->subset.map.ep_num
                                            : UCC_TL_TEAM_SIZE(team);
    size_t              chunk_size        = ucc_min(
        half_scratch_size,
        task->bcast_linear.size - task->bcast_linear.step * half_scratch_size);
    size_t              offset_buff       = task->bcast_linear.step * half_scratch_size;
    ucc_ee_executor_t         *exec;
    ucc_ee_executor_task_t    *etask;
    void                      *sbuf, *dbuf;
    ucc_rank_t                 peer;
    ucc_status_t               st;
    int                        i;

    task->super.status = UCC_INPROGRESS;

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        task->super.status = st;
        return;
    }

    switch (task->bcast_linear.stage) {
    case STAGE_INIT_BAR_ROOT:
        st = root_find_free_barrier(task);
        if (st == UCC_OK) {
            task->bcast_linear.stage = STAGE_SYNC;
        } else if (st != UCC_ERR_NOT_FOUND) {
            task->super.status = st;
        }
        // no free barriers found, try next time
        return;
    case STAGE_FIND_BAR_PEER:
        st = peer_find_free_barrier(task);
        if (st == UCC_OK) {
            // barrier found, continue to next stages
            task->bcast_linear.stage = STAGE_SYNC;
        } else if (st != UCC_ERR_NOT_FOUND) {
            task->super.status = st;
        }
        // no free barriers found by root, try next time
        return;
    case STAGE_SYNC:
        if (ucc_tl_cuda_get_sync_root(task, task->bcast_linear.root) != UCC_OK) {
            return;
        }
        task->bcast_linear.step = 0;
        st                      = ucc_tl_cuda_bcast_linear_setup_start(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->bcast_linear.stage = STAGE_SETUP;
        /* fall through */
    case STAGE_SETUP:
        st = ucc_tl_cuda_bcast_linear_setup_test(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        if (trank == task->bcast_linear.root) {
            task->bcast_linear.stage = STAGE_COPY;
        } else {
            task->bcast_linear.stage = STAGE_WAIT_ROOT;
        }
        /* fall through */
    default:
        break;
    }

    if (trank == task->bcast_linear.root) {
        // Root scenario
        // fall-through between cases is intentional
        switch (task->bcast_linear.stage) {
        case STAGE_COPY:
            // copy from src buffer to scratch
            dbuf = PTR_OFFSET(TASK_SCRATCH(task, trank),
                              task->bcast_linear.step % 2 * half_scratch_size);
            sbuf = PTR_OFFSET(task->bcast_linear.sbuf, offset_buff);
            st   = ecopy(dbuf, sbuf, chunk_size, exec,
                       &task->bcast_linear.exec_task);
            if (st != UCC_OK) {
                ucc_error("failed to post ecopy task");
                task->super.status = st;
                return;
            }
            task->bcast_linear.stage = STAGE_WAIT_COPY;
            /* fall through */
        case STAGE_WAIT_COPY:
            etask = task->bcast_linear.exec_task;
            ucc_assert(NULL != etask);
            st = ucc_ee_executor_task_test(etask);
            if (st != UCC_OK) {
                return; // not ready
            }
            ucc_ee_executor_task_finalize(etask);
            task->bcast_linear.exec_task = NULL;
            // signal others
            ++task->bcast_linear.step;
            set_rank_step(task, task->bcast_linear.root,
                          task->bcast_linear.step, 0);
            task->bcast_linear.stage = STAGE_WAIT_ALL;
            /* fall through */
        case STAGE_WAIT_ALL:
            for (i = 0; i < tsize; ++i) {
                if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))) {
                    // eval phys rank from virt
                    peer = ucc_ep_map_eval(task->subset.map, i);
                } else {
                    peer = i;
                }
                // need to wait until all ranks complete step - 1, because of double buffering
                if (get_rank_step(task, peer, 0) <
                    task->bcast_linear.step - 1) {
                    // rank is not ready, lets wait
                    return;
                }
            }
            if (task->bcast_linear.step < task->bcast_linear.num_steps) {
                // go to next iteration
                task->bcast_linear.stage = STAGE_COPY;
                return;
            }
            // finish
            st = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to start barrier from root rank");
                task->super.status = st;
                return;
            }
            task->bcast_linear.stage = STAGE_WAIT_COMPLETION;
            /* fall through */
        case STAGE_WAIT_COMPLETION:
            st = ucc_tl_cuda_shm_barrier_test(trank, task->bar);
            if (st != UCC_OK) {
                // peers still working, lets check next time
                task->super.status = st;
                return;
            }
            // set barrier free to unlock others, this is roots responsibility
            ucc_debug("Release bar: %p with tag: %ld", task->bar,
                      task->bar->tag);
            task->bar->tag = UCC_TL_CUDA_TAG_FREE;
            ucc_tl_cuda_put_sync_root(task, task->bcast_linear.root);
            task->super.status = UCC_OK;
            break;
        default:
            ucc_assert(0);
            break;
        }
    } else {
        // clients
        // fall-through between cases is intentional
        switch (task->bcast_linear.stage) {
        case STAGE_WAIT_ROOT:
            if (get_rank_step(task, task->bcast_linear.root, 0) >
                task->bcast_linear.step) {
                task->bcast_linear.stage = STAGE_CLIENT_COPY;
                break;
            } else {
                return;
            }
            /* fall through */
        case STAGE_CLIENT_COPY:
            // need to copy from root's scratch buffer
            dbuf = PTR_OFFSET(task->bcast_linear.sbuf, offset_buff);
            sbuf = PTR_OFFSET(TASK_SCRATCH(task, task->bcast_linear.root),
                              task->bcast_linear.step % 2 * half_scratch_size);
            st   = ecopy(dbuf, sbuf, chunk_size, exec,
                       &task->bcast_linear.exec_task);
            if (st != UCC_OK) {
                ucc_error("failed to post ecopy task at client");
                task->super.status = st;
                return;
            }
            task->bcast_linear.stage = STAGE_CLIENT_COPY_WAIT;
            /* fall through */
        case STAGE_CLIENT_COPY_WAIT:
            etask = task->bcast_linear.exec_task;
            ucc_assert(NULL != etask);
            st = ucc_ee_executor_task_test(etask);
            if (st != UCC_OK) {
                return; // executor task is not ready
            }
            ucc_ee_executor_task_finalize(etask);
            task->bcast_linear.exec_task = NULL;
            ++task->bcast_linear.step;
            set_rank_step(task, trank, task->bcast_linear.step, 0);
            if (task->bcast_linear.step < task->bcast_linear.num_steps) {
                task->bcast_linear.stage = STAGE_WAIT_ROOT;
                return;
            }
            // start barrier to sync with root
            st = ucc_tl_cuda_shm_barrier_start(trank, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to start barrier from peer rank");
                task->super.status = st;
                return;
            }
            task->bcast_linear.stage = STAGE_CLIENT_WAIT_COMPLETION;
            /* fall through */
        case STAGE_CLIENT_WAIT_COMPLETION:
            st = ucc_tl_cuda_shm_barrier_test(trank, task->bar);
            if (st != UCC_OK) {
                // someone still working, lets check next time
                task->super.status = st;
                return;
            }
            task->super.status = UCC_OK;
            break;
        default:
            ucc_assert(0);
            break;
        }
    }
}

static ucc_status_t ucc_tl_cuda_bcast_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);

    task->bcast_linear.step  = 0;
    task->bcast_linear.stage = STAGE_SYNC;
    // in case of active set bcast we need to do additional steps to find free barriers
    if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))) {
        task->bcast_linear.stage =
            UCC_TL_TEAM_RANK(team) == task->bcast_linear.root
                ? STAGE_INIT_BAR_ROOT
                : STAGE_FIND_BAR_PEER;
    }

    ucc_debug("bcast linear start dt: %s, buffer size: %ld, num_steps: %d",
              ucc_datatype_str(task->bcast_linear.dt), task->bcast_linear.size,
              task->bcast_linear.num_steps);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_cuda_bcast_linear_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *tl_team,
                                           ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (!ucc_tl_cuda_team_topo_is_fully_connected(team->topo) ||
        UCC_TL_TEAM_SIZE(team) - 1 > UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->bcast_linear.root = coll_args->args.root;
    task->bcast_linear.dt   = coll_args->args.src.info.datatype;
    task->bcast_linear.sbuf = coll_args->args.src.info.buffer;
    task->bcast_linear.size =
        ucc_dt_size(task->bcast_linear.dt) * coll_args->args.src.info.count;
    task->bcast_linear.num_steps = ucc_div_round_up(
        task->bcast_linear.size, get_raw_scratch_size(team) / 2);

    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_cuda_bcast_linear_start;
    task->super.progress = ucc_tl_cuda_bcast_linear_progress;
    task->super.finalize = ucc_tl_cuda_bcast_linear_finalize;

    *task_p = &task->super;
    return UCC_OK;
}
