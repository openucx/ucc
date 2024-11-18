/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast.h"

enum {
    // Barrier setup
    STAGE_INIT_BAR_ROOT,
    STAGE_FIND_BAR_PEER,

    STAGE_SYNC,
    STAGE_SETUP,
    // root
    STAGE_COPY,      // post copy task: copy block from src to scratch buffer
    STAGE_WAIT_COPY, // wait for copy finishes
    STAGE_WAIT_ALL,  // wait for all others rank be on same step
    // non-root
    STAGE_WAIT_ROOT,   // clients wait while root writes to own scratch buffer
    STAGE_CLIENT_COPY, // clients submit copy task
    STAGE_CLIENT_COPY_WAIT, // clients wait completion of copy from root's scratch
};

ucc_status_t ucc_tl_cuda_bcast_linear_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t        status;

    set_rank_step(task, trank, 0, 0);
    ucc_memory_cpu_store_fence();
    // initiate barrier wait while all ranks set theirs steps to 0
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_bcast_linear_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

static inline size_t get_raw_scratch_size(ucc_tl_cuda_team_t *team)
{
    return UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
}

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

static inline ucc_status_t root_find_free_barrier(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    uint32_t max_concurrent  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.max_concurrent;
    bool     found           = false;
    ucc_tl_cuda_shm_barrier_t *curr_bar;
    int                        i;
    ucc_status_t               st;

    /* search first free barrier in active set pool */
    for (i = 0; i < max_concurrent; ++i) {
        curr_bar = UCC_TL_CUDA_TEAM_BARRIER(team, max_concurrent + i);
        // try to set user specified tag to mark that this barrier is used by this task
        if (ucc_atomic_cswap64(&curr_bar->tag, UCC_TAG_FREE,
                               task->bcast_linear.key) == UCC_TAG_FREE) {
            ucc_print("found free barrier: %d marked with tag: %ld", i,
                      curr_bar->tag);
            // free
            task->bar = curr_bar;
            st        = ucc_tl_cuda_shm_barrier_init_root(
                task->subset.map.ep_num, task->subset.myrank,
                task->bcast_linear.root, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to init root barrier");
                return UCC_ERR_NO_RESOURCE;
            }
            found         = true;
            task->coll_id = i + max_concurrent;
            break;
        }
    }
    if (!found) {
        // try next time
        return UCC_ERR_NOT_FOUND;
    }
    return UCC_OK;
}

static inline ucc_status_t peer_find_free_barrier(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    uint32_t max_concurrent  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.max_concurrent;
    bool     found           = false;
    ucc_tl_cuda_shm_barrier_t *curr_bar;
    int                        i;
    ucc_status_t               st;
    for (i = 0; i < max_concurrent; ++i) {
        curr_bar = UCC_TL_CUDA_TEAM_BARRIER(team, max_concurrent + i);
        if (curr_bar->tag == task->bcast_linear.key) {
            task->bar = curr_bar;
            st = ucc_tl_cuda_shm_barrier_init_root(
                task->subset.map.ep_num, task->subset.myrank,
                task->bcast_linear.root, task->bar);
            if (ucc_unlikely(st != UCC_OK)) {
                ucc_error("failed to init peer barrier");
                return UCC_ERR_NO_RESOURCE;
            }
            found                    = true;
            task->coll_id            = i + max_concurrent;
            task->bcast_linear.stage = STAGE_SYNC;
            break;
        }
    }
    if (!found) {
        // try next time
        return UCC_ERR_NOT_FOUND;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_bcast_linear_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

void ucc_tl_cuda_bcast_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task              = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team              = TASK_TEAM(task);
    ucc_rank_t          trank             = UCC_TL_TEAM_RANK(team);
    size_t              half_scratch_size = get_raw_scratch_size(team) / 2;
    ucc_rank_t          tsize             = (ucc_rank_t)task->subset.map.ep_num;
    size_t              chunk_size        =
        task->bcast_linear.step < task->bcast_linear.num_steps
                         ? ucc_min(half_scratch_size, task->bcast_linear.size)
                         : task->bcast_linear.size -
                  (task->bcast_linear.step - 1) * half_scratch_size;
    size_t              offset_buff       = task->bcast_linear.step * half_scratch_size;
    ucc_ee_executor_t         *exec;
    ucc_ee_executor_task_t    *etask;
    ucc_status_t               st;
    void                      *sbuf, *dbuf;
    int                        i;
    ucc_rank_t                 peer;

    task->super.status = UCC_INPROGRESS;

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        task->super.status = st;
        return;
    }

    switch (task->bcast_linear.stage) {
    case STAGE_INIT_BAR_ROOT:
        st = root_find_free_barrier(task);
        if (st == UCC_ERR_NOT_FOUND) {
            // no free barriers found, try next time
            return;
        }
        if (st == UCC_OK) {
            task->bcast_linear.stage = STAGE_SYNC;
            break; // prevent from fall to next case
        } else {
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
    case STAGE_FIND_BAR_PEER:
        st = peer_find_free_barrier(task);
        if (st == UCC_ERR_NOT_FOUND) {
            // no free barriers found, wait for root
            return;
        }
        if (st == UCC_OK) {
            // barrier found, continue to next stages
            task->bcast_linear.stage = STAGE_SYNC;
            break;
        } else {
            task->super.status = UCC_ERR_NO_RESOURCE;
            return;
        }
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
        case STAGE_WAIT_COPY:
            etask = task->bcast_linear.exec_task;
            if (etask) {
                st = ucc_ee_executor_task_test(etask);
                if (st == UCC_OK) {
                    ucc_ee_executor_task_finalize(etask);
                    task->bcast_linear.exec_task = NULL;
                    // signal others
                    ++task->bcast_linear.step;
                    set_rank_step(task, task->bcast_linear.root,
                                  task->bcast_linear.step, 0);
                    task->bcast_linear.stage = STAGE_WAIT_ALL;
                } else {
                    // not ready
                    return;
                }
            } else {
                ucc_debug("etask is nullptr");
                return;
            }
        case STAGE_WAIT_ALL:
            for (i = 0; i < tsize; ++i) {
                if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task)))
                {
                    // eval phys rank from virt
                    peer = ucc_ep_map_eval(task->subset.map, i);
                }
                else
                {
                    peer = i;
                }
                // need to wait until all ranks complete step - 1, because of double buffering
                if (get_rank_step(task, peer, 0) < task->bcast_linear.step - 1) {
                    // rank is not ready, lets wait
                    return;
                }
            }
            task->bcast_linear.stage = STAGE_COPY;
            if (task->bcast_linear.step < task->bcast_linear.num_steps) {
                // go to next iteration
                task->bcast_linear.stage = STAGE_COPY;
                return;
            } else {
                // finish
                ucc_tl_cuda_put_sync_root(task, task->bcast_linear.root);
                task->super.status = UCC_OK;
                break;
            }
        default:
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
        case STAGE_CLIENT_COPY:
            // need to copy from root's scratch buffer
            dbuf = PTR_OFFSET(task->bcast_linear.sbuf, offset_buff);
            sbuf = PTR_OFFSET(TASK_SCRATCH(task, task->bcast_linear.root),
                              task->bcast_linear.step % 2 * chunk_size);
            st   = ecopy(dbuf, sbuf, chunk_size, exec,
                         &task->bcast_linear.exec_task);
            if (st != UCC_OK) {
                ucc_error("failed to post ecopy task at client");
                task->super.status = st;
                return;
            }
            task->bcast_linear.stage = STAGE_CLIENT_COPY_WAIT;
        case STAGE_CLIENT_COPY_WAIT:
            etask = task->bcast_linear.exec_task;
            if (etask) {
                st = ucc_ee_executor_task_test(etask);
                if (st == UCC_OK) {
                    ucc_ee_executor_task_finalize(etask);
                    task->bcast_linear.exec_task = NULL;
                    ++task->bcast_linear.step;
                    set_rank_step(task, trank, task->bcast_linear.step, 0);
                    if (task->bcast_linear.step <
                        task->bcast_linear.num_steps) {
                        task->bcast_linear.stage = STAGE_WAIT_ROOT;
                        return;
                    } else {
                        // Done
                        task->super.status = UCC_OK;
                        break;
                    }
                }
            }
        default:
            break;
        }
    }
}

ucc_status_t ucc_tl_cuda_bcast_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucc_datatype_t      dt   = task->bcast_linear.dt;
    size_t              half_scratch_size = get_raw_scratch_size(team) / 2;

    task->bcast_linear.stage = STAGE_SYNC;

    // in case of active set bcast we need to do additional steps to find free barriers
    if (UCC_COLL_ARGS_ACTIVE_SET(&TASK_ARGS(task))) {
        task->bcast_linear.stage = UCC_TL_TEAM_RANK(team) == task->bcast_linear.root ? STAGE_INIT_BAR_ROOT : STAGE_FIND_BAR_PEER;
    }
    
    task->bcast_linear.size = ucc_dt_size(dt) * args->src.info.count;
    task->bcast_linear.num_steps =
        ucc_div_round_up(task->bcast_linear.size, half_scratch_size);

    ucc_debug("bcast linear dt: %s, buffer size: %ld, num_steps: %d",
              ucc_datatype_str(dt), task->bcast_linear.size,
              task->bcast_linear.num_steps);

    task->bcast_linear.sbuf = args->src.info.buffer;
    task->bcast_linear.step = 0;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_cuda_bcast_linear_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t      *tl_team,
                                           ucc_coll_task_t     **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t        status;

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_connected(team->topo) ||
                     UCC_TL_TEAM_SIZE(team) - 1 >
                         UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->bcast_linear.root = coll_args->args.root;
    task->bcast_linear.dt   = coll_args->args.src.info.datatype;
    task->bcast_linear.sbuf = coll_args->args.src.info.buffer;

    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_cuda_bcast_linear_start;
    task->super.progress = ucc_tl_cuda_bcast_linear_progress;
    task->super.finalize = ucc_tl_cuda_bcast_linear_finalize;

    *task_p = &task->super;
    return UCC_OK;
}
