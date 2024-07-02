/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast/bcast.h"

enum {
    STAGE_UNDEF,
    // root
    STAGE_COPY,      // post copy task: copy block from src to scratch buffer
    STAGE_WAIT_COPY, // wait for copy finishes
    STAGE_WAIT_ALL,  // wait for all others rank be on same step
    // non-root
};

// TODO: move out to common with allgather
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

ucc_status_t ucc_tl_cuda_bcast_linear_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

void ucc_tl_cuda_bcast_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize = UCC_TL_TEAM_SIZE(team);
    ucc_status_t        st;
    (void)team;
    (void)st;
    ucc_ee_executor_t      *exec;
    ucc_ee_executor_task_t *etask;
    ucc_status_t            status;
    void                   *sbuf, *dbuf;
    task->super.status = UCC_INPROGRESS;

    status = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        return;
    }

    if (trank == 0) {
        // fall-through between cases is intentional
        switch (task->bcast_linear.stage) {
        case STAGE_COPY:
            // copy from src buffer to scratch
            dbuf   = TASK_SCRATCH(task, trank);
            sbuf   = task->bcast_linear.sbuf;
            status = ecopy(dbuf, sbuf, task->bcast_linear.size, exec,
                           &task->bcast_linear.exec_task);
            task->bcast_linear.stage = STAGE_WAIT_COPY;
            break;
        case STAGE_WAIT_COPY:
            etask = task->bcast_linear.exec_task;
            if (etask) {
                status = ucc_ee_executor_task_test(etask);
                if (status == UCC_OK) {
                    ucc_ee_executor_task_finalize(etask);
                    task->bcast_linear.exec_task = NULL;
                    ucc_info("hello from rank: %d, copy done!", trank);
                    // signal others
                    ++task->bcast_linear.step;
                    set_rank_step(task, 0, task->bcast_linear.step, 0);
                    task->bcast_linear.stage = STAGE_WAIT_ALL;
                }
            }
            break;
        case STAGE_WAIT_ALL:
            for (int i = 1; i < tsize; ++i) {
                int other_rank_step = get_rank_step(task, i, 0);
                ucc_info("rank %d, step: %d, my step: %d", i, other_rank_step,
                         task->bcast_linear.step);
                if (other_rank_step < task->bcast_linear.step) {
                    ucc_info("rank %d is not ready", i);
                    return;
                }
            }
            task->bcast_linear.stage = STAGE_COPY;
            ucc_info("all others ready for next step");
            // TODO: remove
            task->bcast_linear.stage = STAGE_UNDEF;
            break;
        default:
            break;
        }
        // Root scenario
    } else {
        // others
    }
}

ucc_status_t ucc_tl_cuda_bcast_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t    *args  = &TASK_ARGS(task);
    ucc_rank_t          tsize = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt    = task->bcast_linear.dt;

    (void)tsize;
    (void)args;
    (void)dt;
    task->bcast_linear.stage = STAGE_COPY;
    ucc_info("bcast start with dt: %ld", dt);

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

    ucc_info("bcast init");

    if (ucc_unlikely(!ucc_tl_cuda_team_topo_is_fully_conntected(team->topo) ||
                     UCC_TL_TEAM_SIZE(team) - 1 >
                         UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    // task->allgatherv_linear.get_count  = ucc_tl_cuda_allgather_get_count;
    // task->allgatherv_linear.get_offset = ucc_tl_cuda_allgather_get_offset;
    task->bcast_linear.dt = coll_args->args.src.info.datatype;
    ucc_info("bcast start with dt: %ld", task->bcast_linear.dt);

    // task->allgatherv_linear.sbuf       = coll_args->args.src.info.buffer;
    // task->allgatherv_linear.rbuf       = coll_args->args.dst.info.buffer;

    task->bcast_linear.sbuf = coll_args->args.src.info.buffer;

    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post     = ucc_tl_cuda_bcast_linear_start;
    task->super.progress = ucc_tl_cuda_bcast_linear_progress;
    task->super.finalize = ucc_tl_cuda_bcast_linear_finalize;
    task->bar            = TASK_BAR(task);

    ucc_info("bcast init success");

    *task_p = &task->super;
    return UCC_OK;
}
