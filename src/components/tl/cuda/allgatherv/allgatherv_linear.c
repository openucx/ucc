/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

/*
 * fragmented buffered copy linear allgatherv algorithm
 *
 * Description:
 *  Definitions:
 *      blockI  - full send buffer at Rank I
 *      fragI_J - fragment of send buffer at Rank I and step J
 *      NS      - number of steps
 *      NF      - number of fragments
 *      N       - team size
 *
 *  Setup:
 *      max_frag_size = ucc_min(ucc_max(block1, block2, ..., block N),
 *                              scratch_size / 2 / N)
 *      NF            = ucc_max(block1, block2, ..., block N) / max_frag_size
 *      NS            = 1 + NF
 *
 *  Algorithm
 *      for rank R
 *      step 1:    copy fragR_1 to remote scratch buffers for all ranks
 *                 if not inplace copy local src buffer to local dst buffer
 *
 *      step 1:    copy frag1_1, frag2_1, ..., fragN_1 from local scratch buffer
 *                 to local dst buffer
 *                 copy fragR_2 from local dst buffer to remote scratch buffers
 *                 for all ranks
 *
 *      step NS-1: copy frag1_(NS-2), frag2_(NS-2), ..., fragN_(NS-2) from local
 *                 scratch buffer to local dst buffer
 *                 copy fragR_NS from local dst buffer to remote scratch buffers
 *                 for all ranks
 *
 *      step NS:   copy frag1_(NS-1), frag2_(NS-1), ..., fragN_(NS-1) from local
 *                 scratch buffer to local dst buffer
 */

enum
{
    STAGE_SYNC,    /*< Wait for free SYNC segment */
    STAGE_SETUP,   /*< Wait for memhandle setup to finish */
    STAGE_COPIES,  /*< Linear algorithm is running */
    STAGE_BARRIER, /*< Linear algorithm is done, waiting for
                    *  other ranks to finish */
};

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

ucc_status_t ucc_tl_cuda_allgatherv_linear_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_allgatherv_linear_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t        status;

    set_rank_step(task, trank, 0, 0);
    ucc_memory_cpu_store_fence();
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_allgatherv_linear_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

static inline size_t get_scratch_size(ucc_tl_cuda_team_t *team,
                                      ucc_datatype_t      dt)
{
    size_t     dt_size = ucc_dt_size(dt);
    ucc_rank_t tsize   = UCC_TL_TEAM_SIZE(team);
    size_t     div     = 2 * dt_size * tsize;

    return (UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size / div) * div;
}

static inline size_t get_scratch_offset(ucc_tl_cuda_team_t *team,
                                        ucc_datatype_t dt, int block)
{
    size_t     step_ssize = get_scratch_size(team, dt) / 2;
    ucc_rank_t tsize      = UCC_TL_TEAM_SIZE(team);

    return ucc_buffer_block_offset(step_ssize, tsize, block);
}

static inline ucc_status_t ecopy(void *dst, void *src, size_t size,
                                 ucc_ee_executor_t *      exec,
                                 ucc_ee_executor_task_t **etask)
{
    ucc_ee_executor_task_args_t exec_args;

    exec_args.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    exec_args.copy.dst  = dst;
    exec_args.copy.src  = src;
    exec_args.copy.len  = size;
    return ucc_ee_executor_task_post(exec, &exec_args, etask);
}

ucc_status_t
ucc_tl_cuda_allgatherv_linear_progress_frag(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *    team      = TASK_TEAM(task);
    ucc_rank_t              trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t              tsize     = UCC_TL_TEAM_SIZE(team);
    ucc_coll_args_t *       args      = &TASK_ARGS(task);
    ucc_datatype_t          dt        = task->allgatherv_linear.dt;
    size_t                  dt_size   = ucc_dt_size(dt);
    size_t                  ssize     = get_scratch_size(team, dt);
    int                     nfrags    = task->allgatherv_linear.num_frags;
    int                     num_steps = nfrags + 1;
    ucc_ee_executor_task_t *etask;
    ucc_ee_executor_t *     exec;
    ucc_status_t            st;
    int                     step, i;
    void *                  sbuf, *dbuf;
    size_t send_size, frag_size, frag_offset, local_offset, remote_offset,
        scratch_offset, rank_offset;

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        return st;
    }

    step = get_rank_step(task, trank, 0);
    while (step < num_steps) {
        if (task->allgatherv_linear.num_posted >
            task->allgatherv_linear.num_completed) {
            for (i = 0; i < tsize * 2; i++) {
                etask = task->allgatherv_linear.exec_task[i];
                if (etask != NULL) {
                    st = ucc_ee_executor_task_test(etask);
                    if (st == UCC_OK) {
                        ucc_ee_executor_task_finalize(etask);
                        task->allgatherv_linear.exec_task[i] = NULL;
                        task->allgatherv_linear.num_completed++;
                    } else {
                        if (ucc_likely(st > 0)) {
                            return UCC_INPROGRESS;
                        }
                        return st;
                    }
                }
            }
            step++;
            set_rank_step(task, trank, step, 0);
            continue;
        }

        for (i = 0; i < tsize; i++) {
            if (get_rank_step(task, i, 0) < step) {
                return UCC_INPROGRESS;
            }
        }

        if (step % 2) {
            remote_offset = ssize / 2;
            local_offset  = 0;
        } else {
            remote_offset = 0;
            local_offset  = ssize / 2;
        }

        if (step == 0) {
            send_size = task->allgatherv_linear.get_count(task, trank);
            frag_size =
                ucc_buffer_block_count(send_size, nfrags, step) * dt_size;
            frag_offset =
                ucc_buffer_block_offset(send_size, nfrags, step) * dt_size;
            if (UCC_IS_INPLACE(*args)) {
                rank_offset =
                    task->allgatherv_linear.get_offset(task, trank) * dt_size;
                sbuf = PTR_OFFSET(task->allgatherv_linear.rbuf,
                                  frag_offset + rank_offset);
            } else {
                sbuf = PTR_OFFSET(task->allgatherv_linear.sbuf, frag_offset);
            }
            for (i = 0; i < tsize; i++) {
                if (i == trank) {
                    continue;
                }
                scratch_offset = get_scratch_offset(team, dt, trank);
                dbuf           = PTR_OFFSET(TASK_SCRATCH(task, i),
                                  remote_offset + scratch_offset);

                st = ecopy(dbuf, sbuf, frag_size, exec,
                           &task->allgatherv_linear.exec_task[i]);
                if (ucc_unlikely(st != UCC_OK)) {
                    return st;
                }
            }
            if (!UCC_IS_INPLACE(*args)) {
                rank_offset =
                    task->allgatherv_linear.get_offset(task, trank) * dt_size;
                dbuf = PTR_OFFSET(task->allgatherv_linear.rbuf, rank_offset);

                st = ecopy(dbuf, sbuf,
                           task->allgatherv_linear.get_count(task, trank) *
                               dt_size,
                           exec, &task->allgatherv_linear.exec_task[tsize]);
                if (ucc_unlikely(st != UCC_OK)) {
                    return st;
                }
                task->allgatherv_linear.num_posted++;
            }
            task->allgatherv_linear.num_posted += tsize - 1;
        } else if (step == (num_steps - 1)) {
            for (i = 0; i < tsize; i++) {
                if (i == trank) {
                    continue;
                }
                scratch_offset = get_scratch_offset(team, dt, i);
                rank_offset =
                    task->allgatherv_linear.get_offset(task, i) * dt_size;
                send_size = task->allgatherv_linear.get_count(task, i);
                frag_offset =
                    ucc_buffer_block_offset(send_size, nfrags, step - 1) *
                    dt_size;
                frag_size =
                    ucc_buffer_block_count(send_size, nfrags, step - 1) *
                    dt_size;
                sbuf = PTR_OFFSET(TASK_SCRATCH(task, trank),
                                  local_offset + scratch_offset);
                dbuf = PTR_OFFSET(task->allgatherv_linear.rbuf,
                                  rank_offset + frag_offset);

                st = ecopy(dbuf, sbuf, frag_size, exec,
                           &task->allgatherv_linear.exec_task[i]);
                if (ucc_unlikely(st != UCC_OK)) {
                    return st;
                }
            }
            task->allgatherv_linear.num_posted += tsize - 1;
        } else {
            send_size = task->allgatherv_linear.get_count(task, trank);
            frag_size =
                ucc_buffer_block_count(send_size, nfrags, step) * dt_size;
            frag_offset =
                ucc_buffer_block_offset(send_size, nfrags, step) * dt_size;
            rank_offset =
                task->allgatherv_linear.get_offset(task, trank) * dt_size;
            sbuf           = PTR_OFFSET(task->allgatherv_linear.rbuf,
                              rank_offset + frag_offset);
            scratch_offset = get_scratch_offset(team, dt, trank);
            for (i = 0; i < tsize; i++) {
                if (i == trank) {
                    continue;
                }
                rank_offset =
                    task->allgatherv_linear.get_offset(task, i) * dt_size;
                dbuf = PTR_OFFSET(TASK_SCRATCH(task, i),
                                  remote_offset + scratch_offset);

                st = ecopy(dbuf, sbuf, frag_size, exec,
                           &task->allgatherv_linear.exec_task[i]);
                if (ucc_unlikely(st != UCC_OK)) {
                    return st;
                }
            }

            for (i = 0; i < tsize; i++) {
                if (i == trank) {
                    continue;
                }
                scratch_offset = get_scratch_offset(team, dt, i);
                rank_offset =
                    task->allgatherv_linear.get_offset(task, i) * dt_size;
                send_size = task->allgatherv_linear.get_count(task, i);
                frag_offset =
                    ucc_buffer_block_offset(send_size, nfrags, step - 1) *
                    dt_size;
                frag_size =
                    ucc_buffer_block_count(send_size, nfrags, step - 1) *
                    dt_size;
                sbuf = PTR_OFFSET(TASK_SCRATCH(task, trank),
                                  local_offset + scratch_offset);
                dbuf = PTR_OFFSET(task->allgatherv_linear.rbuf,
                                  rank_offset + frag_offset);

                st = ecopy(dbuf, sbuf, frag_size, exec,
                           &task->allgatherv_linear.exec_task[tsize + i]);
                if (ucc_unlikely(st != UCC_OK)) {
                    return st;
                }
            }
            task->allgatherv_linear.num_posted += 2 * (tsize - 1);
        }
    }

    return UCC_OK;
}

void ucc_tl_cuda_allgatherv_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_status_t        st;

    task->super.status = UCC_INPROGRESS;
    switch (task->allgatherv_ring.stage) {
    case STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        st = ucc_tl_cuda_allgatherv_linear_setup_start(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->allgatherv_linear.stage = STAGE_SETUP;
    case STAGE_SETUP:
        st = ucc_tl_cuda_allgatherv_linear_setup_test(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->allgatherv_linear.stage = STAGE_COPIES;
    case STAGE_COPIES:
        st = ucc_tl_cuda_allgatherv_linear_progress_frag(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }

        st = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.status = st;
            return;
        }
        task->allgatherv_linear.stage = STAGE_BARRIER;
    default:
        ucc_assert(task->allgatherv_linear.stage == STAGE_BARRIER);
        break;
    }
    task->super.status =
        ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
    if (task->super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
}

ucc_status_t ucc_tl_cuda_allgatherv_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task  = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t *   args  = &TASK_ARGS(task);
    ucc_rank_t          tsize = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t      dt    = task->allgatherv_ring.dt;
    ucc_rank_t          i;
    size_t              send_size, frag_size, ssize;

    task->allgatherv_linear.stage         = STAGE_SYNC;
    task->allgatherv_linear.sbuf          = args->src.info.buffer;
    task->allgatherv_linear.num_posted    = 0;
    task->allgatherv_linear.num_completed = 0;
    if (args->coll_type == UCC_COLL_TYPE_ALLGATHERV) {
        task->allgatherv_linear.rbuf = args->dst.info_v.buffer;
    } else {
        task->allgatherv_linear.rbuf = args->dst.info.buffer;
    }

    send_size = task->allgatherv_linear.get_count(task, 0);
    for (i = 1; i < tsize; i++) {
        send_size =
            ucc_max(send_size, task->allgatherv_linear.get_count(task, i));
    }

    if (send_size == 0) {
        task->super.status = UCC_OK;
        return ucc_task_complete(&task->super);
    }

    ssize     = get_scratch_size(team, dt);
    frag_size = ucc_min(ssize / 2 / ucc_dt_size(dt) / tsize, send_size);
    task->allgatherv_linear.num_frags = ucc_div_round_up(send_size, frag_size);

    memset(task->allgatherv_linear.exec_task, 0,
           2 * tsize * sizeof(ucc_ee_executor_task_t *));
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_cuda_allgatherv_linear_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t *     tl_team,
                                                ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task = ucc_tl_cuda_task_init(coll_args, team);

    task->allgatherv_linear.get_count  = ucc_tl_cuda_allgatherv_get_count;
    task->allgatherv_linear.get_offset = ucc_tl_cuda_allgatherv_get_offset;
    task->allgatherv_linear.dt         = coll_args->args.dst.info_v.datatype;
    task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post           = ucc_tl_cuda_allgatherv_linear_start;
    task->super.triggered_post = ucc_triggered_post;
    task->super.progress       = ucc_tl_cuda_allgatherv_linear_progress;
    task->super.finalize       = ucc_tl_cuda_allgatherv_linear_finalize;
    task->bar                  = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
