/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "tl_cuda_ring.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);

    set_rank_step(task, trank, 0, 0);
    ucc_memory_cpu_store_fence();
    return ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_ring_progress_ring(ucc_tl_cuda_task_t * task,
                                               uint32_t ring_id)
{
    ucc_tl_cuda_team_t *team     = TASK_TEAM(task);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_rank_t          trank    = UCC_TL_TEAM_RANK(team);
    int                 tsize    = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t          sendto   = get_send_to(team, trank, tsize, ring_id);
    ucc_rank_t          recvfrom = get_recv_from(team, trank, tsize, ring_id);
    ucc_datatype_t      dt       = task->reduce_scatterv_ring.dt;
    void               *sbuf     = task->reduce_scatterv_ring.sbuf;
    void               *dbuf     = task->reduce_scatterv_ring.rbuf;
    size_t              ssize    = UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t exec_args;
    int step, send_step, recv_step, frag, frag_step;
    size_t frag_count, block_count, block;
    size_t remote_offset, local_offset, frag_offset, block_offset;
    void *rsrc1, *rsrc2, *rdst;
    ucc_status_t st;

    if (UCC_IS_INPLACE(*args)) {
        sbuf = dbuf;
    }

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        task->super.super.status = st;
        return task->super.super.status;
    }

    step = get_rank_step(task, trank, ring_id);
    while (step < (tsize * task->reduce_scatterv_ring.num_frags)) {
        if (task->reduce_scatterv_ring.exec_task != NULL) {
            st = ucc_ee_executor_task_test(task->reduce_scatterv_ring.exec_task);
            if (st != UCC_OK) {
                if (st > 0) {
                    task->super.super.status = UCC_INPROGRESS;
                } else {
                    task->super.super.status = st;
                }
                return task->super.super.status;
            }
            ucc_ee_executor_task_finalize(task->reduce_scatterv_ring.exec_task);
            task->reduce_scatterv_ring.exec_task = NULL;
            step++;
            set_rank_step(task, trank, step, ring_id);
            continue;
        }
        send_step = get_rank_step(task, sendto, ring_id);
        recv_step = get_rank_step(task, recvfrom, ring_id);
        if ((send_step < step) || (recv_step < step)) {
            task->super.super.status = UCC_INPROGRESS;
            return task->super.super.status;
        }
        frag = step / tsize;
        frag_step = step % tsize;

        block        = get_recv_block(team, trank, tsize, frag_step, ring_id);
        block_count  = task->reduce_scatterv_ring.get_count(task, block);
        block_offset = task->reduce_scatterv_ring.get_offset(task, block);
        frag_offset  = ucc_buffer_block_offset(block_count,
                                               task->reduce_scatterv_ring.num_frags,
                                               frag);
        frag_count   = ucc_buffer_block_count(block_count,
                                              task->reduce_scatterv_ring.num_frags,
                                              frag);
        if (step % 2) {
            remote_offset = 0;
            local_offset = ssize / 2;
        } else {
            remote_offset = ssize / 2;
            local_offset = 0;
        }

        if (frag_step == 0) {
            exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
            exec_args.bufs[0]   = PTR_OFFSET(TASK_SCRATCH(task, trank), local_offset);
            exec_args.bufs[1]   = PTR_OFFSET(sbuf, (block_offset + frag_offset) *
                                             ucc_dt_size(dt));
            exec_args.count     = frag_count * ucc_dt_size(dt);
        } else {
            rsrc1 = PTR_OFFSET(sbuf, (block_offset + frag_offset) * ucc_dt_size(dt));
            rsrc2 = PTR_OFFSET(TASK_SCRATCH(task, recvfrom), remote_offset);
            if (frag_step == tsize - 1) {
                if (UCC_IS_INPLACE(*args)) {
                    rdst = PTR_OFFSET(dbuf, (block_offset + frag_offset) *
                                      ucc_dt_size(dt));
                } else {
                    rdst = PTR_OFFSET(dbuf, frag_offset * ucc_dt_size(dt));
                }
            } else {
                rdst = PTR_OFFSET(TASK_SCRATCH(task, trank), local_offset);
            }
            exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_REDUCE;
            exec_args.bufs[0]   = rdst;
            exec_args.bufs[1]   = rsrc1;
            exec_args.bufs[2]   = rsrc2;
            exec_args.count     = frag_count;
            exec_args.dt        = dt;
            exec_args.op        = args->op;
        }
        st = ucc_ee_executor_task_post(exec, &exec_args,
                                       &task->reduce_scatterv_ring.exec_task);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return task->super.super.status;
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_status_t st;

    switch (task->reduce_scatterv_ring.stage) {
    case RING_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.super.status = UCC_INPROGRESS;
            return task->super.super.status;
        }
        st = ucc_tl_cuda_reduce_scatterv_ring_setup_start(task);
        if (st != UCC_OK) {
            task->super.super.status = st;
            return task->super.super.status;
        }
        task->reduce_scatterv_ring.stage = RING_STAGE_SETUP;
    case RING_STAGE_SETUP:
        st = ucc_tl_cuda_reduce_scatterv_ring_setup_test(task);
        if (st != UCC_OK) {
            task->super.super.status = st;
            return task->super.super.status;
        }
        task->reduce_scatterv_ring.stage = RING_STAGE_RING;
    case RING_STAGE_RING:
        /* TODO: add support for multiple rings, only ring 0 is used so far */
        st = ucc_tl_cuda_reduce_scatterv_ring_progress_ring(task, 0);
        if (st != UCC_OK) {
            return task->super.super.status;
        }

        st = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return task->super.super.status;
        }

        task->reduce_scatterv_ring.stage = RING_STAGE_BARRIER;
    default:
        ucc_assert(task->reduce_scatterv_ring.stage == RING_STAGE_BARRIER);
        break;
    }

    task->super.super.status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team),
                                                            task->bar);
    if (task->super.super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
    return task->super.super.status;
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);

    task->reduce_scatterv_ring.exec_task = NULL;
    task->reduce_scatterv_ring.stage = RING_STAGE_SYNC;
    ucc_tl_cuda_reduce_scatterv_ring_progress(coll_task);
    if (task->super.super.status == UCC_INPROGRESS) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

size_t ucc_tl_cuda_reduce_scatterv_ring_get_count(const ucc_tl_cuda_task_t *task,
                                                  ucc_rank_t block)
{
    const ucc_coll_args_t *args  = &TASK_ARGS(task);

    return ucc_coll_args_get_count(args, args->dst.info_v.counts, block);
}

size_t ucc_tl_cuda_reduce_scatterv_ring_get_offset(const ucc_tl_cuda_task_t *task,
                                                   ucc_rank_t block)
{
    size_t     offset = 0;
    ucc_rank_t i;

    for (i = 0; i < block; i++) {
        offset += task->reduce_scatterv_ring.get_count(task, i);
    }

    return offset;
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_init(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_rank_t          tsize  = UCC_TL_TEAM_SIZE(team);
    size_t              ssize  = UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
    ucc_datatype_t      dt     = args->dst.info_v.datatype;
    size_t send_size, frag_size;
    ucc_rank_t i;

    task->reduce_scatterv_ring.get_count  = ucc_tl_cuda_reduce_scatterv_ring_get_count;
    task->reduce_scatterv_ring.get_offset = ucc_tl_cuda_reduce_scatterv_ring_get_offset;
    task->reduce_scatterv_ring.dt         = args->dst.info_v.datatype;
    task->reduce_scatterv_ring.sbuf       = args->src.info.buffer;
    task->reduce_scatterv_ring.rbuf       = args->dst.info_v.buffer;

    send_size = task->reduce_scatterv_ring.get_count(task, 0);
    for (i = 1; i < tsize; i++) {
        send_size = ucc_max(send_size,
                            task->reduce_scatterv_ring.get_count(task, i));
    }
    frag_size = ucc_min(ssize / ucc_dt_size(dt) / 2, send_size);

    task->super.flags                    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->reduce_scatterv_ring.num_frags = ucc_div_round_up(send_size, frag_size);
    task->super.post                     = ucc_tl_cuda_reduce_scatterv_ring_start;
    task->super.triggered_post           = ucc_triggered_post;
    task->super.progress                 = ucc_tl_cuda_reduce_scatterv_ring_progress;
    task->super.finalize                 = ucc_tl_cuda_reduce_scatterv_ring_finalize;
    task->bar                            = TASK_BAR(task);

    return UCC_OK;
}
