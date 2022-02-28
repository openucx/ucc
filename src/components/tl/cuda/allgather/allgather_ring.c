/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "allgather.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

enum {
    ALLGATHER_RING_STAGE_SYNC,    /*< Wait for free SYNC segment */
    ALLGATHER_RING_STAGE_SETUP,   /*< Wait for memhandle setup to finish */
    ALLGATHER_RING_STAGE_RING,    /*< Ring algorithm is running */
    ALLGATHER_RING_STAGE_BAR,     /*< Ring algorithm is done, waiting for
                                   *  other ranks to finish
                                   */
};

static inline ucc_rank_t get_send_to(ucc_tl_cuda_team_t *team,
                                     ucc_rank_t trank, ucc_rank_t tsize,
                                     int ring_id)
{
    ucc_tl_cuda_ring_t *ring = &team->topo->rings[ring_id];

    return ring->ring[(ring->iring[trank] + 1) % tsize];
}

static inline ucc_rank_t get_recv_from(ucc_tl_cuda_team_t *team,
                                       ucc_rank_t trank, ucc_rank_t tsize,
                                       int ring_id)
{
    ucc_tl_cuda_ring_t *ring = &team->topo->rings[ring_id];

    return ring->ring[(ring->iring[trank] - 1 + tsize) % tsize];
}

static inline ucc_rank_t get_send_block(ucc_tl_cuda_team_t *team,
                                        ucc_rank_t trank, ucc_rank_t tsize,
                                        uint32_t step, int ring_id)
{
    ucc_tl_cuda_ring_t *ring = &team->topo->rings[ring_id];

    return ring->ring[(ring->iring[trank] + tsize - step) % tsize];
}

static inline ucc_rank_t get_recv_block(ucc_tl_cuda_team_t *team,
                                        ucc_rank_t trank, ucc_rank_t tsize,
                                        uint32_t step, int ring_id)
{
    ucc_tl_cuda_ring_t *ring = &team->topo->rings[ring_id];

    return ring->ring[(ring->iring[trank] + tsize - step - 1) % tsize];
}

static inline size_t ucc_ring_block_offset(size_t total_count,
                                           ucc_rank_t n_blocks,
                                           ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    size_t offset      = block * block_count + left;
    return (block < left) ? offset - (left - block) : offset;
}

static inline size_t ucc_ring_block_count(size_t total_count,
                                          ucc_rank_t n_blocks, ucc_rank_t block)
{
    size_t block_count = total_count / n_blocks;
    size_t left        = total_count % n_blocks;
    return (block < left) ? block_count + 1 : block_count;
}

static inline size_t max_frag_size(size_t total_count, ucc_rank_t n_blocks,
                                   ucc_rank_t n_frags)
{
    size_t block_count = ucc_ring_block_count(total_count, n_blocks, 0);
    return ucc_ring_block_count(block_count, n_frags, 0);
}

static inline int get_rank_step(ucc_tl_cuda_task_t *task, ucc_rank_t rank,
                                int ring_id)
{
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);
    return sync->seq_num;
}

static inline void set_rank_step(ucc_tl_cuda_task_t *task, ucc_rank_t rank,
                                 int step, int ring_id)
{
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);
    sync->seq_num = step;
}

ucc_status_t ucc_tl_cuda_allgather_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_allgather_ring_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t        status;

    if (UCC_IS_INPLACE(TASK_ARGS(task))) {
        set_rank_step(task, trank, 0, 0);
    } else {
        set_rank_step(task, trank, -1, 0);
    }
    ucc_memory_bus_fence();
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_allgather_ring_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

ucc_status_t ucc_tl_cuda_allgather_ring_progress_ring(ucc_tl_cuda_task_t * task,
                                                      uint32_t ring_id)
{
    ucc_tl_cuda_team_t *team     = TASK_TEAM(task);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_rank_t          trank    = UCC_TL_TEAM_RANK(team);
    int                 tsize    = (int)UCC_TL_TEAM_SIZE(team);
    size_t              ccount   = args->dst.info.count / tsize;
    ucc_datatype_t      dt       = args->dst.info.datatype;
    size_t              ds       = ccount * ucc_dt_size(dt);
    ucc_rank_t          sendto   = get_send_to(team, trank, tsize, ring_id);
    ucc_rank_t          recvfrom = get_recv_from(team, trank, tsize, ring_id);
    void               *rbuf     = args->dst.info.buffer;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t exec_args;
    void *dbuf1, *dbuf2, *sbuf;
    int step, send_step, recv_step, frag, frag_step, i;
    ucc_rank_t peer_block;
    size_t remote_offset, local_offset, frag_offset, frag_size;
    ucc_status_t st;

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        task->super.super.status = st;
        return task->super.super.status;
    }
    step = get_rank_step(task, trank, ring_id);
    while (step < (tsize * task->allgather_ring.num_frags)) {
        if ((task->allgather_ring.exec_task[0] != NULL) ||
            (task->allgather_ring.exec_task[1] != NULL)) {
            for (i = 0 ; i < 2; i++) {
                if (task->allgather_ring.exec_task[i] != NULL) {
                    st = ucc_ee_executor_task_test(task->allgather_ring.exec_task[i]);
                    if (st == UCC_OK) {
                        ucc_ee_executor_task_finalize(task->allgather_ring.exec_task[i]);
                        task->allgather_ring.exec_task[i] = NULL;
                    } else {
                        if (st > 0) {
                            task->super.super.status = UCC_INPROGRESS;
                        } else {
                            task->super.super.status = st;
                        }
                        return task->super.super.status;
                    }
                }
            }
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
        frag_offset = ucc_ring_block_offset(ds, task->allgather_ring.num_frags,
                                            frag);
        frag_size = ucc_ring_block_count(ds, task->allgather_ring.num_frags,
                                         frag);
        if (step % 2) {
            remote_offset = frag_size;
            local_offset = 0;
        } else {
            remote_offset = 0;
            local_offset = frag_size;
        }
        peer_block = get_send_block(team, trank, tsize, frag_step, ring_id);
        if (frag_step == 0) {
            sbuf  = PTR_OFFSET(rbuf, peer_block * ds + frag_offset);
            dbuf1 = PTR_OFFSET(TASK_SCRATCH(task, sendto), remote_offset);
            dbuf2 = NULL;
        } else if (frag_step == (tsize -1)) {
            sbuf  = PTR_OFFSET(TASK_SCRATCH(task, trank), local_offset);
            dbuf1 = PTR_OFFSET(rbuf, peer_block * ds + frag_offset);
            dbuf2 = NULL;
        } else {
            sbuf  = PTR_OFFSET(TASK_SCRATCH(task, trank), local_offset);
            dbuf1 = PTR_OFFSET(TASK_SCRATCH(task, sendto), remote_offset);
            dbuf2 = PTR_OFFSET(rbuf, peer_block * ds + frag_offset);
        }

        exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
        exec_args.bufs[0]   = dbuf1;
        exec_args.bufs[1]   = sbuf;
        exec_args.count     = frag_size;
        st = ucc_ee_executor_task_post(exec, &exec_args,
                                       &task->allgather_ring.exec_task[0]);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return task->super.super.status;
        }
        if (dbuf2 != NULL) {
            exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
            exec_args.bufs[0]   = dbuf2;
            exec_args.bufs[1]   = sbuf;
            exec_args.count     = frag_size;
            st = ucc_ee_executor_task_post(exec, &exec_args,
                                           &task->allgather_ring.exec_task[1]);
            if (ucc_unlikely(st != UCC_OK)) {
                task->super.super.status = st;
                return task->super.super.status;
            }
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_allgather_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_status_t st;

    switch (task->allgather_ring.stage) {
    case ALLGATHER_RING_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.super.status = UCC_INPROGRESS;
            return task->super.super.status;
        }
        task->allgather_ring.stage = ALLGATHER_RING_STAGE_RING;
        st = ucc_tl_cuda_allgather_ring_setup_start(task);
        if (st != UCC_OK) {
            task->super.super.status = st;
            return task->super.super.status;
        }
        task->allgather_ring.stage = ALLGATHER_RING_STAGE_SETUP;
    case ALLGATHER_RING_STAGE_SETUP:
        st = ucc_tl_cuda_allgather_ring_setup_test(task);
        if (st != UCC_OK) {
            task->super.super.status = st;
            return task->super.super.status;
        }
        task->allgather_ring.stage = ALLGATHER_RING_STAGE_RING;
    case ALLGATHER_RING_STAGE_RING:
        /* TODO: add support for multiple rings, only ring 0 is used so far */
        st = ucc_tl_cuda_allgather_ring_progress_ring(task, 0);
        if (st != UCC_OK) {
            return task->super.super.status;
        }

        st = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return task->super.super.status;
        }

        task->allgather_ring.stage = ALLGATHER_RING_STAGE_BAR;
    default:
        ucc_assert(task->allgather_ring.stage == ALLGATHER_RING_STAGE_BAR);
        break;
    }

    task->super.super.status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team),
                                                            task->bar);
    if (task->super.super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
    return task->super.super.status;
}

ucc_status_t ucc_tl_cuda_allgather_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_rank_t         tsize    = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         trank    = UCC_TL_TEAM_RANK(team);
    size_t             ccount   = args->dst.info.count / tsize;
    ucc_datatype_t     dt       = args->dst.info.datatype;
    size_t             ds       = ccount * ucc_dt_size(dt);
    void               *sbuf    = args->src.info.buffer;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t exec_args;
    ucc_status_t st;

    task->allgather_ring.exec_task[0] = NULL;
    task->allgather_ring.exec_task[1] = NULL;
    if (!UCC_IS_INPLACE(*args)) {
        st = ucc_coll_task_get_executor(&task->super, &exec);
        if (ucc_unlikely(st != UCC_OK)) {
           task->super.super.status = st;
           return ucc_task_complete(coll_task);
        }

        exec_args.task_type = UCC_EE_EXECUTOR_TASK_TYPE_COPY;
        exec_args.bufs[0]   = PTR_OFFSET(args->dst.info.buffer, ds * trank);
        exec_args.bufs[1]   = sbuf;
        exec_args.count     = ds;
        st = ucc_ee_executor_task_post(exec, &exec_args,
                                       &task->allgather_ring.exec_task[0]);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return ucc_task_complete(coll_task);
        }
    }

    task->allgather_ring.stage = ALLGATHER_RING_STAGE_SYNC;
    st = ucc_tl_cuda_allgather_ring_progress(coll_task);
    if (task->super.super.status == UCC_INPROGRESS) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_cuda_allgather_ring_init(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t    *args  = &TASK_ARGS(task);
    ucc_rank_t          tsize = UCC_TL_TEAM_SIZE(team);
    size_t              ssize = UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size;
    size_t              send_size, frag_size;

    send_size = ucc_dt_size(args->dst.info.datatype) *
                (args->dst.info.count / tsize);
    frag_size = ucc_min(ssize/2, send_size);

    task->super.flags              |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->allgather_ring.num_frags = (send_size + frag_size - 1) / frag_size;
    task->super.post               = ucc_tl_cuda_allgather_ring_start;
    task->super.triggered_post     = ucc_triggered_post;
    task->super.progress           = ucc_tl_cuda_allgather_ring_progress;
    task->super.finalize           = ucc_tl_cuda_allgather_ring_finalize;
    task->bar                      = TASK_BAR(task);

    return UCC_OK;
}
