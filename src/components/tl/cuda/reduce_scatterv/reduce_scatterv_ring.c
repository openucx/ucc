/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "tl_cuda_ring.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

#define BLOCK_ALIGN 64

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
    int                 chunk;

    for (chunk = 0; chunk < 1; chunk++) {
        set_rank_step(task, trank, 0, chunk);
    }
    ucc_memory_cpu_store_fence();
    return ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

static inline int
ucc_tl_cuda_check_peers_ready(ucc_tl_cuda_task_t *task, int chunk, int step)
{
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          tsize  = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t          trank  = UCC_TL_TEAM_RANK(team);
    int                 nrings = task->reduce_scatterv_ring.num_rings;
    int ring, peer_step;
    ucc_rank_t peer;

    for (ring = 0; ring < nrings; ring++) {
        peer      = get_send_to(team, trank, tsize, ring);
        peer_step = get_rank_step(task, peer, chunk);

        if (peer_step < step) {
            return 0;
        }

        peer      = get_recv_from(team, trank, tsize, ring);
        peer_step = get_rank_step(task, peer, chunk);

        if (peer_step < step) {
            return 0;
        }
    }

    return 1;
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_ring_progress_ring(ucc_tl_cuda_task_t * task,
                                               int chunk)
{
    ucc_tl_cuda_team_t *team     = TASK_TEAM(task);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_rank_t          trank    = UCC_TL_TEAM_RANK(team);
    int                 tsize    = UCC_TL_TEAM_SIZE(team);
    int                 nsteps   = tsize;
    int                 nrings   = task->reduce_scatterv_ring.num_rings;
    int                 nfrags   = task->reduce_scatterv_ring.num_frags;
    ucc_datatype_t      dt       = task->reduce_scatterv_ring.dt;
    size_t              dt_size  = ucc_dt_size(dt);
    void               *sbuf     = task->reduce_scatterv_ring.sbuf;
    void               *dbuf     = task->reduce_scatterv_ring.rbuf;
    size_t              ssize    = get_scratch_size(team, nrings, 1, dt_size);
    ucc_rank_t recvfrom;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t eargs;
    ucc_ee_executor_task_t *etask;
    void *rsrc1, *rsrc2, *rdst;
    ucc_status_t st;
    int step, ring, frag, frag_step, k;
    size_t ring_frag_count, frag_count, block_count, block, ring_frag_offset,
           remote_offset, local_offset, frag_offset, block_offset,
           ring_scratch_offset;

    if (UCC_IS_INPLACE(*args)) {
        sbuf = dbuf;
    }

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        return st;
    }

    step = get_rank_step(task, trank, chunk);
    while (step < (nsteps * nfrags)) {
        etask = task->reduce_scatterv_ring.exec_task[chunk];
        if (etask != NULL) {
            st = ucc_ee_executor_task_test(etask);
            if (st != UCC_OK) {
                if (ucc_likely(st > 0)) {
                    st = UCC_INPROGRESS;
                }
                return st;
            }
            ucc_ee_executor_task_finalize(etask);
            task->reduce_scatterv_ring.exec_task[chunk] = NULL;
            step++;
            set_rank_step(task, trank, step, chunk);
            continue;
        }

        if (!ucc_tl_cuda_check_peers_ready(task, chunk, step)) {
            return UCC_INPROGRESS;
        }

        frag      = step / tsize;
        frag_step = step % tsize;

        if (step % 2) {
            remote_offset = 0;
            local_offset = ssize / 2;
        } else {
            remote_offset = ssize / 2;
            local_offset = 0;
        }

        k = 0;
        ring_scratch_offset = ssize / 2 / nrings;
        for (ring = 0; ring < nrings; ring++) {
            recvfrom = get_recv_from(team, trank, tsize, ring);
            block    = get_recv_block(team, trank, tsize, frag_step, ring);

            block_count = task->reduce_scatterv_ring.get_count(task, block);
            frag_count = ucc_buffer_block_count_aligned(
                block_count, nfrags, frag, BLOCK_ALIGN);
            ring_frag_count = ucc_buffer_block_count_aligned(
                frag_count, nrings, ring, BLOCK_ALIGN);

            if (ring_frag_count == 0) {
                continue;
            }

            block_offset = task->reduce_scatterv_ring.get_offset(task, block);
            frag_offset = ucc_buffer_block_offset_aligned(
                block_count, nfrags, frag, BLOCK_ALIGN);
            ring_frag_offset = ucc_buffer_block_offset_aligned(
                frag_count, nrings, ring, BLOCK_ALIGN);

            if (frag_step == 0) {
                /* copy data from local source buffer to local scratch buffer */
                eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY_MULTI;
                eargs.copy_multi.src[k]    = PTR_OFFSET(sbuf,
                        (block_offset + frag_offset + ring_frag_offset) * dt_size);
                eargs.copy_multi.dst[k]    = PTR_OFFSET(TASK_SCRATCH(task, trank),
                        local_offset + ring_scratch_offset * ring);
                eargs.copy_multi.counts[k] = ring_frag_count * dt_size;
                eargs.copy_multi.num_vectors = k + 1;
            } else {
                /* start reduction for data in remote scratch and local source buffer */
                rsrc1 = PTR_OFFSET(sbuf, (block_offset + frag_offset +
                                   ring_frag_offset) * dt_size);
                rsrc2 = PTR_OFFSET(TASK_SCRATCH(task, recvfrom), remote_offset +
                                   ring_scratch_offset * ring);
                if (frag_step == tsize - 1) {
                    /* save reduction result to local destination buffer on last step */
                    if (UCC_IS_INPLACE(*args)) {
                        rdst = PTR_OFFSET(dbuf, (block_offset + frag_offset +
                                          ring_frag_offset) * dt_size);
                    } else {
                        rdst = PTR_OFFSET(dbuf, (frag_offset + ring_frag_offset) *
                                          dt_size);
                    }
                } else {
                    /* save reduction result to local scratch buffer */
                    rdst = PTR_OFFSET(TASK_SCRATCH(task, trank), local_offset +
                                      ring_scratch_offset * ring);
                }
                eargs.task_type = UCC_EE_EXECUTOR_TASK_REDUCE_MULTI_DST;
                eargs.reduce_multi_dst.dst[k]    = rdst;
                eargs.reduce_multi_dst.src1[k]   = rsrc1;
                eargs.reduce_multi_dst.src2[k]   = rsrc2;
                eargs.reduce_multi_dst.counts[k] = ring_frag_count;
                eargs.reduce_multi_dst.dt        = dt;
                eargs.reduce_multi_dst.op        = args->op;
                eargs.reduce_multi_dst.n_bufs    = k + 1;
            }
            k++;
        }

        if (k == 0) {
            /* have no work to do at current step, go to next step */
            step++;
            set_rank_step(task, trank, step, chunk);
            continue;
        }
        st = ucc_ee_executor_task_post(exec, &eargs,
                &task->reduce_scatterv_ring.exec_task[chunk]);
        if (ucc_unlikely(st != UCC_OK)) {
            return st;
        }
    }

    return UCC_OK;
}

void ucc_tl_cuda_reduce_scatterv_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_status_t st;
    int chunk, num_done;

    task->super.status = UCC_INPROGRESS;
    switch (task->reduce_scatterv_ring.stage) {
    case RING_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        st = ucc_tl_cuda_reduce_scatterv_ring_setup_start(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->reduce_scatterv_ring.stage = RING_STAGE_SETUP;
        /* fall through */
    case RING_STAGE_SETUP:
        st = ucc_tl_cuda_reduce_scatterv_ring_setup_test(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->reduce_scatterv_ring.stage = RING_STAGE_RING;
        /* fall through */
    case RING_STAGE_RING:
        num_done = 0;
        for (chunk = 0; chunk < 1; chunk++) {
            st = ucc_tl_cuda_reduce_scatterv_ring_progress_ring(task, chunk);
            if (ucc_unlikely(st < 0)) {
                task->super.status = st;
                return;
            } else if (st == UCC_OK) {
                num_done++;
            }
        }
        if (num_done != 1) {
            return;
        }


        st = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.status = st;
            return;
        }

        task->reduce_scatterv_ring.stage = RING_STAGE_BARRIER;
        break;
    default:
        ucc_assert(task->reduce_scatterv_ring.stage == RING_STAGE_BARRIER);
        break;
    }

    task->super.status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team),
                                                      task->bar);
    if (task->super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
        UCC_TL_CUDA_PROFILE_REQUEST_EVENT(coll_task, "cuda_rsv_ring_done", 0);
    }
}

ucc_status_t ucc_tl_cuda_reduce_scatterv_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_tl_cuda_lib_t  *lib     = UCC_TL_CUDA_TEAM_LIB(team);
    ucc_rank_t          tsize   = UCC_TL_TEAM_SIZE(team);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_datatype_t      dt      = task->reduce_scatterv_ring.dt;
    size_t              dt_size = ucc_dt_size(dt);
    size_t              send_size, frag_size, ssize;
    int                 nrings;
    ucc_rank_t          i;

    UCC_TL_CUDA_PROFILE_REQUEST_EVENT(coll_task, "cuda_rsv_ring_start", 0);
    send_size = task->reduce_scatterv_ring.get_count(task, 0);
    for (i = 1; i < tsize; i++) {
        send_size = ucc_max(send_size,
                            task->reduce_scatterv_ring.get_count(task, i));
    }

    if (send_size == 0) {
        task->super.status = UCC_OK;
        return ucc_task_complete(&task->super);
    }

    nrings = get_num_rings(team, send_size * dt_size,
                           lib->cfg.reduce_scatter_ring_max_rings);

    task->reduce_scatterv_ring.sbuf      = args->src.info.buffer;
    task->reduce_scatterv_ring.num_rings = nrings;
    if (args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV) {
        task->reduce_scatterv_ring.rbuf = args->dst.info_v.buffer;
    } else {
        task->reduce_scatterv_ring.rbuf = args->dst.info.buffer;
    }

    ssize = get_scratch_size(team, nrings, 1, dt_size);
    frag_size = ucc_min(ssize / dt_size / 2, send_size);
    task->reduce_scatterv_ring.num_frags    = ucc_div_round_up(send_size, frag_size);
    task->reduce_scatterv_ring.stage        = RING_STAGE_SYNC;
    task->reduce_scatterv_ring.exec_task[0] = NULL;
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t
ucc_tl_cuda_reduce_scatterv_ring_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t *     tl_team,
                                      ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task;
    ucc_status_t status;

    if (coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_cuda_task_init(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    task->super.flags                    |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post                      = ucc_tl_cuda_reduce_scatterv_ring_start;
    task->super.progress                  = ucc_tl_cuda_reduce_scatterv_ring_progress;
    task->super.finalize                  = ucc_tl_cuda_reduce_scatterv_ring_finalize;
    task->reduce_scatterv_ring.get_count  = ucc_tl_cuda_reduce_scatterv_get_count;
    task->reduce_scatterv_ring.get_offset = ucc_tl_cuda_reduce_scatterv_get_offset;
    task->reduce_scatterv_ring.dt         = coll_args->args.dst.info_v.datatype;
    task->bar                             = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
