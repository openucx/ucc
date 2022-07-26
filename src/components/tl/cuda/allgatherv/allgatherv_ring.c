/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "tl_cuda_ring.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

ucc_status_t ucc_tl_cuda_allgatherv_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_cuda_task_put(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    ucc_rank_t          trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t        status;
    int                 chunk;

    for (chunk = 0; chunk < task->allgatherv_ring.num_chunks; chunk++) {
        set_rank_step(task, trank, 0, chunk);
    }
    ucc_memory_cpu_store_fence();
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team  = TASK_TEAM(task);
    return ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
}

static inline void
ucc_tl_cuda_allgatherv_ring_size_offset(ucc_tl_cuda_task_t *task, int block,
                                        int frag, int chunk, int ring,
                                        size_t *ring_frag_size,
                                        size_t *block_offset, size_t *frag_offset,
                                        size_t *chunk_offset, size_t *ring_frag_offset)
{
    ucc_datatype_t dt_size = ucc_dt_size(task->allgatherv_ring.dt);
    int            nrings  = task->allgatherv_ring.num_rings;
    int            nfrags  = task->allgatherv_ring.num_frags;
    int            nchunks = task->allgatherv_ring.num_chunks;
    size_t block_size, frag_size, chunk_size;

    block_size       = task->allgatherv_ring.get_count(task, block) * dt_size;
    frag_size        = ucc_buffer_block_count_aligned(block_size, nfrags, frag, 64);
    chunk_size       = ucc_buffer_block_count_aligned(frag_size, nchunks, chunk, 64);
    *ring_frag_size  = ucc_buffer_block_count_aligned(chunk_size, nrings, ring, 64);

    *block_offset     = task->allgatherv_ring.get_offset(task, block) * dt_size;
    *frag_offset      = ucc_buffer_block_offset_aligned(block_size, nfrags, frag, 64);
    *chunk_offset     = ucc_buffer_block_offset_aligned(frag_size, nchunks, chunk, 64);
    *ring_frag_offset = ucc_buffer_block_offset_aligned(chunk_size, nrings, ring, 64);
}

static inline int
ucc_tl_cuda_check_peers_ready(ucc_tl_cuda_task_t *task, int chunk, int step)
{
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          tsize  = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t          trank  = UCC_TL_TEAM_RANK(team);
    int                 nsteps = (tsize == 2) ? tsize: tsize - 1;
    int                 nrings = task->allgatherv_ring.num_rings;
    int ring, l, ll, r, rr, substep;
    ucc_rank_t peer;

    substep = step % nsteps;
    for (ring = 0; ring < nrings; ring++) {
        peer = get_send_to(team, trank, tsize, ring);
        r    = get_rank_step(task, peer, chunk);

        peer = get_send_to(team, peer, tsize, ring);
        rr   = get_rank_step(task, peer, chunk);

        peer = get_recv_from(team, trank, tsize, ring);
        l    = get_rank_step(task, peer, chunk);

        peer = get_recv_from(team, peer, tsize, ring);
        ll   = get_rank_step(task, peer, chunk);

        if (substep == 0) {
            if ((l < step) || (r < step) || (rr < step)) {
                return 0;
            }
        } else if (substep == nsteps - 1) {
            if ((l < step) || (r < step) || (ll < step)) {
                return 0;
            }
        } else {
            if ((l < step) || (r < step)) {
                return 0;
            }
        }
    }

    return 1;
}

static inline size_t get_scratch_size(ucc_tl_cuda_team_t *team, int nrings,
                                      int nchunks)
{
    return ucc_align_down_pow2((UCC_TL_CUDA_TEAM_LIB(team)->cfg.scratch_size /
                                nrings / nchunks /2), 64) * 2 * nrings * nchunks;
}

static inline size_t
ucc_tl_cuda_allgatherv_ring_scratch_offset(ucc_tl_cuda_task_t *task, int chunk,
                                           int ring)
{
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    int                 nrings  = task->allgatherv_ring.num_rings;
    int                 nchunks = task->allgatherv_ring.num_chunks;
    size_t              ssize   = get_scratch_size(team, nrings, nchunks);
    size_t chunk_size, chunk_offset;

    chunk_size   = ucc_buffer_block_count(ssize/2, nchunks, chunk);
    chunk_offset = ucc_buffer_block_offset(ssize/2, nchunks, chunk);
    return ucc_buffer_block_offset(chunk_size, nrings, ring) + chunk_offset;
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_progress_ring(ucc_tl_cuda_task_t *task,
                                                       int chunk)
{
    ucc_tl_cuda_team_t *team     = TASK_TEAM(task);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_rank_t          trank    = UCC_TL_TEAM_RANK(team);
    int                 tsize    = (int)UCC_TL_TEAM_SIZE(team);
    void               *rbuf     = task->allgatherv_ring.rbuf;
    int                 nsteps   = (tsize == 2) ? tsize: tsize - 1;
    int                 nrings   = task->allgatherv_ring.num_rings;
    int                 nchunks  = task->allgatherv_ring.num_chunks;
    size_t              ssize    = get_scratch_size(team, nrings, nchunks);
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t eargs_loc, eargs_rem;
    ucc_ee_executor_task_t *etask;
    void *dbuf1, *dbuf2, *sbuf1, *sbuf2;
    int step, frag, frag_step, i, ring;
    ucc_rank_t peer_block, sendto, recvfrom;
    ucc_status_t st;
    size_t remote_offset, local_offset, frag_offset,
           block_offset, ring_frag_offset, chunk_offset, ring_frag_size,
           frag_count1, frag_count2;
    size_t ring_scratch_offset;

    st = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(st != UCC_OK)) {
        return st;
    }

    step = get_rank_step(task, trank, chunk);
    while (step < (nsteps * task->allgatherv_ring.num_frags)) {
        if ((task->allgatherv_ring.exec_task[2*chunk] != NULL) ||
            (task->allgatherv_ring.exec_task[2*chunk + 1] != NULL)) {
            for (i = 1; i >= 0; i--) {
                etask = task->allgatherv_ring.exec_task[2*chunk + i];
                if (etask != NULL) {
                    st = ucc_ee_executor_task_test(etask);
                    if (st == UCC_OK) {
                        ucc_ee_executor_task_finalize(etask);
                        task->allgatherv_ring.exec_task[2*chunk + i] = NULL;
                    } else {
                        if (ucc_likely(st > 0)) {
                            return UCC_INPROGRESS;
                        }
                        return st;
                    }
                }
            }
            step++;
            set_rank_step(task, trank, step, chunk);
            continue;
        }

        frag      = step / nsteps;
        frag_step = step % nsteps;
        if (!ucc_tl_cuda_check_peers_ready(task, chunk, step)) {
            return UCC_INPROGRESS;
        }

        if (step % 2) {
            remote_offset = ssize / 2;
            local_offset = 0;
        } else {
            remote_offset = 0;
            local_offset = ssize / 2;
        }
        eargs_loc.copy_multi.num_vectors = 0;
        eargs_rem.copy_multi.num_vectors = 0;

        for (ring = 0; ring < nrings; ring++) {
            ring_scratch_offset = ucc_tl_cuda_allgatherv_ring_scratch_offset(task, chunk, ring);
            sendto   = get_send_to(team, trank, tsize, ring);
            recvfrom = get_recv_from(team, trank, tsize, ring);
            if (frag_step == nsteps - 1) {
                if (tsize != 2) {
                    peer_block = get_recv_block(team, trank, tsize, frag_step, ring);
                    ucc_tl_cuda_allgatherv_ring_size_offset(task, peer_block, frag, chunk, ring,
                                                            &ring_frag_size, &block_offset,
                                                            &frag_offset, &chunk_offset,
                                                            &ring_frag_offset);
                    sbuf1 = PTR_OFFSET(TASK_SCRATCH(task, recvfrom),
                                    local_offset + ring_scratch_offset);
                    dbuf1 = PTR_OFFSET(rbuf, block_offset + chunk_offset +
                                    frag_offset + ring_frag_offset);
                    frag_count1 = ring_frag_size;
                } else {
                    sbuf1 = NULL;
                    dbuf1 = NULL;
                }
                peer_block = get_send_block(team, trank, tsize, frag_step, ring);
                ucc_tl_cuda_allgatherv_ring_size_offset(task, peer_block, frag, chunk, ring,
                                                        &ring_frag_size, &block_offset,
                                                        &frag_offset, &chunk_offset,
                                                        &ring_frag_offset);
                sbuf2 = PTR_OFFSET(TASK_SCRATCH(task, trank), local_offset + ring_scratch_offset);
                dbuf2 = PTR_OFFSET(rbuf, block_offset + frag_offset +
                                         chunk_offset + ring_frag_offset);
                frag_count2 = ring_frag_size;
            } else {
                peer_block = get_send_block(team, trank, tsize, frag_step, ring);
                ucc_tl_cuda_allgatherv_ring_size_offset(task, peer_block, frag, chunk, ring,
                                                        &ring_frag_size, &block_offset,
                                                        &frag_offset, &chunk_offset,
                                                        &ring_frag_offset);
                if (frag_step == 0) {
                    if (UCC_IS_INPLACE(*args)) {
                        sbuf1 = PTR_OFFSET(rbuf, block_offset + frag_offset +
                                           ring_frag_offset + chunk_offset);
                        sbuf2 = NULL;
                        dbuf2 = NULL;
                    } else {
                        sbuf1 = PTR_OFFSET(task->allgatherv_ring.sbuf,
                                           frag_offset + ring_frag_offset + chunk_offset);
                        sbuf2 = sbuf1;
                        dbuf2 = PTR_OFFSET(rbuf, block_offset + frag_offset +
                                                 ring_frag_offset + chunk_offset);
                    }
                    dbuf1 = PTR_OFFSET(TASK_SCRATCH(task, sendto),
                                        remote_offset + ring_scratch_offset);
                } else {
                    sbuf1  = PTR_OFFSET(TASK_SCRATCH(task, trank),
                                        local_offset + ring_scratch_offset);
                    sbuf2 = sbuf1;
                    dbuf1 = PTR_OFFSET(TASK_SCRATCH(task, sendto),
                                       remote_offset + ring_scratch_offset);
                    dbuf2 = PTR_OFFSET(rbuf, block_offset + frag_offset +
                                       ring_frag_offset + chunk_offset);
                }
                frag_count1 = ring_frag_size;
                frag_count2 = ring_frag_size;
            }

            if (dbuf1 != NULL) {
                eargs_rem.task_type               = UCC_EE_EXECUTOR_TASK_TYPE_COPY_MULTI;
                eargs_rem.copy_multi.src[ring]    = sbuf1;
                eargs_rem.copy_multi.dst[ring]    = dbuf1;
                eargs_rem.copy_multi.counts[ring] = frag_count1;
                eargs_rem.copy_multi.num_vectors++;
            }

            if (dbuf2 != NULL) {
                eargs_loc.task_type               = UCC_EE_EXECUTOR_TASK_TYPE_COPY_MULTI;
                eargs_loc.copy_multi.src[ring]    = sbuf2;
                eargs_loc.copy_multi.dst[ring]    = dbuf2;
                eargs_loc.copy_multi.counts[ring] = frag_count2;
                eargs_loc.copy_multi.num_vectors++;
            }
        }

        if (eargs_rem.copy_multi.num_vectors != 0) {
            st = ucc_ee_executor_task_post(exec, &eargs_rem,
                                        &task->allgatherv_ring.exec_task[2*chunk]);
            if (ucc_unlikely(st != UCC_OK)) {
                return st;
            }
        }

        if (eargs_loc.copy_multi.num_vectors != 0) {
            st = ucc_ee_executor_task_post(exec, &eargs_loc,
                                           &task->allgatherv_ring.exec_task[2*chunk + 1]);
            if (ucc_unlikely(st != UCC_OK)) {
                return st;
            }
        }
    }

    return UCC_OK;
}

void ucc_tl_cuda_allgatherv_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_status_t st;
    int chunk, num_done;

    task->super.status = UCC_INPROGRESS;
    switch (task->allgatherv_ring.stage) {
    case RING_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.status = UCC_INPROGRESS;
            return;
        }
        st = ucc_tl_cuda_allgatherv_ring_setup_start(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->allgatherv_ring.stage = RING_STAGE_SETUP;
    case RING_STAGE_SETUP:
        st = ucc_tl_cuda_allgatherv_ring_setup_test(task);
        if (st != UCC_OK) {
            task->super.status = st;
            return;
        }
        task->allgatherv_ring.stage = RING_STAGE_RING;
    case RING_STAGE_RING:
        num_done = 0;
        for (chunk = 0; chunk < task->allgatherv_ring.num_chunks; chunk++) {
            st = ucc_tl_cuda_allgatherv_ring_progress_ring(task, chunk);
            if (ucc_unlikely(st < 0)) {
                task->super.status = st;
                return;
            } else if (st == UCC_OK) {
                num_done++;
            }
        }

        if (num_done != task->allgatherv_ring.num_chunks) {
            return;
        }

        st = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.status = st;
            return;
        }

        task->allgatherv_ring.stage = RING_STAGE_BARRIER;
    default:
        ucc_assert(task->allgatherv_ring.stage == RING_STAGE_BARRIER);
        break;
    }

    task->super.status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team),
                                                      task->bar);
    if (task->super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_tl_cuda_lib_t  *lib     = UCC_TL_CUDA_TEAM_LIB(team);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_rank_t          tsize   = UCC_TL_TEAM_SIZE(team);
    int                 nrings  = lib->cfg.allgather_ring_max_rings;
    int                 nchunks = lib->cfg.allgather_ring_num_chunks;
    size_t              ssize;
    size_t              send_size;
    size_t              frag_size;
    ucc_rank_t          i;

    nrings = ucc_min(nrings, team->topo->num_rings);
    nrings = ucc_min(nrings, UCC_EE_EXECUTOR_NUM_COPY_BUFS);
    ssize  = get_scratch_size(team, nrings, nchunks);
    task->allgatherv_ring.sbuf         = args->src.info.buffer;
    task->allgatherv_ring.num_rings    = nrings;
    task->allgatherv_ring.num_chunks   = nchunks;
    if (args->coll_type == UCC_COLL_TYPE_ALLGATHERV) {
        task->allgatherv_ring.rbuf     = args->dst.info_v.buffer;
    } else {
        task->allgatherv_ring.rbuf     = args->dst.info.buffer;
    }

    send_size = task->allgatherv_ring.get_count(task, 0);
    for (i = 1; i < tsize; i++) {
        send_size = ucc_max(send_size, task->allgatherv_ring.get_count(task, i));
    }

    if (send_size == 0) {
        task->super.status = UCC_OK;
        return ucc_task_complete(&task->super);
    }

    memset(task->allgatherv_ring.exec_task, 0,
           2 *  task->allgatherv_ring.num_chunks *
           sizeof(ucc_ee_executor_task_t*));

    send_size = ucc_dt_size(task->allgatherv_ring.dt) * send_size;
    frag_size = ucc_min(ssize /  2, send_size);
    task->allgatherv_ring.num_frags = ucc_div_round_up(send_size, frag_size);
    task->allgatherv_ring.stage     = RING_STAGE_SYNC;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_cuda_allgatherv_ring_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *     tl_team,
                                              ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_task_t *task = ucc_tl_cuda_task_init(coll_args, team);

    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    task->allgatherv_ring.get_count  = ucc_tl_cuda_allgatherv_get_count;
    task->allgatherv_ring.get_offset = ucc_tl_cuda_allgatherv_get_offset;
    task->allgatherv_ring.dt         = coll_args->args.dst.info_v.datatype;

    task->super.flags               |= UCC_COLL_TASK_FLAG_EXECUTOR;
    task->super.post                 = ucc_tl_cuda_allgatherv_ring_start;
    task->super.triggered_post       = ucc_triggered_post;
    task->super.progress             = ucc_tl_cuda_allgatherv_ring_progress;
    task->super.finalize             = ucc_tl_cuda_allgatherv_ring_finalize;
    task->bar                        = TASK_BAR(task);

    *task_p = &task->super;
    return UCC_OK;
}
