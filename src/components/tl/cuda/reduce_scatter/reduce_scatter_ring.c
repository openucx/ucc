/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatter.h"
#include "components/ec/ucc_ec.h"
#include "tl_cuda_cache.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"

enum {
    REDUCE_SCATTER_RING_STAGE_SYNC,    /*< Wait for free SYNC segment */
    REDUCE_SCATTER_RING_STAGE_SETUP,   /*< Wait for memhandle setup to finish */
    REDUCE_SCATTER_RING_STAGE_RING,    /*< Ring algorithm is running */
    REDUCE_SCATTER_RING_STAGE_BAR,     /*< Ring algorithm is done, waiting for
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

static inline void set_rank_step(ucc_tl_cuda_task_t *task, ucc_rank_t rank,
                                 uint32_t step, int ring_id)
{
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);
    sync->seq_num = step;
}

static inline uint32_t get_rank_step(ucc_tl_cuda_task_t *task, ucc_rank_t rank,
                                     int ring_id)
{
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, rank);
    return sync->seq_num;
}

static inline size_t max_frag_size(size_t total_count, ucc_rank_t n_blocks,
                                   ucc_rank_t n_frags)
{
    size_t block_count = ucc_ring_block_count(total_count, n_blocks, 0);
    return ucc_ring_block_count(block_count, n_frags, 0);
}

ucc_status_t ucc_tl_cuda_reduce_scatter_ring_setup_start(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team = TASK_TEAM(task);
    ucc_tl_cuda_sync_t *sync = TASK_SYNC(task, UCC_TL_TEAM_RANK(team));
    ucc_status_t        status;

    memcpy(&sync->mem_info_dst, &task->reduce_scatter_ring.scratch_mem_info,
           sizeof(ucc_tl_cuda_mem_info_t));
    ucc_memory_bus_fence();
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
    if (ucc_unlikely(status != UCC_OK)) {
        goto exit_err;
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_reduce_scatter_ring_setup_test(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t          *team  = TASK_TEAM(task);
    ucc_rank_t                   trank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t                   tsize = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t                   recvfrom;
    volatile ucc_tl_cuda_sync_t *peer_sync;
    ucc_tl_cuda_cache_t         *cache;
    ucc_status_t                 status;

    status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), task->bar);
    if (status != UCC_OK) {
        return status;
    }

    recvfrom = get_recv_from(team, trank, tsize, 0);
    peer_sync = TASK_SYNC(task, recvfrom);
    cache = ucc_tl_cuda_get_cache(team, recvfrom);
    status = ucc_tl_cuda_map_memhandle(peer_sync->mem_info_dst.ptr,
                                       peer_sync->mem_info_dst.length,
                                       peer_sync->mem_info_dst.handle,
                                       &task->reduce_scatter_ring.peer_map_addr,
                                       cache);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(team), "ucc_tl_cuda_map_memhandle failed");
    }
    return status;
}

ucc_status_t
ucc_tl_cuda_reduce_scatter_ring_progress_ring(ucc_tl_cuda_task_t * task,
                                              uint32_t ring_id)
{
    ucc_tl_cuda_team_t *team     = TASK_TEAM(task);
    ucc_coll_args_t    *args     = &TASK_ARGS(task);
    ucc_rank_t          trank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize    = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t          sendto   = get_send_to(team, trank, tsize, ring_id);
    ucc_rank_t          recvfrom = get_recv_from(team, trank, tsize, ring_id);
    size_t              ccount   = args->dst.info.count * tsize;
    ucc_datatype_t      dt       = args->dst.info.datatype;
    void               *sbuf     = args->src.info.buffer;
    void               *dbuf     = args->dst.info.buffer;
    void               *scratch  = task->reduce_scatter_ring.scratch;
    ucc_tl_cuda_sync_t *sync;
    uint32_t step, send_step, recv_step;
    size_t block_count, frag_count, recv_block, max_count;
    size_t remote_offset, local_offset, frag_offset, block_offset;
    void *rsrc1, *rsrc2, *rdst;
    ucc_status_t st;

    if (UCC_IS_INPLACE(*args)) {
        ccount = args->dst.info.count;
        sbuf = args->dst.info.buffer;
        dbuf = PTR_OFFSET(args->dst.info.buffer,
                          (ccount / tsize) * trank * ucc_dt_size(dt));
    }
    step = get_rank_step(task, trank, ring_id);
    while (step < tsize) {
        send_step = get_rank_step(task, sendto, ring_id);
        recv_step = get_rank_step(task, recvfrom, ring_id);
        if ((send_step < step) || (recv_step < step)) {
            task->super.super.status = UCC_INPROGRESS;
            return task->super.super.status;
        }

        max_count = max_frag_size(ccount, tsize, 1);
        recv_block = get_recv_block(team, trank, tsize, step, ring_id);
        block_count = ucc_ring_block_count(ccount, tsize, recv_block);
        frag_count = ucc_ring_block_count(block_count, 1, ring_id);
        block_offset = ucc_ring_block_offset(ccount, tsize, recv_block);
        frag_offset = ucc_ring_block_offset(block_count, 1, ring_id);

        if (step % 2) {
            remote_offset = 0;
            local_offset = max_count * ucc_dt_size(dt);
        } else {
            remote_offset = max_count * ucc_dt_size(dt);
            local_offset = 0;
        }
        sync = TASK_SYNC(task, recvfrom);
        rsrc1 = PTR_OFFSET(sbuf, (block_offset + frag_offset) * ucc_dt_size(dt));
        rsrc2 = PTR_OFFSET(task->reduce_scatter_ring.peer_map_addr,
                           sync->mem_info_dst.offset + remote_offset);
        if (step == tsize - 1) {
            rdst = PTR_OFFSET(dbuf, frag_offset * ucc_dt_size(dt));
        } else {
            rdst = PTR_OFFSET(scratch, local_offset);
        }
        st = ucc_mc_reduce(rsrc1, rsrc2, rdst, frag_count, dt,
                           args->op, UCC_MEMORY_TYPE_CUDA);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return task->super.super.status;
        }
        step++;
        set_rank_step(task, trank, step, ring_id);
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_reduce_scatter_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_rank_t          trank   = UCC_TL_TEAM_RANK(team);
    ucc_status_t st;

    switch (task->reduce_scatter_ring.stage) {
    case REDUCE_SCATTER_RING_STAGE_SYNC:
        if (ucc_tl_cuda_get_sync(task) != UCC_OK) {
            task->super.super.status = UCC_INPROGRESS;
            return task->super.super.status;
        }
        set_rank_step(task, trank, 1, 0);
        st = ucc_tl_cuda_reduce_scatter_ring_setup_start(task);
        if (st != UCC_OK) {
            task->super.super.status = st;
            return task->super.super.status;
        }
        task->reduce_scatter_ring.stage = REDUCE_SCATTER_RING_STAGE_SETUP;
    case REDUCE_SCATTER_RING_STAGE_SETUP:
        st = ucc_tl_cuda_reduce_scatter_ring_setup_test(task);
        if (st != UCC_OK) {
            task->super.super.status = st;
            return task->super.super.status;
        }
        task->reduce_scatter_ring.stage = REDUCE_SCATTER_RING_STAGE_RING;
    case REDUCE_SCATTER_RING_STAGE_RING:
        /* TODO: add support for multiple rings, only ring 0 is used so far */
        st = ucc_tl_cuda_reduce_scatter_ring_progress_ring(task, 0);
        if (st != UCC_OK) {
            return task->super.super.status;
        }

        st = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), task->bar);
        if (ucc_unlikely(st != UCC_OK)) {
            task->super.super.status = st;
            return task->super.super.status;
        }

        task->reduce_scatter_ring.stage = REDUCE_SCATTER_RING_STAGE_BAR;
    default:
        ucc_assert(task->reduce_scatter_ring.stage ==
                   REDUCE_SCATTER_RING_STAGE_BAR);
        break;
    }

    task->super.super.status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team),
                                                            task->bar);
    if (task->super.super.status == UCC_OK) {
        ucc_tl_cuda_put_sync(task);
    }
    return task->super.super.status;
}

ucc_status_t ucc_tl_cuda_reduce_scatter_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task    = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);
    ucc_tl_cuda_team_t *team    = TASK_TEAM(task);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_rank_t          trank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          tsize   = UCC_TL_TEAM_SIZE(team);
    size_t              ccount  = args->dst.info.count * tsize;
    ucc_datatype_t      dt      = args->dst.info.datatype;
    void               *sbuf    = args->src.info.buffer;
    void               *scratch = task->reduce_scatter_ring.scratch;
    size_t block_count, block_offset, frag_count, frag_offset, block;
    ucc_status_t st;

    if (UCC_IS_INPLACE(*args)) {
        ccount = args->dst.info.count;
        sbuf = args->dst.info.buffer;
    }
    block = get_send_block(team, trank, tsize, 1, 0);
    block_count = ucc_ring_block_count(ccount, tsize, block);
    frag_count = ucc_ring_block_count(block_count, 1, 0);
    block_offset = ucc_ring_block_offset(ccount, tsize, block);
    frag_offset = ucc_ring_block_offset(block_count, 1, 0);
    st = ucc_mc_memcpy(scratch, PTR_OFFSET(sbuf, (block_offset + frag_offset) *
                       ucc_dt_size(dt)), frag_count * ucc_dt_size(dt),
                       UCC_MEMORY_TYPE_CUDA, UCC_MEMORY_TYPE_CUDA);
    if (ucc_unlikely(st != UCC_OK)) {
        task->super.super.status = st;
        return ucc_task_complete(coll_task);
    }
    task->reduce_scatter_ring.stage = REDUCE_SCATTER_RING_STAGE_SYNC;
    st = ucc_tl_cuda_reduce_scatter_ring_progress(coll_task);
    if (task->super.super.status == UCC_INPROGRESS) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t ucc_tl_cuda_reduce_scatter_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_cuda_task_t *task = ucc_derived_of(coll_task, ucc_tl_cuda_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_mc_free(task->reduce_scatter_ring.scratch_mc_header);
    ucc_tl_cuda_task_put(task);

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_reduce_scatter_ring_init(ucc_tl_cuda_task_t *task)
{
    ucc_tl_cuda_team_t *team   = TASK_TEAM(task);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_rank_t          tsize  = UCC_TL_TEAM_SIZE(team);
    size_t              ccount = args->dst.info.count * tsize;
    ucc_datatype_t      dt     = args->dst.info.datatype;
    ucc_status_t status;
    size_t frag_count;

    if (UCC_IS_INPLACE(*args)) {
        ccount = args->dst.info.count;
    }
    frag_count = max_frag_size(ccount, tsize, 1);
    status = ucc_mc_alloc(&task->reduce_scatter_ring.scratch_mc_header,
                          2 * frag_count * ucc_dt_size(dt),
                          UCC_MEMORY_TYPE_CUDA);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }
    task->reduce_scatter_ring.scratch =
                            task->reduce_scatter_ring.scratch_mc_header->addr;

    status = ucc_tl_cuda_mem_info_get(task->reduce_scatter_ring.scratch,
                                      2 * frag_count * ucc_dt_size(dt), team,
                                      &task->reduce_scatter_ring.scratch_mem_info);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_mc_free(task->reduce_scatter_ring.scratch_mc_header);
        return status;
    }

    task->super.post     = ucc_tl_cuda_reduce_scatter_ring_start;
    task->super.progress = ucc_tl_cuda_reduce_scatter_ring_progress;
    task->super.finalize = ucc_tl_cuda_reduce_scatter_ring_finalize;
    task->bar            = TASK_BAR(task);

    return UCC_OK;
}
