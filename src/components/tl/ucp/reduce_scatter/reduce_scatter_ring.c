/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatter.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "core/ucc_mc.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

static inline void ucc_ring_frag_count(ucc_tl_ucp_task_t *task, size_t count,
                                       ucc_rank_t block, size_t *frag_count)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    size_t             size = UCC_TL_TEAM_SIZE(team);
    int                n_frags, frag;
    size_t             block_count;

    n_frags = task->reduce_scatter_ring.n_frags;
    frag    = task->reduce_scatter_ring.frag;

    block_count = ucc_buffer_block_count(count, size, block);
    *frag_count = ucc_buffer_block_count(block_count, n_frags, frag);
}

static inline void ucc_ring_frag_block_offset(ucc_tl_ucp_task_t *task,
                                              size_t count, ucc_rank_t block,
                                              size_t *block_offset,
                                              size_t *frag_offset)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    size_t             size = UCC_TL_TEAM_SIZE(team);
    int                n_frags, frag;
    size_t             block_count;

    n_frags = task->reduce_scatter_ring.n_frags;
    frag    = task->reduce_scatter_ring.frag;

    block_count   = ucc_buffer_block_count(count, size, block);
    *frag_offset  = ucc_buffer_block_offset(block_count, n_frags, frag);
    *block_offset = ucc_buffer_block_offset(count, size, block);
}

static inline ucc_status_t ucc_tl_ucp_test_ring(ucc_tl_ucp_task_t *task)
{
    int polls = 0;
    while (polls++ < task->n_polls) {
        if (task->send_posted - task->send_completed <= 1 &&
            task->recv_posted == task->recv_completed) {
            return UCC_OK;
        }
        ucp_worker_progress(TASK_CTX(task)->ucp_worker);
    }
    return UCC_INPROGRESS;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t *  args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ucc_rank_t         size     = task->subset.map.ep_num;
    ucc_rank_t         rank     = task->subset.myrank;
    void *             sbuf     = args->src.info.buffer;
    ucc_memory_type_t  mem_type = args->dst.info.mem_type;
    size_t             count    = args->dst.info.count * size;
    ucc_datatype_t     dt       = args->dst.info.datatype;
    size_t             dt_size  = ucc_dt_size(dt);
    ucc_rank_t         sendto   = (rank + 1) % size;
    ucc_rank_t         recvfrom = (rank - 1 + size) % size;
    ucc_rank_t         prevblock, recv_data_from;
    ucc_status_t       status;
    size_t max_block_size, block_offset, frag_count, frag_offset, final_offset;
    int    step, is_avg;
    void  *r_scratch, *s_scratch[2];

    final_offset = 0;
    if (UCC_IS_INPLACE(*args)) {
        sbuf = args->dst.info.buffer;
        count /= size;
        final_offset =
            ucc_buffer_block_offset(count, size, UCC_TL_TEAM_RANK(team));
    }

    sendto   = ucc_ep_map_eval(task->reduce_scatter_ring.inv_map, sendto);
    recvfrom = ucc_ep_map_eval(task->reduce_scatter_ring.inv_map, recvfrom);

    max_block_size = task->reduce_scatter_ring.max_block_count * dt_size;
    r_scratch      = task->reduce_scatter_ring.scratch;
    s_scratch[0]   = PTR_OFFSET(r_scratch, max_block_size);
    s_scratch[1]   = PTR_OFFSET(s_scratch[0], max_block_size);

    if (UCC_INPROGRESS == ucc_tl_ucp_test_ring(task)) {
        return task->super.super.status;
    }
    while (task->recv_posted > 0) {
        void *reduce_target = s_scratch[(task->recv_completed - 1) % 2];
        step                = task->send_posted;
        prevblock           = (rank - 1 - step + size) % size;
        prevblock =
            ucc_ep_map_eval(task->reduce_scatter_ring.inv_map, prevblock);
        /* reduction */
        ucc_assert(task->recv_posted == task->recv_completed);
        ucc_assert(task->recv_posted < size);

        ucc_ring_frag_count(task, count, prevblock, &frag_count);
        ucc_ring_frag_block_offset(task, count, prevblock, &block_offset,
                                   &frag_offset);
        if (task->recv_completed == size - 1) {
            reduce_target = PTR_OFFSET(args->dst.info.buffer,
                                       (frag_offset + final_offset) * dt_size);
        }
        is_avg =
            (args->op == UCC_OP_AVG) && (task->recv_completed == (size - 1));
        if (UCC_OK !=
            (status = ucc_tl_ucp_reduce_multi(
                 r_scratch,
                 PTR_OFFSET(sbuf, (block_offset + frag_offset) * dt_size),
                 reduce_target, 1, frag_count, 0, dt, mem_type, task,
                 is_avg))) {
            tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
            task->super.super.status = status;
            return status;
        }
        if (task->recv_completed == size - 1) {
            task->recv_posted = task->recv_completed = 0;
            break;
        }
        ucc_assert(task->send_posted - task->send_completed <= 1);
        ucc_assert(task->send_posted < size);

        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(s_scratch[(task->send_posted - 1) % 2],
                                         frag_count * dt_size, mem_type, sendto,
                                         team, task),
                      task, out);

        recv_data_from = (rank - 2 - step + size) % size;
        recv_data_from =
            ucc_ep_map_eval(task->reduce_scatter_ring.inv_map, recv_data_from);

        ucc_ring_frag_count(task, count, recv_data_from, &frag_count);

        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(r_scratch, frag_count * dt_size, mem_type,
                                         recvfrom, team, task),
                      task, out);

        if (UCC_INPROGRESS == ucc_tl_ucp_test_ring(task)) {
            return task->super.super.status;
        }
    }
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return task->super.super.status;
    }
    task->super.super.status = UCC_OK;
out:
    return task->super.super.status;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t *  args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ucc_rank_t         size     = task->subset.map.ep_num;
    ucc_rank_t         rank     = task->subset.myrank;
    ucc_rank_t         sendto   = (rank + 1) % size;
    ucc_rank_t         recvfrom = (rank - 1 + size) % size;
    size_t             count    = args->dst.info.count * size;
    ucc_datatype_t     dt       = args->dst.info.datatype;
    size_t             dt_size  = ucc_dt_size(dt);
    ucc_memory_type_t  mem_type = args->dst.info.mem_type;
    void *             sbuf     = args->src.info.buffer;
    int                step     = 0;
    ucc_status_t       status;
    size_t             block_offset, frag_count, frag_offset;
    void              *r_scratch;
    ucc_rank_t         send_block, recv_block;

    if (UCC_IS_INPLACE(*args)) {
        sbuf = args->dst.info.buffer;
        count /= size;
    }
    task->super.super.status = UCC_INPROGRESS;
    ucc_tl_ucp_task_reset(task);

    sendto     = ucc_ep_map_eval(task->reduce_scatter_ring.inv_map, sendto);
    recvfrom   = ucc_ep_map_eval(task->reduce_scatter_ring.inv_map, recvfrom);
    r_scratch  = task->reduce_scatter_ring.scratch;
    recv_block = (rank - 2 - step + size) % size;
    send_block = (rank - 1 - step + size) % size;
    recv_block = ucc_ep_map_eval(task->reduce_scatter_ring.inv_map, recv_block);
    send_block = ucc_ep_map_eval(task->reduce_scatter_ring.inv_map, send_block);

    ucc_ring_frag_count(task, count, recv_block, &frag_count);
    UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(r_scratch, frag_count * dt_size, mem_type,
                                     recvfrom, team, task),
                  task, out);

    ucc_ring_frag_count(task, count, send_block, &frag_count);
    ucc_ring_frag_block_offset(task, count, send_block, &block_offset,
                               &frag_offset);
    UCPCHECK_GOTO(ucc_tl_ucp_send_nb(
                      PTR_OFFSET(sbuf, (block_offset + frag_offset) * dt_size),
                      frag_count * dt_size, mem_type, sendto, team, task),
                  task, out);

    status = ucc_tl_ucp_reduce_scatter_ring_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
out:
    return task->super.super.status;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->reduce_scatter_ring.inv_map.type != UCC_EP_MAP_FULL) {
        ucc_ep_map_destroy(&task->reduce_scatter_ring.inv_map);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}

static ucc_status_t ucc_tl_ucp_reduce_scatter_ring_init_subset(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_subset_t subset, int n_frags, int frag,
    void *scratch, size_t max_block_count)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_reduce_scatter_ring_start;
    task->super.progress = ucc_tl_ucp_reduce_scatter_ring_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatter_ring_finalize;
    task->subset         = subset;

    if (task->subset.map.type != UCC_EP_MAP_FULL) {
        status = ucc_ep_map_create_inverse(task->subset.map,
                                           &task->reduce_scatter_ring.inv_map);
        if (UCC_OK != status) {
            return status;
        }
    } else {
        task->reduce_scatter_ring.inv_map.type   = UCC_EP_MAP_FULL;
        task->reduce_scatter_ring.inv_map.ep_num = task->subset.map.ep_num;
    }
    task->reduce_scatter_ring.n_frags         = n_frags;
    task->reduce_scatter_ring.frag            = frag;
    task->reduce_scatter_ring.scratch         = scratch;
    task->reduce_scatter_ring.max_block_count = max_block_count;

    *task_h = &task->super;
    return UCC_OK;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_sched_post(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);

    return ucc_schedule_start(schedule);
}

static ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_sched_finalize(ucc_coll_task_t *task)
{
    ucc_tl_ucp_schedule_t *schedule = ucc_derived_of(task,
                                                     ucc_tl_ucp_schedule_t);
    ucc_status_t    status;

    ucc_mc_free(schedule->scratch_mc_header);
    status = ucc_schedule_finalize(task);
    ucc_tl_ucp_put_schedule(&schedule->super.super);
    return status;
}

ucc_status_t
ucc_tl_ucp_reduce_scatter_ring_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *     team,
                                    ucc_coll_task_t **    task_h)
{

    ucc_tl_ucp_team_t *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size     = UCC_TL_TEAM_SIZE(tl_team);
    size_t             count    = coll_args->args.dst.info.count * size;
    ucc_datatype_t     dt       = coll_args->args.dst.info.datatype;
    size_t             dt_size  = ucc_dt_size(dt);
    ucc_memory_type_t  mem_type = coll_args->args.dst.info.mem_type;
    int                bidir =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_scatter_ring_bidirectional;
    size_t                 to_alloc_per_set, max_segcount, count_per_set;
    ucc_tl_ucp_schedule_t *tl_schedule;
    ucc_schedule_t        *schedule;
    ucc_coll_task_t       *ctask;
    ucc_status_t           status;
    ucc_subset_t           s[2];
    int                    i, n_subsets;

    if (UCC_TL_TEAM_SIZE(tl_team) == 2) {
        return ucc_tl_ucp_reduce_scatter_knomial_init(coll_args, team, task_h);
    }
    if (UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_avg_pre_op &&
        coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    tl_schedule  = ucc_tl_ucp_get_schedule(tl_team, coll_args);
    schedule     = &tl_schedule->super.super;
    /* if count == size then we have 1 elem per rank, not enough
       to split into 2 sets */
    n_subsets    = (bidir && (count > size)) ? 2 : 1;

    s[0].myrank     = UCC_TL_TEAM_RANK(tl_team);
    s[0].map.type   = UCC_EP_MAP_FULL;
    s[0].map.ep_num = UCC_TL_TEAM_SIZE(tl_team);

    s[1].map    = ucc_ep_map_create_reverse(UCC_TL_TEAM_SIZE(tl_team));
    s[1].myrank = ucc_ep_map_eval(s[1].map, UCC_TL_TEAM_RANK(tl_team));

    if (UCC_IS_INPLACE(coll_args->args)) {
        count /= size;
    }

    count_per_set    = (count + n_subsets - 1) / n_subsets;
    max_segcount     = ucc_buffer_block_count(count_per_set, size, 0);
    /* in flight we can have 2 sends from 2 differnt blocks and 1 recv:
       need 3 * max_segcount of scratch per set */
    to_alloc_per_set = max_segcount * 3;
    status = ucc_mc_alloc(&tl_schedule->scratch_mc_header,
                          to_alloc_per_set * dt_size * n_subsets, mem_type);

    if (status != UCC_OK) {
        ucc_tl_ucp_put_schedule(schedule);
        return status;
    }

    for (i = 0; i < n_subsets; i++) {
        status = ucc_tl_ucp_reduce_scatter_ring_init_subset(
            coll_args, team, &ctask, s[i], n_subsets, i,
            PTR_OFFSET(tl_schedule->scratch_mc_header->addr,
                       to_alloc_per_set * i * dt_size), max_segcount);
        if (UCC_OK != status) {
            tl_error(UCC_TL_TEAM_LIB(tl_team), "failed to allocate ring task");
            return status;
        }
        ctask->n_deps = 1;
        ucc_schedule_add_task(schedule, ctask);
        ucc_event_manager_subscribe(&schedule->super.em,
                                    UCC_EVENT_SCHEDULE_STARTED, ctask,
                                    ucc_task_start_handler);
    }
    schedule->super.post     = ucc_tl_ucp_reduce_scatter_ring_sched_post;
    schedule->super.finalize = ucc_tl_ucp_reduce_scatter_ring_sched_finalize;
    *task_h                  = &schedule->super;
    return UCC_OK;
}
