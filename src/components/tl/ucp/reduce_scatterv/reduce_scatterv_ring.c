/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "components/mc/ucc_mc.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_dt_reduce.h"

static inline void send_completion_common(void *request, ucs_status_t status,
                                          void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in rs ring completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->tagged.send_completed++;
    if (request) {
        ucp_request_free(request);
    }
}

static void send_completion_1(void *request, ucs_status_t status,
                              void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->reduce_scatterv_ring.s_scratch_busy[0] = 0;
    send_completion_common(request, status, user_data);
}

static void send_completion_2(void *request, ucs_status_t status,
                              void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;

    task->reduce_scatterv_ring.s_scratch_busy[1] = 0;
    send_completion_common(request, status, user_data);
}

static inline void ucc_ring_frag_count(ucc_tl_ucp_task_t *task,
                                       ucc_rank_t block, size_t *frag_count)
{
    int    n_frags, frag;
    size_t block_count;

    n_frags = task->reduce_scatterv_ring.n_frags;
    frag    = task->reduce_scatterv_ring.frag;

    block_count = ucc_coll_args_get_count(
        &TASK_ARGS(task), TASK_ARGS(task).dst.info_v.counts, block);
    *frag_count = ucc_buffer_block_count(block_count, n_frags, frag);
}

static inline size_t get_block_offset(ucc_coll_args_t *args, ucc_rank_t block)
{
    size_t offset = 0;
    int    i;

    for (i = 0; i < block; i++) {
        offset += ucc_coll_args_get_count(args, args->dst.info_v.counts, i);
    }
    return offset;
}

static inline void ucc_ring_frag_block_offset(ucc_tl_ucp_task_t *task,
                                              ucc_rank_t         block,
                                              size_t *           block_offset,
                                              size_t *           frag_offset)
{
    int    n_frags, frag;
    size_t block_count;

    n_frags = task->reduce_scatterv_ring.n_frags;
    frag    = task->reduce_scatterv_ring.frag;

    block_count = ucc_coll_args_get_count(
        &TASK_ARGS(task), TASK_ARGS(task).dst.info_v.counts, block);
    *frag_offset  = ucc_buffer_block_offset(block_count, n_frags, frag);
    *block_offset = get_block_offset(&TASK_ARGS(task), block);
}

static void ucc_tl_ucp_reduce_scatterv_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *     task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t *       args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *     team = TASK_TEAM(task);
    ucc_rank_t              size = task->subset.map.ep_num;
    ucc_rank_t              rank = task->subset.myrank;
    void *                  sbuf = args->src.info.buffer;
    ucc_memory_type_t       mem_type = args->dst.info_v.mem_type;
    ucc_datatype_t          dt       = args->dst.info_v.datatype;
    size_t                  dt_size  = ucc_dt_size(dt);
    ucc_rank_t              sendto   = (rank + 1) % size;
    ucc_rank_t              recvfrom = (rank - 1 + size) % size;
    ucp_send_nbx_callback_t cb[2]    = {send_completion_1, send_completion_2};
    ucc_rank_t              prevblock, recv_data_from;
    ucc_status_t            status;
    size_t max_block_size, block_offset, frag_count, frag_offset, final_offset;
    int    step, is_avg, id;
    void * r_scratch, *s_scratch[2], *reduce_target;
    volatile char *busy;

    final_offset = 0;
    if (UCC_IS_INPLACE(*args)) {
        sbuf         = args->dst.info_v.buffer;
        final_offset = get_block_offset(args, UCC_TL_TEAM_RANK(team));
    }

    sendto   = ucc_ep_map_eval(task->reduce_scatterv_ring.inv_map, sendto);
    recvfrom = ucc_ep_map_eval(task->reduce_scatterv_ring.inv_map, recvfrom);

    max_block_size = task->reduce_scatterv_ring.max_block_count * dt_size;
    busy           = task->reduce_scatterv_ring.s_scratch_busy;
    r_scratch      = task->reduce_scatterv_ring.scratch;
    s_scratch[0]   = PTR_OFFSET(r_scratch, max_block_size);
    s_scratch[1]   = PTR_OFFSET(s_scratch[0], max_block_size);

    if (UCC_INPROGRESS == ucc_tl_ucp_test_ring(task)) {
        return;
    }
    while (task->tagged.recv_posted > 0) {
        /* always have at least 1 send completion, ie 1 free slot */
        ucc_assert(!busy[0] || !busy[1]);
        id            = busy[0] ? 1 : 0;
        reduce_target = s_scratch[id];
        step          = task->tagged.send_posted;
        prevblock     = (rank - 1 - step + size) % size;
        prevblock =
            ucc_ep_map_eval(task->reduce_scatterv_ring.inv_map, prevblock);
        /* reduction */
        ucc_assert(task->tagged.recv_posted == task->tagged.recv_completed);
        ucc_assert(task->tagged.recv_posted < size);

        ucc_ring_frag_count(task, prevblock, &frag_count);
        ucc_ring_frag_block_offset(task, prevblock, &block_offset,
                                   &frag_offset);
        if (task->tagged.recv_completed == size - 1) {
            reduce_target = PTR_OFFSET(args->dst.info_v.buffer,
                                       (frag_offset + final_offset) * dt_size);
        }
        is_avg =
            (args->op == UCC_OP_AVG) && (task->tagged.recv_completed == (size - 1));
        if (UCC_OK !=
            (status = ucc_dt_reduce(
                 r_scratch,
                 PTR_OFFSET(sbuf, (block_offset + frag_offset) * dt_size),
                 reduce_target, frag_count, dt, args,
                 is_avg ? UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA : 0,
                 AVG_ALPHA(task), task->reduce_scatterv_ring.executor,
                 &task->reduce_scatterv_ring.etask))) {
            tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
            task->super.status = status;
            return;
        }
        EXEC_TASK_WAIT(task->reduce_scatterv_ring.etask);
        if (task->tagged.recv_completed == size - 1) {
            task->tagged.recv_posted = task->tagged.recv_completed = 0;
            break;
        }
        ucc_assert(task->tagged.send_posted - task->tagged.send_completed <= 1);
        ucc_assert(task->tagged.send_posted < size);

        busy[id] = 1;
        UCPCHECK_GOTO(ucc_tl_ucp_send_cb(reduce_target, frag_count * dt_size,
                                         mem_type, sendto, team, task, cb[id], (void *)task),
                      task, out);

        recv_data_from = (rank - 2 - step + size) % size;
        recv_data_from =
            ucc_ep_map_eval(task->reduce_scatterv_ring.inv_map, recv_data_from);

        ucc_ring_frag_count(task, recv_data_from, &frag_count);

        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(r_scratch, frag_count * dt_size,
                                         mem_type, recvfrom, team, task),
                      task, out);

        if (UCC_INPROGRESS == ucc_tl_ucp_test_ring(task)) {
            return;
        }
    }
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }
    task->super.status = UCC_OK;
out:
    return;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatterv_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t *  args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ucc_rank_t         size     = task->subset.map.ep_num;
    ucc_rank_t         rank     = task->subset.myrank;
    ucc_rank_t         sendto   = (rank + 1) % size;
    ucc_rank_t         recvfrom = (rank - 1 + size) % size;
    ucc_datatype_t     dt       = args->dst.info_v.datatype;
    size_t             dt_size  = ucc_dt_size(dt);
    ucc_memory_type_t  mem_type = args->dst.info_v.mem_type;
    void *             sbuf     = args->src.info.buffer;
    int                step     = 0;
    size_t             block_offset, frag_count, frag_offset;
    void *             r_scratch;
    ucc_rank_t         send_block, recv_block;
    ucc_status_t       status;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    if (UCC_IS_INPLACE(*args)) {
        sbuf = args->dst.info_v.buffer;
    }
    status = ucc_coll_task_get_executor(&task->super,
                                        &task->reduce_scatter_ring.executor);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    sendto     = ucc_ep_map_eval(task->reduce_scatterv_ring.inv_map, sendto);
    recvfrom   = ucc_ep_map_eval(task->reduce_scatterv_ring.inv_map, recvfrom);
    r_scratch  = task->reduce_scatterv_ring.scratch;
    recv_block = (rank - 2 - step + size) % size;
    send_block = (rank - 1 - step + size) % size;
    recv_block =
        ucc_ep_map_eval(task->reduce_scatterv_ring.inv_map, recv_block);
    send_block =
        ucc_ep_map_eval(task->reduce_scatterv_ring.inv_map, send_block);

    ucc_ring_frag_count(task, recv_block, &frag_count);
    UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(r_scratch, frag_count * dt_size, mem_type,
                                     recvfrom, team, task),
                  task, out);

    ucc_ring_frag_count(task, send_block, &frag_count);
    ucc_ring_frag_block_offset(task, send_block, &block_offset, &frag_offset);
    UCPCHECK_GOTO(ucc_tl_ucp_send_nb(
                      PTR_OFFSET(sbuf, (block_offset + frag_offset) * dt_size),
                      frag_count * dt_size, mem_type, sendto, team, task),
                  task, out);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return task->super.status;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatterv_ring_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->reduce_scatterv_ring.inv_map.type != UCC_EP_MAP_FULL) {
        ucc_ep_map_destroy(&task->reduce_scatterv_ring.inv_map);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}

static ucc_status_t ucc_tl_ucp_reduce_scatterv_ring_init_subset(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_subset_t subset, int n_frags, int frag,
    void *scratch, size_t max_block_count)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_reduce_scatterv_ring_start;
    task->super.progress = ucc_tl_ucp_reduce_scatterv_ring_progress;
    task->super.finalize = ucc_tl_ucp_reduce_scatterv_ring_finalize;
    task->subset         = subset;

    if (task->subset.map.type != UCC_EP_MAP_FULL) {
        status = ucc_ep_map_create_inverse(task->subset.map,
                                           &task->reduce_scatterv_ring.inv_map);
        if (UCC_OK != status) {
            return status;
        }
    } else {
        task->reduce_scatterv_ring.inv_map.type   = UCC_EP_MAP_FULL;
        task->reduce_scatterv_ring.inv_map.ep_num = task->subset.map.ep_num;
    }
    task->reduce_scatterv_ring.n_frags           = n_frags;
    task->reduce_scatterv_ring.frag              = frag;
    task->reduce_scatterv_ring.scratch           = scratch;
    task->reduce_scatterv_ring.max_block_count   = max_block_count;
    task->reduce_scatterv_ring.s_scratch_busy[0] = 0;
    task->reduce_scatterv_ring.s_scratch_busy[1] = 0;
    *task_h                                      = &task->super;
    return UCC_OK;
}

static ucc_status_t
ucc_tl_ucp_reduce_scatterv_ring_sched_post(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    int             i;

    for (i = 0; i < schedule->n_tasks; i++) {
        schedule->tasks[i]->bargs.args.src = task->bargs.args.src;
        schedule->tasks[i]->bargs.args.dst = task->bargs.args.dst;
    }

    return ucc_schedule_start(task);
}

static ucc_status_t
ucc_tl_ucp_reduce_scatterv_ring_sched_finalize(ucc_coll_task_t *task)
{
    ucc_tl_ucp_schedule_t *schedule =
        ucc_derived_of(task, ucc_tl_ucp_schedule_t);
    ucc_status_t status;

    ucc_mc_free(schedule->scratch_mc_header);
    status = ucc_schedule_finalize(task);
    ucc_tl_ucp_put_schedule(&schedule->super.super);
    return status;
}

ucc_status_t
ucc_tl_ucp_reduce_scatterv_ring_init(ucc_base_coll_args_t *coll_args,
                                     ucc_base_team_t *     team,
                                     ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size     = UCC_TL_TEAM_SIZE(tl_team);
    ucc_datatype_t     dt       = coll_args->args.dst.info_v.datatype;
    size_t             dt_size  = ucc_dt_size(dt);
    ucc_memory_type_t  mem_type = coll_args->args.dst.info_v.mem_type;
    int                bidir =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_scatterv_ring_bidirectional;
    size_t                 to_alloc_per_set, max_segcount, count_per_set, count;
    ucc_tl_ucp_schedule_t *tl_schedule;
    ucc_schedule_t *       schedule;
    ucc_coll_task_t *      ctask;
    ucc_status_t           status;
    ucc_subset_t           s[2];
    int                    i, n_subsets;

    if (UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.reduce_avg_pre_op &&
        coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_IS_INPLACE(coll_args->args)) {
        count = ucc_coll_args_get_total_count(
            &coll_args->args, coll_args->args.dst.info_v.counts, size);
    } else {
        count = coll_args->args.src.info.count;
    }

    status = ucc_tl_ucp_get_schedule(tl_team, coll_args, &tl_schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    schedule    = &tl_schedule->super.super;
    /* if count == size then we have 1 elem per rank, not enough
       to split into 2 sets */
    n_subsets = (bidir && (count > size)) ? 2 : 1;

    s[0].myrank     = UCC_TL_TEAM_RANK(tl_team);
    s[0].map.type   = UCC_EP_MAP_FULL;
    s[0].map.ep_num = UCC_TL_TEAM_SIZE(tl_team);

    s[1].map    = ucc_ep_map_create_reverse(UCC_TL_TEAM_SIZE(tl_team));
    s[1].myrank = ucc_ep_map_eval(s[1].map, UCC_TL_TEAM_RANK(tl_team));

    if (coll_args->mask & UCC_BASE_CARGS_MAX_FRAG_COUNT) {
        max_segcount = coll_args->max_frag_count;
    } else {
        max_segcount = ucc_coll_args_get_max_count(
            &coll_args->args, coll_args->args.dst.info_v.counts, size);
    }
    /* in flight we can have 2 sends from 2 differnt blocks and 1 recv:
       need 3 * max_segcount of scratch per set */
    count_per_set = (max_segcount + n_subsets - 1) / n_subsets;

    to_alloc_per_set = count_per_set * 3;
    UCC_CHECK_GOTO(ucc_mc_alloc(&tl_schedule->scratch_mc_header,
                                to_alloc_per_set * dt_size * n_subsets,
                                mem_type),
                   out, status);

    for (i = 0; i < n_subsets; i++) {
        UCC_CHECK_GOTO(ucc_tl_ucp_reduce_scatterv_ring_init_subset(
                           coll_args, team, &ctask, s[i], n_subsets, i,
                           PTR_OFFSET(tl_schedule->scratch_mc_header->addr,
                                      to_alloc_per_set * i * dt_size),
                           count_per_set),
                       out_free, status);
        ctask->n_deps = 1;
        UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, ctask), out_free,
                       status);
        UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                           &schedule->super, UCC_EVENT_SCHEDULE_STARTED, ctask,
                           ucc_task_start_handler),
                       out_free, status);
    }
    schedule->super.flags   |= UCC_COLL_TASK_FLAG_EXECUTOR;
    schedule->super.post     = ucc_tl_ucp_reduce_scatterv_ring_sched_post;
    schedule->super.finalize = ucc_tl_ucp_reduce_scatterv_ring_sched_finalize;
    *task_h                  = &schedule->super;
    return UCC_OK;

out_free:
    ucc_mc_free(tl_schedule->scratch_mc_header);
out:
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}
