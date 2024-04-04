/**
 * Copyright(c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "allreduce_sliding_window.h"
#include "../allgather/allgather.h"
#include "../barrier/barrier.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"

ucc_status_t ucc_tl_ucp_barrier_knomial_start(ucc_coll_task_t *task);

static ucc_status_t ucc_tl_ucp_allreduce_sliding_window_register(
    ucp_context_h ucp_context, ucc_tl_ucp_team_t *tl_team,
    struct ucc_tl_ucp_allreduce_sw_export_buf *ebuf, void *packed_memh)
{
    ucs_status_t         ucs_status;
    ucp_mem_map_params_t params = {0};

    ebuf->ucp_context = ucp_context;

    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = packed_memh;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    if (UCS_OK != ucs_status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "import using ucp_mem_map() returned error: %s\n",
                 ucs_status_string(ucs_status));
        return ucs_status_to_ucc_status(ucs_status);
    }

    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        ucs_status_t unmap_status = ucp_mem_unmap(ucp_context, ebuf->memh);
        tl_error(UCC_TL_TEAM_LIB(tl_team),
            "ucp_rkey_pack() returned error: %s%s\n",
            ucs_status_string(ucs_status),
            unmap_status == UCS_OK ? "" : 
            ". While handling this error, unmapping the memh had an error\n");
        return ucs_status_to_ucc_status(ucs_status);
    }

    return UCC_OK;
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_reset_buf(ucc_tl_ucp_allreduce_sw_buf_t *buf)
{
    buf->state   = FREE;
    buf->count   = 0;
    buf->bytes   = 0;
    buf->ucp_req = NULL;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(
    ucc_tl_ucp_allreduce_sw_pipeline_t *pipe, ucc_rank_t rank,
    size_t put_window_size)
{
    int i;

    pipe->avail_buffs   = pipe->num_buffers;
    pipe->src_rank      = pipe->dst_rank       = rank;
    pipe->get_idx       = pipe->red_idx        = 0;
    pipe->done_get      = pipe->done_red       = 0;
    pipe->done_put      = pipe->posted_put     = 0;
    pipe->count_issued  = pipe->count_received = 0;
    pipe->count_reduced = pipe->count_serviced = 0;
    pipe->my_count      = pipe->my_offset      = 0;

    ucc_tl_ucp_allreduce_sliding_window_reset_buf(&pipe->accbuf);
    for (i = 0; i < pipe->num_buffers; i++) {
        ucc_tl_ucp_allreduce_sliding_window_reset_buf(&pipe->getbuf[i]);
    }

    for (i = 0; i < put_window_size; i++) {
        pipe->put_requests[i] = NULL;
    }
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_start(ucc_coll_task_t *coll_task)
{
    ucc_base_coll_args_t *coll_args   = &coll_task->bargs;
    ucc_tl_ucp_task_t    *task        = ucc_derived_of(coll_task,
                                                     ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t    *team        = TASK_TEAM(task);
    ucc_tl_ucp_context_t *tl_ctx      = UCC_TL_UCP_TEAM_CTX(team);
    ucc_rank_t            rank        = UCC_TL_TEAM_RANK(team);
    uint32_t              count_total = coll_task->bargs.args.dst.info.count;
    ucc_rank_t            size        = coll_task->team->params.size;
    ucc_datatype_t        dtype       = TASK_ARGS(task).dst.info.datatype;
    size_t                dt_size     = ucc_dt_size(dtype);
    int                   inplace     = UCC_IS_INPLACE(coll_args->args);
    ucc_status_t          status      = UCC_OK;
    int                   put_window_size = UCC_TL_UCP_TEAM_LIB(team)
                                 ->cfg.allreduce_sliding_window_put_window_size;
    ucc_tl_ucp_allreduce_sw_pipeline_t *pipe =
        task->allreduce_sliding_window.pipe;
    ucc_tl_ucp_allreduce_sw_host_allgather_t *allgather_data =
        task->allreduce_sliding_window.allgather_data;
    size_t allgather_size = sizeof(ucc_tl_ucp_allreduce_sw_host_allgather_t);
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t *gwbi_p =
        coll_args->args.global_work_buffer;

    ucc_base_coll_args_t bargs    = {
        .mask = 0,
        .args = {
            .coll_type = UCC_COLL_TYPE_ALLGATHER,
            .mask      = 0,
            .src.info = {.buffer   = allgather_data,
                         .count    = allgather_size,
                         .datatype = UCC_DT_UINT8,
                         .mem_type = UCC_MEMORY_TYPE_HOST},
            .dst.info = {.buffer   = PTR_OFFSET(allgather_data, allgather_size),
                         .count    = allgather_size * size,
                         .datatype = UCC_DT_UINT8,
                         .mem_type = UCC_MEMORY_TYPE_HOST}
        }
    };

    // Register the src and dst bufs
    if (!inplace) {
        status = ucc_tl_ucp_allreduce_sliding_window_register(
            tl_ctx->worker.ucp_context, team,
            task->allreduce_sliding_window.src_ebuf, gwbi_p->packed_src_memh);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to register src memh: %s",
                        ucc_status_string(status));
            goto out;
        }
        memcpy(allgather_data->packed_src_key,
               task->allreduce_sliding_window.src_ebuf->packed_key,
               task->allreduce_sliding_window.src_ebuf->packed_key_len);
    }

    status = ucc_tl_ucp_allreduce_sliding_window_register(
        tl_ctx->worker.ucp_context, team,
        task->allreduce_sliding_window.dst_ebuf, gwbi_p->packed_dst_memh);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task), "failed to register dst memh: %s",
                    ucc_status_string(status));
        goto out;
    }
    memcpy(allgather_data->packed_dst_key,
           task->allreduce_sliding_window.dst_ebuf->packed_key,
           task->allreduce_sliding_window.dst_ebuf->packed_key_len);

    UCC_CHECK_GOTO(ucc_tl_ucp_allgather_ring_init(&bargs,
                    &team->super.super,
                    &task->allreduce_sliding_window.allgather_task),
        out, status);

    UCC_CHECK_GOTO(ucc_tl_ucp_allgather_ring_start(
                    task->allreduce_sliding_window.allgather_task),
        out, status);

    if (put_window_size <= 0)
        put_window_size = size;

    ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(
        pipe, rank, put_window_size);

    pipe->my_count  = count_total / size;
    pipe->my_offset = pipe->my_count * dt_size * rank;
    if (rank == size - 1) {
        pipe->my_count += count_total % size;
    }

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    task->allreduce_sliding_window.reduce_task  = NULL;
    task->allreduce_sliding_window.barrier_task = NULL;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);

out:
    ucc_tl_ucp_allreduce_sliding_window_free_task(coll_task);
    ucc_tl_ucp_allreduce_sliding_window_free_pipe(coll_task);
    ucc_tl_ucp_coll_finalize(task->allreduce_sliding_window.allgather_task);
    tl_error(UCC_TASK_LIB(task), "failed to start allreduce sliding window: %s", ucc_status_string(status));
    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       st   = UCC_OK;

    ucc_tl_ucp_allreduce_sliding_window_free_task(coll_task);
    ucc_tl_ucp_allreduce_sliding_window_free_pipe(coll_task);

    st = ucc_tl_ucp_coll_finalize(coll_task);

    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to finalize collective");
    }

    return st;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_reduction(
    ucc_coll_task_t *coll_task, ucc_tl_ucp_allreduce_sw_buf_t *accbuf,
    ucc_tl_ucp_allreduce_sw_buf_t *getbuf)
{
    ucc_status_t       status = UCC_OK;
    ucc_tl_ucp_task_t *task   = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args   = &TASK_ARGS(task);
    ucc_datatype_t     dt     = TASK_ARGS(task).dst.info.datatype;

    status =
        ucc_dt_reduce(accbuf->buf, getbuf->buf, accbuf->buf, accbuf->count, dt,
                      args, 0, 0, task->super.executor,
                      &task->allreduce_sliding_window.reduce_task);

    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failed to perform dt reduction");
        task->super.status = status;
        return;
    }
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_test_reduction(ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;

    #define SAVE_STATE(_phase)

    EXEC_TASK_TEST(NULL,
                   "failed to perform dt reduction",
                   task->allreduce_sliding_window.reduce_task);

    // If it didn't complete, we would have returned by now. So, clear the flag
    task->allreduce_sliding_window.reduce_task = NULL;
}

static inline ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_req_test(ucs_status_ptr_t   request,
                                             ucc_tl_ucp_task_t *task)
{
    if (request == NULL) {
        return UCC_OK;
    } else if (UCS_PTR_IS_ERR(request)) {
        tl_error(UCC_TASK_LIB(task), "unable to complete UCX request=%p: %d\n",
                 request, UCS_PTR_STATUS(request));
        return ucs_status_to_ucc_status(UCS_PTR_STATUS(request));
    } else {
        return ucs_status_to_ucc_status(ucp_request_check_status(request));
    }
}

static inline void ucc_tl_ucp_allreduce_sliding_window_key_exchange_progress(
    ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_task_t   *allgather_task =
                           task->allreduce_sliding_window.allgather_task;
    ucc_status_t       status    = allgather_task->super.status;

    if (status < 0) {
        goto err;
    }
    if (UCC_INPROGRESS == status) {
        ucc_tl_ucp_allgather_ring_progress(allgather_task);
        return;
    }
    ucc_assert(status == UCC_OK);

    // copy from allgather recvbuf into sliding_window task
    UCC_CHECK_GOTO(
        ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(task),
        err, status);

out:
    ucc_tl_ucp_coll_finalize(allgather_task);
    task->allreduce_sliding_window.allgather_task = NULL;
    return;
err:
    ucc_tl_ucp_allreduce_sliding_window_free_task(coll_task);
    ucc_tl_ucp_allreduce_sliding_window_free_pipe(coll_task);
    tl_error(coll_task->team->context->lib,
                "key exchange failure: %s",
                ucc_status_string(status));
    goto out;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_free_rkeys(
    ucc_coll_task_t *coll_task)
{
    int                i;
    ucc_base_team_t   *team      = coll_task->team;
    ucc_rank_t         team_size = (ucc_rank_t)team->params.size;
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    int                inplace   = UCC_IS_INPLACE(coll_task->bargs.args);

    for (i = 0; i < team_size; i++) {
        if (!inplace) {
            ucp_rkey_destroy(task->allreduce_sliding_window.src_rkeys[i]);
        }
        ucp_rkey_destroy(task->allreduce_sliding_window.dst_rkeys[i]);
    }
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_barrier(ucc_coll_task_t *coll_task)
{
    ucc_status_t       status;
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_base_team_t   *team = coll_task->team;

    ucc_base_coll_args_t coll_args = {
        .team           = coll_task->team->params.team,
        .args.coll_type = UCC_COLL_TYPE_BARRIER,
    };

    status = ucc_tl_ucp_coll_init(&coll_args, team,
                                  &task->allreduce_sliding_window.barrier_task);
    if (status < 0) {
        tl_error(coll_task->team->context->lib,
                 "failure during sliding window barrier init: %s",
                 ucc_status_string(status));
        task->super.status = status;
        return;
    }

    status = ucc_tl_ucp_barrier_knomial_start(
        task->allreduce_sliding_window.barrier_task);
    if (status < 0) {
        tl_error(coll_task->team->context->lib,
                 "failure during sliding window barrier start: %s",
                 ucc_status_string(status));
        task->super.status = status;
    }
}

void ucc_tl_ucp_allreduce_sliding_window_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_allreduce_sw_buf_t *redbuf;
    ucc_tl_ucp_allreduce_sw_buf_t *getbuf;
    size_t                         remaining_elems;
    size_t                         get_idx;
    size_t                         count;
    size_t                         get_offset;
    size_t                         data_size;
    ucc_rank_t                     src_rank;
    ucc_rank_t                     dst_rank;
    void                          *src_addr;
    void                          *dst_addr;
    ucs_status_ptr_t               request;
    size_t                         red_idx;
    size_t                         put_offset;
    int                            window;
    int                            put_idx;
    ucp_ep_h                       ep;
    ucc_tl_ucp_task_t *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_rank_t         size    = (ucc_rank_t)task->subset.map.ep_num;
    ucc_datatype_t     dtype   = TASK_ARGS(task).dst.info.datatype;
    size_t             dt_size = ucc_dt_size(dtype);
    uint32_t           host_team_size = size;
    ucc_base_team_t   *base_team      = coll_task->team;
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_allreduce_sw_pipeline_t *pipe =
        task->allreduce_sliding_window.pipe;
    ucc_tl_ucp_context_t          *tl_ctx    = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_tl_ucp_allreduce_sw_buf_t *accbuf    = &pipe->accbuf;
    ucp_request_param_t            req_param = {0};
    int                            i         = 0;
    ucc_coll_task_t *allgather_task     =
        task->allreduce_sliding_window.allgather_task;
    ucc_coll_task_t *barrier_task       =
        task->allreduce_sliding_window.barrier_task;
    ucc_ee_executor_task_t **reduce_task = 
        &task->allreduce_sliding_window.reduce_task;
    int              put_window_size =
        UCC_TL_UCP_TEAM_LIB(tl_team)->
            cfg.allreduce_sliding_window_put_window_size;

    ucc_assert(host_team_size > 0);

    if (barrier_task != NULL) {
        // mark sliding window task complete once barrier finishes
        if (barrier_task->super.status == UCC_OK) {
            ucc_tl_ucp_put_task(
                ucc_derived_of(task->allreduce_sliding_window.barrier_task,
                               ucc_tl_ucp_task_t));
            task->allreduce_sliding_window.barrier_task = NULL;
            task->super.status                          = UCC_OK;
        }

        ucc_assert(barrier_task->super.status >= 0);
        return;
    }

    if (allgather_task != NULL) {
        ucc_tl_ucp_allreduce_sliding_window_key_exchange_progress(coll_task);
        return;
    }

    if (*reduce_task != NULL) {
        // We've previously started a reduction on the accbuf that hasn't yet
        // completed.
        ucc_tl_ucp_allreduce_sliding_window_test_reduction(task);

        if (*reduce_task != NULL) {
            return;
        }
    }

    if (pipe->count_serviced < pipe->my_count) {
        if ((pipe->count_received < pipe->my_count) &&
            (pipe->done_get < host_team_size) && (pipe->avail_buffs > 0) &&
            (accbuf->state != REDUCED && accbuf->state != SENDING)) {
            remaining_elems = pipe->my_count - pipe->count_received;
            get_idx         = pipe->get_idx % pipe->num_buffers;
            count      = ucc_min(pipe->buffer_size / dt_size, remaining_elems);
            get_offset = pipe->count_received * dt_size + pipe->my_offset;
            data_size  = count * dt_size;
            src_rank   = pipe->src_rank;
            getbuf = accbuf->state == FREE ? accbuf : &pipe->getbuf[get_idx];
            src_addr = (char*)
                task->allreduce_sliding_window.sbufs[src_rank] + get_offset;
            dst_addr = getbuf->buf;

            ucc_assert(getbuf->state == FREE);

            getbuf->state   = RECVING;
            getbuf->count   = count;
            getbuf->bytes   = data_size;
            ucc_tl_ucp_get_ep(tl_team, src_rank, &ep);
            getbuf->ucp_req = ucp_get_nbx(
                ep, dst_addr,
                data_size, (uint64_t)src_addr,
                task->allreduce_sliding_window.src_rkeys[src_rank], &req_param);

            pipe->src_rank = (src_rank + 1) % host_team_size;

            if (getbuf != accbuf) {
                pipe->avail_buffs--;
                pipe->get_idx++;
            }

            pipe->done_get++;
            if (pipe->done_get == host_team_size) {
                pipe->count_received += count;
            }
        }

        if (accbuf->state == RECVING) {
            request = accbuf->ucp_req;
            if (ucc_tl_ucp_allreduce_sliding_window_req_test(request, task) ==
                UCC_OK) {
                if (request)
                    ucp_request_free(request);
                accbuf->state   = REDUCING;
                accbuf->ucp_req = NULL;
            }
        }

        red_idx = pipe->red_idx % pipe->num_buffers;
        redbuf  = &pipe->getbuf[red_idx];
        if (accbuf->state == REDUCING && redbuf->state == RECVING) {
            request = redbuf->ucp_req;
            if (ucc_tl_ucp_allreduce_sliding_window_req_test(request, task) ==
                UCC_OK) {
                if (request)
                    ucp_request_free(request);
                redbuf->state   = REDUCING;
                redbuf->ucp_req = NULL;

                ucc_tl_ucp_allreduce_sliding_window_reduction(coll_task, accbuf,
                                                              redbuf);

                ucc_tl_ucp_allreduce_sliding_window_test_reduction(task);

                if (*reduce_task != NULL) {
                    return;
                }

                redbuf->state = FREE;
                pipe->avail_buffs++;
                pipe->red_idx++;
                pipe->done_red++;

                if (pipe->done_red == host_team_size - 1) {
                    accbuf->state = REDUCED;
                    pipe->count_reduced += accbuf->count;
                }
            }
        }

        if ((pipe->count_serviced < pipe->count_reduced) &&
            (accbuf->state == REDUCED)) {
            data_size  = accbuf->bytes;
            put_offset = pipe->count_serviced * dt_size + pipe->my_offset;

            if (put_window_size <= 0)
                put_window_size = host_team_size;

            ucc_assert(put_window_size > 0);

            window = ucc_min(put_window_size,
                             host_team_size - pipe->posted_put);

            for (i = 0; i < window; i++) {
                dst_rank = pipe->dst_rank;
                src_addr = accbuf->buf;
                dst_addr = (char*)
                    task->allreduce_sliding_window.rbufs[dst_rank] + put_offset;
                put_idx = pipe->posted_put %
                          put_window_size;

                if (task->allreduce_sliding_window.put_requests[put_idx] !=
                    NULL) {
                    // We've already posted a put at this index that didn't yet
                    // complete, left this function and came back. Skip to check
                    // whether this request finished instead of overwriting it
                    // with another put
                    break;
                }

                ucp_worker_fence(tl_ctx->worker.ucp_worker);
                ucc_tl_ucp_get_ep(tl_team, dst_rank, &ep);
                task->allreduce_sliding_window.put_requests[put_idx] = 
                    ucp_put_nbx(
                        ep, src_addr,
                        data_size, (uint64_t)dst_addr,
                        task->allreduce_sliding_window.dst_rkeys[dst_rank],
                        &req_param);

                pipe->posted_put++;
                pipe->dst_rank = (dst_rank + 1) % host_team_size;
            }

            for (i = pipe->done_put; i < pipe->posted_put; i++) {
                put_idx = i % put_window_size;
                request = task->allreduce_sliding_window.put_requests[put_idx];

                // These are fenced, so if the first fails, the proceding will
                // too
                if (ucc_tl_ucp_allreduce_sliding_window_req_test(
                        request, task) != UCC_OK)
                    break;

                if (request != NULL)
                    ucp_request_free(request);
                task->allreduce_sliding_window.put_requests[put_idx] = NULL;
                pipe->done_put++;
            }

            if (pipe->done_put == host_team_size) {
                ucc_assert(pipe->avail_buffs == pipe->num_buffers);
                ucc_assert(pipe->done_get == host_team_size);
                ucc_assert(pipe->done_red == host_team_size - 1);
                ucc_assert(pipe->done_put == host_team_size);

                pipe->count_serviced += accbuf->count;

                ucc_tl_ucp_allreduce_sliding_window_reset_buf(accbuf);
                pipe->done_get = 0;
                pipe->done_red = pipe->done_put = pipe->posted_put = 0;

                for (i = 0; i < put_window_size;
                     i++) {
                    task->allreduce_sliding_window.put_requests[i] = NULL;
                }
            }
        }

        ucp_worker_progress(tl_ctx->worker.ucp_worker);
    }

    if (pipe->count_serviced == pipe->my_count) {
        ucc_tl_ucp_allreduce_sliding_window_barrier(coll_task);
        ucc_tl_ucp_allreduce_sliding_window_free_rkeys(coll_task);
    }
}
