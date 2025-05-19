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
    ucc_rank_t put_window_size)
{
    int i;

    pipe->avail_buffs    = pipe->num_buffers;
    pipe->src_rank       = pipe->dst_rank       = rank;
    pipe->get_idx        = pipe->red_idx        = 0;
    pipe->done_get       = pipe->done_red       = 0;
    pipe->done_put       = pipe->posted_put     = 0;
    pipe->count_reduced  = pipe->count_serviced = 0;
    pipe->my_count       = pipe->my_offset      = 0;
    pipe->count_received = 0;

    ucc_tl_ucp_allreduce_sliding_window_reset_buf(&pipe->accbuf);
    for (i = 0; i < pipe->num_buffers; i++) {
        ucc_tl_ucp_allreduce_sliding_window_reset_buf(&pipe->getbuf[i]);
    }

    memset(pipe->put_requests, 0, put_window_size * sizeof(ucs_status_ptr_t));
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_start(ucc_coll_task_t *coll_task)
{
    ucc_base_coll_args_t *coll_args       = &coll_task->bargs;
    ucc_schedule_t       *schedule        = ucc_derived_of(coll_task,
                                                       ucc_schedule_t);
    ucc_base_team_t      *base_team       = schedule->super.team;
    ucc_tl_ucp_team_t    *team            = ucc_derived_of(base_team,
                                                       ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *tl_ctx          = UCC_TL_UCP_TEAM_CTX(team);
    ucc_rank_t            rank            = UCC_TL_TEAM_RANK(team);
    uint32_t              count_total     = coll_task->bargs.args.dst.info.count;
    ucc_rank_t            size            = UCC_TL_TEAM_SIZE(team);
    ucc_datatype_t        dtype           = coll_args->args.dst.info.datatype;
    size_t                dt_size         = ucc_dt_size(dtype);
    int                   inplace         = UCC_IS_INPLACE(coll_args->args);
    ucc_status_t          status          = UCC_OK;
    ucc_rank_t            put_window_size = UCC_TL_UCP_TEAM_LIB(team)
                                 ->cfg.allreduce_sliding_window_put_window_size;
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t
                         *gwbi_p          = coll_args->args.global_work_buffer;
    ucc_tl_ucp_task_t    *rdma_task       = ucc_derived_of(schedule->tasks[0],
                                                           ucc_tl_ucp_task_t);
    ucc_tl_ucp_allreduce_sw_pipeline_t       *pipe;
    ucc_tl_ucp_allreduce_sw_host_allgather_t *allgather_data;

    pipe           = rdma_task->allreduce_sliding_window.pipe;
    allgather_data = rdma_task->allreduce_sliding_window.allgather_data;

    // Register the src buf
    if (!inplace) {
        status = ucc_tl_ucp_allreduce_sliding_window_register(
            tl_ctx->worker.ucp_context, team,
            rdma_task->allreduce_sliding_window.bufs->src_ebuf,
            gwbi_p->packed_src_memh);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(rdma_task), "failed to register src memh: %s",
                        ucc_status_string(status));
            goto out;
        }
        ucc_assert(
            rdma_task->allreduce_sliding_window.bufs->src_ebuf->packed_key_len
            <= ALLREDUCE_PACKED_KEY_MAX_LEN);
        memcpy(allgather_data->packed_src_key,
               rdma_task->allreduce_sliding_window.bufs->src_ebuf->packed_key,
               rdma_task->allreduce_sliding_window.bufs->src_ebuf->packed_key_len);
    }

    // Register the dst buf
    status = ucc_tl_ucp_allreduce_sliding_window_register(
        tl_ctx->worker.ucp_context, team,
        rdma_task->allreduce_sliding_window.bufs->dst_ebuf,
        gwbi_p->packed_dst_memh);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(rdma_task), "failed to register dst memh: %s",
                    ucc_status_string(status));
        goto out;
    }
    ucc_assert(
        rdma_task->allreduce_sliding_window.bufs->dst_ebuf->packed_key_len
        <= ALLREDUCE_PACKED_KEY_MAX_LEN);
    memcpy(allgather_data->packed_dst_key,
           rdma_task->allreduce_sliding_window.bufs->dst_ebuf->packed_key,
           rdma_task->allreduce_sliding_window.bufs->dst_ebuf->packed_key_len);

    if (put_window_size == 0 || put_window_size > size) {
        put_window_size = size;
    }

    ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(
        pipe, rank, put_window_size);

    pipe->my_count = ucc_buffer_block_count(count_total, size, rank);
    pipe->my_offset = ucc_buffer_block_offset(count_total * dt_size, size, rank);

    rdma_task->allreduce_sliding_window.reduce_task = NULL;

    UCC_CHECK_GOTO(ucc_tl_ucp_allgather_ring_start(
                    rdma_task->allreduce_sliding_window.allgather_task),
        out, status);

    return ucc_schedule_start(coll_task);

out:
    tl_error(UCC_TASK_LIB(rdma_task), "failed to start allreduce sliding window: %s",
                                 ucc_status_string(status));
    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_finalize(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(coll_task);
    ucc_tl_ucp_put_schedule(schedule);

    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_rdma_task_post(
    ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t          *task  = ucc_derived_of(coll_task,
                                                       ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t          *team  = TASK_TEAM(task);
    
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

static inline void ucc_tl_ucp_allreduce_sliding_window_free_rkeys(
    ucc_coll_task_t *coll_task)
{
    ucc_base_team_t   *team      = coll_task->team;
    ucc_rank_t         team_size = (ucc_rank_t)team->params.size;
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    int                inplace   = UCC_IS_INPLACE(coll_task->bargs.args);
    ucc_rank_t         i;

    for (i = 0; i < team_size; i++) {
        if (!inplace && task->allreduce_sliding_window.bufs->src_rkeys[i] != NULL) {
            ucp_rkey_destroy(task->allreduce_sliding_window.bufs->src_rkeys[i]);
        }
        if (task->allreduce_sliding_window.bufs->dst_rkeys[i] != NULL) {
            ucp_rkey_destroy(task->allreduce_sliding_window.bufs->dst_rkeys[i]);
        }
    }
}

static ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_rdma_task_finalize(
    ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       st   = UCC_OK;

    ucc_tl_ucp_allreduce_sliding_window_free_rkeys(coll_task);
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
    ucc_ee_executor_t *exec;

    status = ucc_coll_task_get_executor(&task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to get executor");
    }

    status =
        ucc_dt_reduce(accbuf->buf, getbuf->buf, accbuf->buf, accbuf->count, dt,
                      args, 0, 0, exec,
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
        tl_error(UCC_TASK_LIB(task), "unable to complete UCX request=%p: %d",
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

static inline void ucc_tl_ucp_allreduce_sliding_window_mark_redbuf_free(
    ucc_tl_ucp_allreduce_sw_pipeline_t *pipe,
    ucc_tl_ucp_allreduce_sw_buf_t      *accbuf,
    ucc_tl_ucp_allreduce_sw_buf_t      *redbuf,
    ucc_rank_t                          host_team_size)
{
    redbuf->state = FREE;
    pipe->avail_buffs++;
    pipe->red_idx++;
    pipe->done_red++;

    if (pipe->done_red == host_team_size - 1) {
        accbuf->state = REDUCED;
        pipe->count_reduced += accbuf->count;
    }
}

void ucc_tl_ucp_allreduce_sliding_window_rdma_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t                  *task            =
        ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_rank_t                          size            =
        (ucc_rank_t)task->subset.map.ep_num;
    ucc_datatype_t                      dtype           =
        TASK_ARGS(task).dst.info.datatype;
    size_t                              dt_size         = ucc_dt_size(dtype);
    ucc_rank_t                          host_team_size  = size;
    ucc_base_team_t                    *base_team       = coll_task->team;
    ucc_tl_ucp_team_t                  *tl_team         =
        ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_allreduce_sw_pipeline_t *pipe            =
        task->allreduce_sliding_window.pipe;
    ucc_tl_ucp_context_t               *tl_ctx          =
        UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_tl_ucp_allreduce_sw_buf_t      *accbuf          = &pipe->accbuf;
    ucp_request_param_t                 req_param       = {0};
    int                                 i               = 0;
    ucc_coll_task_t                    *allgather_task  = 
        task->allreduce_sliding_window.allgather_task;
    ucc_ee_executor_task_t            **reduce_task     = 
        &task->allreduce_sliding_window.reduce_task;
    ucc_rank_t                          put_window_size =
        UCC_TL_UCP_TEAM_LIB(tl_team)->
            cfg.allreduce_sliding_window_put_window_size;
    ucc_tl_ucp_allreduce_sw_buf_t      *redbuf;
    ucc_tl_ucp_allreduce_sw_buf_t      *getbuf;
    size_t                              remaining_elems;
    ucc_rank_t                          get_idx;
    size_t                              count;
    size_t                              get_offset;
    size_t                              data_size;
    ucc_rank_t                          src_rank;
    ucc_rank_t                          dst_rank;
    void                               *src_addr;
    void                               *dst_addr;
    ucs_status_ptr_t                    request;
    size_t                              red_idx;
    size_t                              put_offset;
    int                                 window;
    int                                 put_idx;
    ucp_ep_h                            ep;
    ucc_status_t                        status;

    ucc_assert(host_team_size > 0);

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

        red_idx = pipe->red_idx % pipe->num_buffers;
        redbuf  = &pipe->getbuf[red_idx];

        ucc_tl_ucp_allreduce_sliding_window_mark_redbuf_free(
            pipe, accbuf, redbuf, host_team_size);
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
            src_addr = PTR_OFFSET(
                            task->allreduce_sliding_window.bufs->sbufs[src_rank],
                            get_offset);
            dst_addr = getbuf->buf;

            ucc_assert(getbuf->state == FREE);

            getbuf->state   = RECVING;
            getbuf->count   = count;
            getbuf->bytes   = data_size;
            ucc_tl_ucp_get_ep(tl_team, src_rank, &ep);
            getbuf->ucp_req = ucp_get_nbx(
                ep, dst_addr,
                data_size, (uint64_t)src_addr,
                task->allreduce_sliding_window.bufs->src_rkeys[src_rank],
                &req_param);

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
            status = ucc_tl_ucp_allreduce_sliding_window_req_test(request, task);
            if (status == UCC_OK) {
                if (request)
                    ucp_request_free(request);
                accbuf->state   = REDUCING;
                accbuf->ucp_req = NULL;
            } else if (status < 0) {
                tl_error(UCC_TL_TEAM_LIB(tl_team), "accbuf request failed: %s",
                    ucc_status_string(status));
            }
        }

        red_idx = pipe->red_idx % pipe->num_buffers;
        redbuf  = &pipe->getbuf[red_idx];
        if (accbuf->state == REDUCING && redbuf->state == RECVING) {
            request = redbuf->ucp_req;
            status = ucc_tl_ucp_allreduce_sliding_window_req_test(request, task);
            if (status == UCC_OK) {
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

                ucc_tl_ucp_allreduce_sliding_window_mark_redbuf_free(
                    pipe, accbuf, redbuf, host_team_size);

            } else if (status < 0) {
                tl_error(UCC_TL_TEAM_LIB(tl_team), "redbuf request failed: %s",
                    ucc_status_string(status));
            }
        }

        if ((pipe->count_serviced < pipe->count_reduced) &&
            (accbuf->state == REDUCED)) {
            data_size  = accbuf->bytes;
            put_offset = pipe->count_serviced * dt_size + pipe->my_offset;

            if (put_window_size == 0 || put_window_size > host_team_size) {
                put_window_size = host_team_size;
            }

            window = ucc_min(put_window_size,
                             host_team_size - pipe->posted_put);

            for (i = 0; i < window; i++) {
                dst_rank = pipe->dst_rank;
                src_addr = accbuf->buf;
                dst_addr = PTR_OFFSET(
                            task->allreduce_sliding_window.bufs->rbufs[dst_rank],
                            put_offset);
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
                        task->allreduce_sliding_window.bufs->dst_rkeys[dst_rank],
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

                pipe->count_serviced += accbuf->count;

                ucc_tl_ucp_allreduce_sliding_window_reset_buf(accbuf);
                pipe->done_get = 0;
                pipe->done_red = pipe->done_put = pipe->posted_put = 0;
            }
        }

        ucp_worker_progress(tl_ctx->worker.ucp_worker);
    }

    if (pipe->count_serviced == pipe->my_count) {
        task->super.status = UCC_OK;
    }
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t      *team,
                                         ucc_coll_task_t     **task_h)
{
    ucc_schedule_t          *schedule       = NULL;
    ucc_status_t             status         = UCC_OK;
    ucc_tl_ucp_team_t       *tl_team        =
        ucc_derived_of(team, ucc_tl_ucp_team_t);
    size_t                   allgather_size =
        sizeof(ucc_tl_ucp_allreduce_sw_host_allgather_t);
    ucc_rank_t               size           = UCC_TL_TEAM_SIZE(tl_team);
    ucc_base_coll_args_t bargs    = {
        .mask = 0,
        .args = {
            .coll_type = UCC_COLL_TYPE_ALLGATHER,
            .mask      = 0,
            .src.info = {.buffer   = NULL,
                         .count    = allgather_size,
                         .datatype = UCC_DT_UINT8,
                         .mem_type = UCC_MEMORY_TYPE_HOST},
            .dst.info = {.buffer   = NULL,
                         .count    = allgather_size * size,
                         .datatype = UCC_DT_UINT8,
                         .mem_type = UCC_MEMORY_TYPE_HOST}
        }
    };
    ucc_base_coll_args_t barrier_coll_args = {
        .team           = team->params.team,
        .args.coll_type = UCC_COLL_TYPE_BARRIER,
    };
    ucc_tl_ucp_allreduce_sw_host_allgather_t *allgather_data;
    ucc_tl_ucp_task_t                        *rdma_task;
    ucc_coll_task_t                          *barrier_task;

    if (!(coll_args->args.mask  & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER) ||
        !(coll_args->args.mask  & UCC_COLL_ARGS_FIELD_FLAGS)              ||
        !(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
        tl_debug(UCC_TL_TEAM_LIB(tl_team), "sliding window allreduce requires "
                                           "exported memh to be passed via "
                                           "global work buffer");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                    (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    *task_h                  = &schedule->super;
    schedule->super.post     = ucc_tl_ucp_allreduce_sliding_window_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_tl_ucp_allreduce_sliding_window_finalize;

    schedule->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;

    rdma_task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!rdma_task)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "Couldnt allocate task");
        return UCC_ERR_NO_MEMORY;
    }

    if (ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(team, rdma_task) != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "failed to alloc pipe: %s",
                                 ucc_status_string(status));
        goto free_rdma_task;
    }

    status = ucc_tl_ucp_allreduce_sliding_window_task_init(coll_args, team,
                                                           rdma_task);
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "failed to init task: %s",
                 ucc_status_string(status));
        goto out;
    }

    allgather_data = rdma_task->allreduce_sliding_window.allgather_data;
    bargs.args.src.info.buffer = allgather_data;
    bargs.args.dst.info.buffer = PTR_OFFSET(allgather_data, allgather_size);

    rdma_task->super.post     = ucc_tl_ucp_allreduce_sliding_window_rdma_task_post;
    rdma_task->super.progress = ucc_tl_ucp_allreduce_sliding_window_rdma_progress;
    rdma_task->super.finalize = ucc_tl_ucp_allreduce_sliding_window_rdma_task_finalize;

    UCC_CHECK_GOTO(ucc_tl_ucp_allgather_ring_init(&bargs, team,
                    &rdma_task->allreduce_sliding_window.allgather_task),
        free_rdma_pipe, status);

    status = ucc_tl_ucp_coll_init(&barrier_coll_args, team,
                                  &barrier_task);
    if (status < 0) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failure during sliding window barrier init: %s",
                 ucc_status_string(status));
        goto free_allgather_task;
    }

    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, &rdma_task->super), out, status);
    UCC_CHECK_GOTO(ucc_event_manager_subscribe(&schedule->super,
                                               UCC_EVENT_SCHEDULE_STARTED,
                                               &rdma_task->super,
                                               ucc_task_start_handler),
                   free_barrier_task, status);

    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, barrier_task), out, status);
    UCC_CHECK_GOTO(ucc_event_manager_subscribe(&rdma_task->super,
                                               UCC_EVENT_COMPLETED,
                                               barrier_task,
                                               ucc_task_start_handler),
                   free_barrier_task, status);

    return status;

free_barrier_task:
    ucc_tl_ucp_coll_finalize(barrier_task);
free_allgather_task:
    ucc_tl_ucp_coll_finalize(rdma_task->allreduce_sliding_window.allgather_task);
free_rdma_pipe:
    ucc_tl_ucp_allreduce_sliding_window_free_pipe(&rdma_task->super);
free_rdma_task:
    ucc_tl_ucp_allreduce_sliding_window_free_task(&rdma_task->super);
out:
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}
