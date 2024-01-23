/**
 * Copyright(c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "../allgather/allgather.h"
#include "../barrier/barrier.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"

static inline void
ucc_tl_ucp_allreduce_sliding_window_reset_buf(ucc_tl_ucp_allreduce_sw_buf *buf)
{
    buf->state   = FREE;
    buf->count   = 0;
    buf->bytes   = 0;
    buf->ucp_req = NULL;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(
    ucc_tl_ucp_allreduce_sw_pipeline *pipe, ucc_rank_t rank,
    size_t put_window_size)
{
    int i;

    pipe->avail_buffs = pipe->num_buffers;
    pipe->src_rank = pipe->dst_rank = rank;
    pipe->get_idx = pipe->red_idx = 0;
    pipe->done_get = pipe->done_red = 0;
    pipe->done_put = pipe->posted_put = 0;
    pipe->count_issued = pipe->count_received = 0;
    pipe->count_reduced = pipe->count_serviced = 0;
    pipe->my_count = pipe->my_offset = 0;

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
    ucc_status_t       status = UCC_OK;
    ucc_tl_ucp_task_t *task   = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team   = TASK_TEAM(task);
    ucc_rank_t         rank   = UCC_TL_TEAM_RANK(team);
    uint32_t           count_total = coll_task->bargs.args.dst.info.count;
    ucc_rank_t         size        = coll_task->team->params.size;
    ucc_datatype_t     dtype       = TASK_ARGS(task).dst.info.datatype;
    size_t             dt_size     = ucc_dt_size(dtype);
    ucc_tl_ucp_allreduce_sw_pipeline *pipe =
        task->allreduce_sliding_window.pipe;
    ucc_tl_ucp_allreduce_sw_host_allgather *allgather_data =
        task->allreduce_sliding_window.allgather_data;
    size_t allgather_size = sizeof(ucc_tl_ucp_allreduce_sw_host_allgather);
    ucc_service_coll_req_t *scoll_req;

    ucc_tl_ucp_allreduce_sliding_window_reset_pipeline(
        pipe, rank, task->allreduce_sliding_window.put_window_size);

    pipe->my_count  = count_total / size;
    pipe->my_offset = pipe->my_count * dt_size * rank;
    if (rank == size - 1) {
        pipe->my_count += count_total % size;
    }

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    task->allreduce_sliding_window.reduce_in_progress = 0;
    task->allreduce_sliding_window.barrier_task       = NULL;

    UCC_CHECK_GOTO(
        ucc_service_allgather(
            UCC_TL_CORE_TEAM(team), allgather_data,
            PTR_OFFSET(allgather_data, allgather_size), allgather_size,
            ucc_sbgp_to_subset(ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL)),
            &scoll_req),
        out, status);

    scoll_req->data                                    = allgather_data;
    task->allreduce_sliding_window.allgather_scoll_req = scoll_req;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       st =
        ucc_ee_executor_finalize(task->allreduce_sliding_window.executor);

    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to finalize executor");
    }

    if (ucc_tl_ucp_allreduce_sliding_window_free_gwbi(coll_task) != UCC_OK) {
        printf("error freeing resources\n");
    }

    st = ucc_tl_ucp_coll_finalize(&task->super);

    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to finalize collective");
    }

    return st;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_reduction(
    ucc_coll_task_t *coll_task, ucc_tl_ucp_allreduce_sw_buf *accbuf,
    ucc_tl_ucp_allreduce_sw_buf *getbuf)
{
    ucc_status_t       status = UCC_OK;
    ucc_tl_ucp_task_t *task   = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t *  args   = &TASK_ARGS(task);
    ucc_datatype_t     dt     = TASK_ARGS(task).dst.info.datatype;

    status =
        ucc_dt_reduce(accbuf->buf, getbuf->buf, accbuf->buf, accbuf->count, dt,
                      args, 0, 0, task->allreduce_sliding_window.executor,
                      &task->allreduce_sliding_window.etask);

    task->allreduce_sliding_window.reduce_in_progress = 1;

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

#define SAVE_STATE(_reduce_in_progress) _reduce_in_progress = 1

    EXEC_TASK_TEST(task->allreduce_sliding_window.reduce_in_progress,
                   "failed to perform dt reduction",
                   task->allreduce_sliding_window.etask);

    // If it didn't complete, we would have returned by now. So, clear the flag
    task->allreduce_sliding_window.reduce_in_progress = 0;
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

static inline void ucc_tl_ucp_allreduce_sliding_window_allgather_info_test(
    ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *     task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_service_coll_req_t *allgather_scoll_req =
        task->allreduce_sliding_window.allgather_scoll_req;
    ucc_status_t status = ucc_service_coll_test(allgather_scoll_req);
    if (status < 0) {
        tl_error(coll_task->team->context->lib,
                 "failure during service coll exchange: %s",
                 ucc_status_string(status));
        ucc_service_coll_finalize(allgather_scoll_req);
        task->super.status = status;
        return;
    }
    if (UCC_INPROGRESS == status) {
        return;
    }
    ucc_assert(status == UCC_OK);

    // copy from allgather recvbuf into gwbi
    ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(
        allgather_scoll_req, task);

    ucc_service_coll_finalize(
        task->allreduce_sliding_window.allgather_scoll_req);
    task->allreduce_sliding_window.allgather_scoll_req = NULL;
}

static inline void ucc_tl_ucp_allreduce_sliding_window_allgather_free_rkeys(
    ucc_coll_task_t *coll_task)
{
    int                i;
    ucc_base_team_t *  team      = coll_task->team;
    ucc_rank_t         team_size = (ucc_rank_t)team->params.size;
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    for (i = 0; i < team_size; i++) {
        if (!task->allreduce_sliding_window.inplace)
            ucp_rkey_destroy(task->allreduce_sliding_window.src_rkeys[i]);
        ucp_rkey_destroy(task->allreduce_sliding_window.dst_rkeys[i]);
    }
}

static inline void
ucc_tl_ucp_allreduce_sliding_window_barrier(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_base_team_t *  team = coll_task->team;
    ucc_status_t       status;

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
    ucc_tl_ucp_task_t *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_rank_t         size    = (ucc_rank_t)task->subset.map.ep_num;
    ucc_datatype_t     dtype   = TASK_ARGS(task).dst.info.datatype;
    size_t             dt_size = ucc_dt_size(dtype);
    uint32_t           host_team_size = size;
    ucc_base_team_t *  base_team      = coll_task->team;
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_context_t *            tl_ctx = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_tl_ucp_allreduce_sw_pipeline *pipe =
        task->allreduce_sliding_window.pipe;
    ucc_tl_ucp_allreduce_sw_buf *accbuf    = &pipe->accbuf;
    ucp_request_param_t          req_param = {0};
    int                          i         = 0;
    ucc_service_coll_req_t *     allgather_scoll_req =
        task->allreduce_sliding_window.allgather_scoll_req;
    ucc_coll_task_t *barrier_task = task->allreduce_sliding_window.barrier_task;
    size_t           remaining_elems;
    size_t           get_idx;
    size_t           count;
    size_t           get_offset;
    size_t           data_size;
    ucc_rank_t       src_rank;
    ucc_rank_t       dst_rank;
    void *           src_addr;
    void *           dst_addr;
    ucs_status_ptr_t request;
    size_t           red_idx;
    ucc_tl_ucp_allreduce_sw_buf *redbuf;
    ucc_tl_ucp_allreduce_sw_buf *getbuf;
    size_t                       put_offset;
    int                          window;
    int                          put_idx;

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

    if (allgather_scoll_req != NULL) {
        ucc_tl_ucp_allreduce_sliding_window_allgather_info_test(coll_task);
        return;
    }

    if (task->allreduce_sliding_window.reduce_in_progress) {
        // We've previously started a reduction on the accbuf that hasn't yet
        // completed.
        ucc_tl_ucp_allreduce_sliding_window_test_reduction(task);

        if (task->allreduce_sliding_window.reduce_in_progress) {
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
            src_addr =
                task->allreduce_sliding_window.sbufs[src_rank] + get_offset;
            dst_addr = getbuf->buf;

            ucc_assert(getbuf->state == FREE);

            getbuf->state   = RECVING;
            getbuf->count   = count;
            getbuf->bytes   = data_size;
            getbuf->ucp_req = ucp_get_nbx(
                task->allreduce_sliding_window.eps[src_rank], dst_addr,
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

                if (task->allreduce_sliding_window.reduce_in_progress) {
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

            window = ucc_min(task->allreduce_sliding_window.put_window_size,
                             host_team_size - pipe->posted_put);

            for (i = 0; i < window; i++) {
                dst_rank = pipe->dst_rank;
                src_addr = accbuf->buf;
                dst_addr =
                    task->allreduce_sliding_window.rbufs[dst_rank] + put_offset;
                put_idx = pipe->posted_put %
                          task->allreduce_sliding_window.put_window_size;

                if (task->allreduce_sliding_window.put_requests[put_idx] !=
                    NULL) {
                    // We've already posted a put at this index that didn't yet
                    // complete, left this function and came back. Skip to check
                    // whether this request finished instead of overwriting it
                    // with another put
                    break;
                }

                ucp_worker_fence(tl_ctx->worker.ucp_worker);
                task->allreduce_sliding_window.put_requests[put_idx] =
                    ucp_put_nbx(
                        task->allreduce_sliding_window.eps[dst_rank], src_addr,
                        data_size, (uint64_t)dst_addr,
                        task->allreduce_sliding_window.dst_rkeys[dst_rank],
                        &req_param);

                pipe->posted_put++;
                pipe->dst_rank = (dst_rank + 1) % host_team_size;
            }

            for (i = pipe->done_put; i < pipe->posted_put; i++) {
                put_idx = i % task->allreduce_sliding_window.put_window_size;
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

                for (i = 0; i < task->allreduce_sliding_window.put_window_size;
                     i++) {
                    task->allreduce_sliding_window.put_requests[i] = NULL;
                }
            }
        }

        ucp_worker_progress(tl_ctx->worker.ucp_worker);
    }

    if (pipe->count_serviced == pipe->my_count) {
        ucc_tl_ucp_allreduce_sliding_window_barrier(coll_task);
        ucc_tl_ucp_allreduce_sliding_window_allgather_free_rkeys(coll_task);
    }
}
