/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp_dpu_offload.h"
#include "allgather/allgather.h"
#include "alltoall/alltoall.h"
#include "barrier/barrier.h"
#include "tl_ucp_ep.h"
#include "tl_ucp_dpu_offload.h"

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_register(
    ucp_context_h ucp_context, ucc_tl_ucp_team_t *tl_team,
    struct ucc_tl_ucp_allreduce_sw_export_buf *ebuf, void *packed_memh)
{
    ucp_mem_map_params_t params = {0};
    ucs_status_t         ucs_status, unmap_status;

    ebuf->ucp_context = ucp_context;

    params.field_mask           = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
    params.exported_memh_buffer = packed_memh;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    if (UCS_OK != ucs_status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "import using ucp_mem_map() returned error: %s",
                 ucs_status_string(ucs_status));
        return ucs_status_to_ucc_status(ucs_status);
    }

    ucs_status = ucp_rkey_pack(ucp_context, ebuf->memh, &ebuf->packed_key,
                               &ebuf->packed_key_len);
    if (UCS_OK != ucs_status) {
        unmap_status = ucp_mem_unmap(ucp_context, ebuf->memh);
        tl_error(UCC_TL_TEAM_LIB(tl_team),
            "ucp_rkey_pack() returned error: %s%s",
            ucs_status_string(ucs_status),
            unmap_status == UCS_OK ? "" : 
            ". While handling this error, unmapping the memh had an error");
        return ucs_status_to_ucc_status(ucs_status);
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_dpu_xgvmi_task_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t      *team,
                                            ucc_tl_ucp_task_t    *task)
{
    void                   *src_buf        = coll_args->args.src.info.buffer;
    void                   *dst_buf        = coll_args->args.dst.info.buffer;
    ucc_tl_ucp_team_t      *tl_team        = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t              team_size      = UCC_TL_TEAM_SIZE(tl_team);
    int                     inplace        = UCC_IS_INPLACE(coll_args->args);
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t
                           *gwbi_p         = NULL;
    size_t                  allgather_size =
        sizeof(ucc_tl_ucp_allreduce_sw_host_allgather_t);
    ucc_tl_ucp_allreduce_sw_host_allgather_t
                           *allgather_data;
    ucc_rank_t              i;
    void                   *buffer;
    void                   *ptr;
    size_t                  bufs_sz, allgather_data_sz, rbufs_sz, dst_rkeys_sz,
                            dst_ebuf_sz, sbufs_sz, src_rkeys_sz, src_ebuf_sz;

    ucc_assert(team_size > 0);

    bufs_sz           = sizeof(ucc_tl_ucp_dpu_offload_buf_info_t);
    allgather_data_sz = allgather_size * (team_size + 1);
    rbufs_sz          = sizeof(void *) * team_size;
    dst_rkeys_sz      = sizeof(ucp_rkey_h) * team_size;
    dst_ebuf_sz       = sizeof(struct ucc_tl_ucp_allreduce_sw_export_buf);

    if (!inplace) {
        sbufs_sz      = sizeof(void *) * team_size;
        src_rkeys_sz  = sizeof(ucp_rkey_h) * team_size;
        src_ebuf_sz   = sizeof(struct ucc_tl_ucp_allreduce_sw_export_buf);
    } else {
        sbufs_sz      = 0;
        src_rkeys_sz  = 0;
        src_ebuf_sz   = 0;
    }

    buffer = ucc_malloc(bufs_sz + allgather_data_sz + rbufs_sz +
                        dst_rkeys_sz + dst_ebuf_sz + sbufs_sz +
                        src_rkeys_sz + src_ebuf_sz);
    if (buffer == NULL) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "error while allocating task");
        return UCC_ERR_NO_RESOURCE;
    }

    ptr = buffer;

    task->dpu_xgvmi.bufs = ptr;

    ptr = allgather_data = PTR_OFFSET(ptr, bufs_sz);
    task->dpu_xgvmi.allgather_data = allgather_data;

    gwbi_p = coll_args->args.global_work_buffer;
    task->super.bargs.args.global_work_buffer = gwbi_p;

    ptr = task->dpu_xgvmi.bufs->rbufs = PTR_OFFSET(ptr, allgather_data_sz);
    ptr = task->dpu_xgvmi.bufs->dst_rkeys = PTR_OFFSET(ptr, rbufs_sz);
    for (i = 0; i < team_size; i++) {
        task->dpu_xgvmi.bufs->dst_rkeys[i] = NULL;
    }

    ptr = task->dpu_xgvmi.bufs->dst_ebuf = PTR_OFFSET(ptr, dst_rkeys_sz);
    task->dpu_xgvmi.bufs->dst_ebuf->memh = NULL;

    allgather_data->dst_buf = dst_buf;

    task->dpu_xgvmi.allgather_data = allgather_data;
    task->dpu_xgvmi.allgather_task = NULL;

    if (!inplace) {
        allgather_data->src_buf = src_buf;

        ptr = task->dpu_xgvmi.bufs->sbufs = PTR_OFFSET(ptr, dst_ebuf_sz);
        ptr = task->dpu_xgvmi.bufs->src_rkeys = PTR_OFFSET(ptr, sbufs_sz);
        for (i = 0; i < team_size; i++) {
            task->dpu_xgvmi.bufs->src_rkeys[i] = NULL;
        }

        task->dpu_xgvmi.bufs->src_ebuf = PTR_OFFSET(ptr, src_rkeys_sz);
        task->dpu_xgvmi.bufs->src_ebuf->memh = NULL;
    } else {
        task->dpu_xgvmi.bufs->src_ebuf = NULL;
    }

    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_dpu_xgvmi_allgather_info_finalize(ucc_tl_ucp_task_t *sw_task)
{
    ucs_status_t       ucs_status = UCS_OK;
    ucc_base_team_t   *base_team  = sw_task->super.team;
    ucc_tl_ucp_team_t *tl_team    = ucc_derived_of(base_team, ucc_tl_ucp_team_t);
    ucc_rank_t         team_size  = UCC_TL_TEAM_SIZE(tl_team);
    void              *recvbuf    = sw_task->dpu_xgvmi.
                                    allgather_task->bargs.args.dst.info.buffer;
    ucc_tl_ucp_allreduce_sw_host_allgather_t *all_host_allgather = recvbuf;
    ucc_status_t       status     = UCC_OK;
    int                inplace    = UCC_IS_INPLACE(TASK_ARGS(sw_task));
    ucc_rank_t         i;
    ucp_ep_h           ep;
    ucp_rkey_h         src_unpacked, dst_unpacked;

    ucc_assert(team_size > 0);

    for (i = 0; i < team_size; i++) {
        status = ucc_tl_ucp_get_ep(tl_team, i, &ep);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }

        ucs_status = ucp_ep_rkey_unpack(
            ep, all_host_allgather[i].packed_dst_key, &dst_unpacked);
        if (UCS_OK != ucs_status) {
            tl_error(UCC_TL_TEAM_LIB(tl_team), "dst rkey unpack failed");
            return ucs_status_to_ucc_status(ucs_status);
        }

        sw_task->dpu_xgvmi.bufs->rbufs[i] =
            all_host_allgather[i].dst_buf;
        sw_task->dpu_xgvmi.bufs->dst_rkeys[i] = dst_unpacked;

        if (!inplace) {
            ucs_status = ucp_ep_rkey_unpack(
                ep, all_host_allgather[i].packed_src_key, &src_unpacked);
            if (UCS_OK != ucs_status) {
                tl_error(UCC_TL_TEAM_LIB(tl_team), "src rkey unpack failed");
                return ucs_status_to_ucc_status(ucs_status);
            }

            sw_task->dpu_xgvmi.bufs->sbufs[i] =
                all_host_allgather[i].src_buf;
            sw_task->dpu_xgvmi.bufs->src_rkeys[i] = src_unpacked;
        } else {
            sw_task->dpu_xgvmi.bufs->sbufs =
                sw_task->dpu_xgvmi.bufs->rbufs;
            sw_task->dpu_xgvmi.bufs->src_rkeys =
                sw_task->dpu_xgvmi.bufs->dst_rkeys;
        }
    }

    return status;
}

void
ucc_tl_ucp_dpu_xgvmi_free_task(ucc_coll_task_t *coll_task)
{
    ucc_base_team_t      *team    = coll_task->team;
    ucc_tl_ucp_team_t    *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t    *task    = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    int                   inplace = UCC_IS_INPLACE(coll_task->bargs.args);
    ucc_tl_ucp_context_t *tl_ctx  = UCC_TL_UCP_TEAM_CTX(tl_team);

    if (task->dpu_xgvmi.bufs) {
        if (!inplace) {
            if (task->dpu_xgvmi.bufs->src_ebuf->memh != NULL) {
                ucp_mem_unmap(tl_ctx->worker.ucp_context,
                              task->dpu_xgvmi.bufs->src_ebuf->memh);
                task->dpu_xgvmi.bufs->src_ebuf->memh = NULL;
            }
        }

        if (task->dpu_xgvmi.bufs->dst_ebuf->memh != NULL) {
            ucp_mem_unmap(tl_ctx->worker.ucp_context,
                          task->dpu_xgvmi.bufs->dst_ebuf->memh);
        }
        ucc_free(task->dpu_xgvmi.bufs);
    }
}

ucc_status_t
ucc_tl_ucp_dpu_xgvmi_start(ucc_coll_task_t *coll_task)
{
    ucc_base_coll_args_t *coll_args       = &coll_task->bargs;
    ucc_schedule_t       *schedule        = ucc_derived_of(coll_task,
                                                       ucc_schedule_t);
    ucc_base_team_t      *base_team       = schedule->super.team;
    ucc_tl_ucp_team_t    *team            = ucc_derived_of(base_team,
                                                       ucc_tl_ucp_team_t);
    ucc_rank_t            team_size       = UCC_TL_TEAM_SIZE(team);
    ucc_tl_ucp_context_t *tl_ctx          = UCC_TL_UCP_TEAM_CTX(team);
    int                   inplace         = UCC_IS_INPLACE(coll_args->args);
    ucc_status_t          status          = UCC_OK;
    ucc_tl_ucp_allreduce_sw_global_work_buf_info_t
                         *gwbi_p          = coll_args->args.global_work_buffer;
    ucc_tl_ucp_task_t    *rdma_task       = ucc_derived_of(schedule->tasks[0],
                                                           ucc_tl_ucp_task_t);
    ucc_tl_ucp_allreduce_sw_host_allgather_t *allgather_data;

    allgather_data = rdma_task->dpu_xgvmi.allgather_data;

    rdma_task->dpu_xgvmi.gets_posted = 0;
    rdma_task->dpu_xgvmi.gets_completed = 0;
    memset(rdma_task->dpu_xgvmi.requests, 0,
           team_size * sizeof(sizeof(ucs_status_ptr_t)));

    // Register the src buf
    if (!inplace) {
        status = ucc_tl_ucp_allreduce_sliding_window_register(
            tl_ctx->worker.ucp_context, team,
            rdma_task->dpu_xgvmi.bufs->src_ebuf,
            gwbi_p->packed_src_memh);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(rdma_task), "failed to register src memh: %s",
                        ucc_status_string(status));
            goto out;
        }
        ucc_assert(
            rdma_task->dpu_xgvmi.bufs->src_ebuf->packed_key_len
            <= ALLREDUCE_PACKED_KEY_MAX_LEN);
        memcpy(allgather_data->packed_src_key,
               rdma_task->dpu_xgvmi.bufs->src_ebuf->packed_key,
               rdma_task->dpu_xgvmi.bufs->src_ebuf->packed_key_len);
    }

    // Register the dst buf
    status = ucc_tl_ucp_allreduce_sliding_window_register(
        tl_ctx->worker.ucp_context, team,
        rdma_task->dpu_xgvmi.bufs->dst_ebuf,
        gwbi_p->packed_dst_memh);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(rdma_task), "failed to register dst memh: %s",
                    ucc_status_string(status));
        goto out;
    }
    ucc_assert(
        rdma_task->dpu_xgvmi.bufs->dst_ebuf->packed_key_len
        <= ALLREDUCE_PACKED_KEY_MAX_LEN);
    memcpy(allgather_data->packed_dst_key,
           rdma_task->dpu_xgvmi.bufs->dst_ebuf->packed_key,
           rdma_task->dpu_xgvmi.bufs->dst_ebuf->packed_key_len);

    UCC_CHECK_GOTO(ucc_tl_ucp_allgather_ring_start(
                    rdma_task->dpu_xgvmi.allgather_task),
        out, status);

    return ucc_schedule_start(coll_task);

out:
    tl_error(UCC_TASK_LIB(rdma_task), "failed to start allgather sliding window: %s",
                                 ucc_status_string(status));
    return status;
}

ucc_status_t
ucc_tl_ucp_dpu_xgvmi_finalize(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(coll_task);
    ucc_tl_ucp_put_schedule(schedule);

    return status;
}

ucc_status_t ucc_tl_ucp_dpu_xgvmi_rdma_task_post(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task,
                                                       ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

void ucc_tl_ucp_dpu_xgvmi_free_rkeys(ucc_coll_task_t *coll_task)
{
    ucc_base_team_t   *team      = coll_task->team;
    ucc_rank_t         team_size = (ucc_rank_t)team->params.size;
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    int                inplace   = UCC_IS_INPLACE(coll_task->bargs.args);
    ucc_rank_t         i;

    for (i = 0; i < team_size; i++) {
        if (!inplace && task->dpu_xgvmi.bufs->src_rkeys[i] != NULL) {
            ucp_rkey_destroy(task->dpu_xgvmi.bufs->src_rkeys[i]);
        }
        if (task->dpu_xgvmi.bufs->dst_rkeys[i] != NULL) {
            ucp_rkey_destroy(task->dpu_xgvmi.bufs->dst_rkeys[i]);
        }
    }
}

ucc_status_t ucc_tl_ucp_dpu_xgvmi_rdma_task_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       st   = UCC_OK;

    ucc_tl_ucp_dpu_xgvmi_free_rkeys(coll_task);
    ucc_tl_ucp_dpu_xgvmi_free_task(coll_task);

    st = ucc_tl_ucp_coll_finalize(coll_task);

    if (ucc_unlikely(st != UCC_OK)) {
        tl_error(UCC_TASK_LIB(task), "failed to finalize collective");
    }

    return st;
}

ucc_status_t ucc_tl_ucp_dpu_xgvmi_req_test(ucs_status_ptr_t   request,
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

void ucc_tl_ucp_dpu_xgvmi_key_exchange_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_task_t   *allgather_task =
                           task->dpu_xgvmi.allgather_task;
    ucc_status_t       status    = allgather_task->super.status;

    if (status < 0) {
        goto err;
    }
    if (UCC_INPROGRESS == status) {
        ucc_tl_ucp_allgather_ring_progress(allgather_task);
        return;
    }
    ucc_assert(status == UCC_OK);

    // copy from allgather recvbuf into xgvmi task
    UCC_CHECK_GOTO(
        ucc_tl_ucp_dpu_xgvmi_allgather_info_finalize(task),
        err, status);

out:
    ucc_tl_ucp_coll_finalize(allgather_task);
    task->dpu_xgvmi.allgather_task = NULL;
    return;
err:
    ucc_tl_ucp_dpu_xgvmi_free_task(coll_task);
    tl_error(coll_task->team->context->lib,
                "key exchange failure: %s",
                ucc_status_string(status));
    goto out;
}

ucc_status_t ucc_tl_ucp_dpu_xgvmi_init(ucc_base_coll_args_t *coll_args,
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

    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                    (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    *task_h                  = &schedule->super;
    schedule->super.post     = ucc_tl_ucp_dpu_xgvmi_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_tl_ucp_dpu_xgvmi_finalize;

    schedule->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;

    rdma_task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!rdma_task)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "Couldnt allocate task");
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_tl_ucp_dpu_xgvmi_task_init(coll_args, team,
                                                           rdma_task);
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "failed to init task: %s",
                 ucc_status_string(status));
        goto out;
    }

    allgather_data = rdma_task->dpu_xgvmi.allgather_data;
    bargs.args.src.info.buffer = allgather_data;
    bargs.args.dst.info.buffer = PTR_OFFSET(allgather_data, allgather_size);

    rdma_task->super.post     = ucc_tl_ucp_dpu_xgvmi_rdma_task_post;
    rdma_task->super.finalize = ucc_tl_ucp_dpu_xgvmi_rdma_task_finalize;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLTOALL:
        rdma_task->super.progress = ucc_tl_ucp_dpu_alltoall_linear_xgvmi_rdma_progress;
        break;
    case UCC_COLL_TYPE_ALLGATHER:
        rdma_task->super.progress = ucc_tl_ucp_dpu_allgather_linear_xgvmi_rdma_progress;
        break;
    default:
        tl_error(UCC_TL_TEAM_LIB(tl_team), "coll_type %s is not supported",
                 ucc_coll_type_str(coll_args->args.coll_type));
        break;
    }

    rdma_task->dpu_xgvmi.requests = ucc_malloc(sizeof(ucs_status_ptr_t) * size);

    UCC_CHECK_GOTO(ucc_tl_ucp_allgather_ring_init(&bargs, team,
                    &rdma_task->dpu_xgvmi.allgather_task),
        free_rdma_task, status);

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
    ucc_tl_ucp_coll_finalize(rdma_task->dpu_xgvmi.allgather_task);
free_rdma_task:
    ucc_tl_ucp_dpu_xgvmi_free_task(&rdma_task->super);
out:
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}
