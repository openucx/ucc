/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_doca_urom.h"
#include "cl_doca_urom_coll.h"
#include "utils/ucc_coll_utils.h"

#include <doca_urom.h>
#include <urom_ucc.h>

static ucc_status_t ucc_cl_doca_urom_triggered_post_setup(ucc_coll_task_t *task)
{
    return UCC_OK;
}

static ucc_status_t ucc_cl_doca_urom_coll_full_start(ucc_coll_task_t *task)
{
    ucc_cl_doca_urom_team_t     *cl_team   = ucc_derived_of(task->team,
                                                ucc_cl_doca_urom_team_t);
    ucc_cl_doca_urom_context_t  *ctx       = UCC_CL_DOCA_UROM_TEAM_CTX(cl_team);
    ucc_cl_doca_urom_lib_t      *cl_lib    = ucc_derived_of(ctx->super.super.lib,
                                                ucc_cl_doca_urom_lib_t);
    ucc_coll_args_t             *coll_args = &task->bargs.args;
    int                          ucp_index = cl_lib->tl_ucp_index;
    ucc_tl_ucp_context_t        *tl_ctx    = ucc_derived_of(
                                                ctx->super.tl_ctxs[ucp_index],
                                                ucc_tl_ucp_context_t);
    union doca_data              cookie    = {0};
    int                          use_xgvmi = 0;
    int                          in_place  = 0;
    ucc_rank_t                   rank      = UCC_CL_TEAM_RANK(cl_team);
    ucc_cl_doca_urom_schedule_t *schedule  = ucc_derived_of(task,
                                                ucc_cl_doca_urom_schedule_t);
    struct export_buf           *src_ebuf  = &schedule->src_ebuf;
    struct export_buf           *dst_ebuf  = &schedule->dst_ebuf;
    doca_error_t                 result;
    ucc_worker_key_buf           keys;

    src_ebuf->memh = NULL;
    dst_ebuf->memh = NULL;

    cookie.ptr = &schedule->res;

    if ( (coll_args->mask & UCC_COLL_ARGS_FIELD_FLAGS  ) &&
         (coll_args->flags & UCC_COLL_ARGS_FLAG_IN_PLACE) ) {
        in_place = 1;
    }

    if (!in_place) {
        ucc_cl_doca_urom_buffer_export_ucc(
            tl_ctx->worker.ucp_context,
            coll_args->src.info.buffer,
            coll_args->src.info.count *
                ucc_dt_size(coll_args->src.info.datatype),
            src_ebuf);
    }

    ucc_cl_doca_urom_buffer_export_ucc(
        tl_ctx->worker.ucp_context,
        coll_args->dst.info.buffer,
        coll_args->dst.info.count *
            ucc_dt_size(coll_args->dst.info.datatype),
        dst_ebuf);

    switch (coll_args->coll_type) {
        case UCC_COLL_TYPE_ALLREDUCE:
        case UCC_COLL_TYPE_ALLTOALL:
        case UCC_COLL_TYPE_ALLGATHER:
        {
            if (!in_place) {
                keys.src_len = src_ebuf->packed_memh_len;
                memcpy(keys.rkeys, src_ebuf->packed_memh, keys.src_len);
            } else {
                keys.src_len = 0;
            }
            keys.dst_len = dst_ebuf->packed_memh_len;
            memcpy(keys.rkeys + keys.src_len,
                   dst_ebuf->packed_memh,
                   keys.dst_len);
            use_xgvmi = 1;
        } break;
        default:
            cl_error(&cl_lib->super, "coll_type %s is not supported",
                     ucc_coll_type_str(coll_args->coll_type));
    }

    coll_args->mask |= UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;

    result = ucc_cl_doca_urom_task_collective(ctx->urom_ctx.urom_worker,
                            cookie,
                            rank,
                            coll_args,
                            cl_team->teams[0],
                            use_xgvmi,
                            &keys,
                            sizeof(ucc_worker_key_buf),
                            0,
                            ucc_cl_doca_urom_collective_finished);
    if (result != DOCA_SUCCESS) {
        cl_error(&cl_lib->super, "Failed to create UCC collective task");
    }

    task->status = UCC_INPROGRESS;

    cl_debug(&cl_lib->super, "pushed the collective to urom");
    return ucc_progress_queue_enqueue(ctx->super.super.ucc_context->pq, task);
}

static ucc_status_t ucc_cl_doca_urom_coll_full_finalize(ucc_coll_task_t *task)
{
    ucc_cl_doca_urom_schedule_t *schedule  = ucc_derived_of(task,
                                                ucc_cl_doca_urom_schedule_t);
    ucc_cl_doca_urom_team_t     *cl_team   = ucc_derived_of(task->team,
                                                ucc_cl_doca_urom_team_t);
    ucc_cl_doca_urom_context_t  *ctx       = UCC_CL_DOCA_UROM_TEAM_CTX(cl_team);
    ucc_cl_doca_urom_lib_t      *cl_lib    = ucc_derived_of(ctx->super.super.lib,
                                                ucc_cl_doca_urom_lib_t);
    int                          ucp_index = cl_lib->tl_ucp_index;
    ucc_tl_ucp_context_t        *tl_ctx    = ucc_derived_of(
                                                ctx->super.tl_ctxs[ucp_index],
                                                ucc_tl_ucp_context_t);
    struct export_buf           *src_ebuf  = &schedule->src_ebuf;
    struct export_buf           *dst_ebuf  = &schedule->dst_ebuf;
    ucc_status_t                 status;

    if (src_ebuf->memh) {
        ucp_mem_unmap(tl_ctx->worker.ucp_context, src_ebuf->memh);
    }
    ucp_mem_unmap(tl_ctx->worker.ucp_context, dst_ebuf->memh);

    status = ucc_schedule_finalize(task);
    ucc_cl_doca_urom_put_schedule(&schedule->super.super);

    return status;
}

static void ucc_cl_doca_urom_coll_full_progress(ucc_coll_task_t *ctask)
{
    ucc_cl_doca_urom_team_t        *cl_team   = ucc_derived_of(ctask->team,
                                                ucc_cl_doca_urom_team_t);
    ucc_cl_doca_urom_context_t     *ctx       = UCC_CL_DOCA_UROM_TEAM_CTX(cl_team);
    ucc_cl_doca_urom_lib_t         *cl_lib    = ucc_derived_of(
                                                ctx->super.super.lib,
                                                ucc_cl_doca_urom_lib_t);
    ucc_cl_doca_urom_schedule_t    *schedule  = ucc_derived_of(ctask,
                                                ucc_cl_doca_urom_schedule_t);
    int                             ucp_index = cl_lib->tl_ucp_index;
    ucc_tl_ucp_context_t           *tl_ctx    = ucc_derived_of(
                                                   ctx->super.tl_ctxs[ucp_index],
                                                   ucc_tl_ucp_context_t);
    struct ucc_cl_doca_urom_result *res       = &schedule->res;
    int                             ret;

    if (res == NULL) {
        cl_error(cl_lib, "Error in UROM");
        ctask->status = UCC_ERR_NO_MESSAGE;
        return;
    }

    ucp_worker_progress(tl_ctx->worker.ucp_worker);

    ret = doca_pe_progress(ctx->urom_ctx.urom_pe);
    if (ret == 0 && res->result == DOCA_SUCCESS) {
        ctask->status = UCC_INPROGRESS;
        return;
    }

    if (res->result != DOCA_SUCCESS) {
        cl_error(&cl_lib->super, "Error in DOCA_UROM, UCC collective task failed");
    }

    ctask->status = res->collective.status;
    cl_debug(&cl_lib->super, "completed the collective from urom");
}  

ucc_status_t ucc_cl_doca_urom_coll_full_init(
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_doca_urom_team_t     *cl_team = ucc_derived_of(team,
                                            ucc_cl_doca_urom_team_t);
    ucc_cl_doca_urom_context_t  *ctx     = UCC_CL_DOCA_UROM_TEAM_CTX(cl_team);
    ucc_cl_doca_urom_lib_t      *cl_lib  = ucc_derived_of(ctx->super.super.lib,
                                            ucc_cl_doca_urom_lib_t);
    ucc_status_t                 status;
    ucc_cl_doca_urom_schedule_t *cl_schedule;
    ucc_base_coll_args_t         args;
    ucc_schedule_t              *schedule;

    cl_schedule = ucc_cl_doca_urom_get_schedule(cl_team);
    if (ucc_unlikely(!cl_schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    schedule = &cl_schedule->super.super;
    memcpy(&args, coll_args, sizeof(args));
    status = ucc_schedule_init(schedule, &args, team);
    if (UCC_OK != status) {
        ucc_cl_doca_urom_put_schedule(schedule);
        return status;
    }

    schedule->super.post                 = ucc_cl_doca_urom_coll_full_start;
    schedule->super.progress             = ucc_cl_doca_urom_coll_full_progress;
    schedule->super.finalize             = ucc_cl_doca_urom_coll_full_finalize;
    schedule->super.triggered_post       = ucc_triggered_post;
    schedule->super.triggered_post_setup = ucc_cl_doca_urom_triggered_post_setup;

    *task = &schedule->super;
    cl_debug(cl_lib, "cl doca urom coll initialized");
    return UCC_OK;
}

ucc_status_t ucc_cl_doca_urom_coll_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t      *team,
                                        ucc_coll_task_t     **task)
{
    ucc_cl_doca_urom_team_t    *cl_team       = ucc_derived_of(team,
                                                    ucc_cl_doca_urom_team_t);
    ucc_cl_doca_urom_context_t *ctx           = UCC_CL_DOCA_UROM_TEAM_CTX(cl_team);
    ucc_cl_doca_urom_lib_t     *doca_urom_lib = ucc_derived_of(
                                                    ctx->super.super.lib,
                                                    ucc_cl_doca_urom_lib_t);

    switch (coll_args->args.coll_type) {
        case UCC_COLL_TYPE_ALLREDUCE:
        case UCC_COLL_TYPE_ALLGATHER:
        case UCC_COLL_TYPE_ALLTOALL:
            return ucc_cl_doca_urom_coll_full_init(coll_args, team, task);
        default:
            cl_error(doca_urom_lib, "coll_type %s is not supported",
                ucc_coll_type_str(coll_args->args.coll_type));
    }

    return UCC_OK;
}
