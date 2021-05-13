/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_dpu.h"
#include "tl_dpu_coll.h"

#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

void ucc_tl_dpu_send_handler_nbx(void *request, ucs_status_t status, void *user_data)
{
    ucc_tl_dpu_request_t *req = (ucc_tl_dpu_request_t *)request;
    req->status = UCC_TL_DPU_UCP_REQUEST_DONE;
}

void ucc_tl_dpu_recv_handler_nbx(void *request, ucs_status_t status,
                      const ucp_tag_recv_info_t *tag_info,
                      void *user_data)
{
  ucc_tl_dpu_request_t *req = (ucc_tl_dpu_request_t *)request;
  req->status = UCC_TL_DPU_UCP_REQUEST_DONE;
}

static ucc_tl_dpu_task_t * ucc_tl_dpu_alloc_task(void)
{
    ucc_tl_dpu_task_t *task = (ucc_tl_dpu_task_t *) ucc_calloc(1, sizeof(ucc_tl_dpu_task_t), "Allocate task");
    return task;
}

static ucc_status_t ucc_tl_dpu_free_task(ucc_tl_dpu_task_t *task)
{
    ucc_free(task);
    return UCC_OK;
}

void ucc_tl_dpu_req_init(void* request)
{
    ucc_tl_dpu_request_t *req = (ucc_tl_dpu_request_t *)request;
    req->status = UCC_TL_DPU_UCP_REQUEST_ACTIVE;
}

void ucc_tl_dpu_req_cleanup(void* request){ 
    return;
}

ucc_status_t ucc_tl_dpu_req_test(ucc_tl_dpu_request_t **req, ucp_worker_h worker) {
    if (*req == NULL) {
        return UCC_OK;
    }

    if ((*req)->status == UCC_TL_DPU_UCP_REQUEST_DONE) {
        (*req)->status = UCC_TL_DPU_UCP_REQUEST_ACTIVE;
        ucp_request_free(*req);
        (*req) = NULL;
        return UCC_OK;
    }
    ucp_worker_progress(worker);
    return UCC_INPROGRESS;
}

inline
ucc_status_t ucc_tl_dpu_req_check(ucc_tl_dpu_team_t *team,
                                      ucc_tl_dpu_request_t *req) {
    if (UCS_PTR_IS_ERR(req)) {
        tl_error(team->super.super.context->lib,
                 "failed to send/recv msg");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}


ucc_status_t ucc_tl_dpu_allreduce_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_dpu_task_t       *task       = ucc_derived_of(coll_task, ucc_tl_dpu_task_t);
    ucc_tl_dpu_team_t       *team       = task->team;
    ucc_tl_dpu_context_t    *ctx        = UCC_TL_DPU_TEAM_CTX(team);
    volatile uint32_t       *check_flag = team->ctrl_seg;
    ucc_status_t            status      = UCC_INPROGRESS;
    int                     coll_poll   = UCC_TL_DPU_COLL_POLL;
    int                     i;
    ucp_request_param_t req_param;

    /* Are we still in start phase? */
    if (NULL != task->reqs[0] ||
        NULL != task->reqs[1]) {
        for (i = 0; i < coll_poll; i++) {
            ucp_worker_progress(ctx->ucp_worker);
            if ((ucc_tl_dpu_req_test(&(task->reqs[0]), ctx->ucp_worker) == UCC_OK) &&
                (ucc_tl_dpu_req_test(&(task->reqs[1]), ctx->ucp_worker) == UCC_OK)) {
                status = UCC_OK;
                break;
            }
        }
        if (UCC_INPROGRESS == status) {
            return UCC_INPROGRESS;
        }
    }

    /* check coll_id (return message from dpu server) */
    if (team->coll_id != (*check_flag + 1)) {
        return UCC_INPROGRESS;
    }

    if (NULL == task->reqs[2]) {
        req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                UCP_OP_ATTR_FIELD_DATATYPE;
        req_param.datatype     = ucp_dt_make_contig(1);
        req_param.cb.recv      = ucc_tl_dpu_recv_handler_nbx;

        task->reqs[2] = ucp_get_nbx(ctx->ucp_ep, task->args.dst.info.buffer,
                            task->args.src.info.count * ucc_dt_size(task->args.src.info.datatype),
                            team->rem_data_out, team->rem_data_out_key,
                            &req_param);
        if (ucc_tl_dpu_req_check(team, task->reqs[2]) != UCC_OK) {
            return UCC_ERR_NO_MESSAGE;
        }
    }
    
    for (i = 0; i < coll_poll; i++) {
        ucp_worker_progress(ctx->ucp_worker);
        if ((ucc_tl_dpu_req_test(&task->reqs[2], ctx->ucp_worker) == UCC_OK)) {
            task->super.super.status = UCC_OK;
            return task->super.super.status;
        }
    }

    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_dpu_allreduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_dpu_task_t           *task       = ucs_derived_of(coll_task, ucc_tl_dpu_task_t);
    ucc_tl_dpu_team_t           *team       = task->team;
    ucc_tl_dpu_context_t        *ctx        = UCC_TL_DPU_TEAM_CTX(team);
    void                        *sbuf       = task->args.src.info.buffer;
    void                        *rbuf       = task->args.dst.info.buffer;
    size_t                      count       = task->args.src.info.count;
    ucc_datatype_t              dt          = task->args.src.info.datatype;
    size_t                      data_size   = count * ucc_dt_size(dt);
    int                         i           = 0;
    int                         coll_poll   = UCC_TL_DPU_COLL_POLL;
    ucp_request_param_t         req_param;
    ucc_status_t                status;

    task->reqs[0] = NULL;
    task->reqs[1] = NULL;
    task->reqs[2] = NULL;

    tl_info(team->super.super.context->lib, "Collective post");

    req_param.op_attr_mask  = UCP_OP_ATTR_FIELD_CALLBACK |
                              UCP_OP_ATTR_FIELD_DATATYPE;
    req_param.datatype      = ucp_dt_make_contig(1);
    req_param.cb.send       = ucc_tl_dpu_send_handler_nbx;

    /* XXX set memory
    req_param.mask          = 0;
    req_param.mem_type      = task->args.src.info.mem_type;
    req_param.memory_type   = ucc_memtype_to_ucs[mtype];
    */

    if (UCC_IS_INPLACE(task->args)) {
        sbuf = rbuf;
    }
    task->reqs[0] = ucp_put_nbx(ctx->ucp_ep, sbuf, data_size,
                             team->rem_data_in, team->rem_data_in_key, &req_param);
    if (ucc_tl_dpu_req_check(team, task->reqs[0]) != UCC_OK) {
        return UCC_ERR_NO_MESSAGE;
    }
    ucp_worker_fence(ctx->ucp_worker);

    task->reqs[1] = ucp_put_nbx(ctx->ucp_ep, &task->sync, sizeof(task->sync),
                              team->rem_ctrl_seg, team->rem_ctrl_seg_key,
                              &req_param);
    if (ucc_tl_dpu_req_check(team, task->reqs[1]) != UCC_OK) {
        return UCC_ERR_NO_MESSAGE;
    }

    status = UCC_INPROGRESS;
    for (i = 0; i < coll_poll; i++) {
        ucp_worker_progress(ctx->ucp_worker);
        if ((ucc_tl_dpu_req_test(&(task->reqs[0]), ctx->ucp_worker) == UCC_OK) &&
            (ucc_tl_dpu_req_test(&(task->reqs[1]), ctx->ucp_worker) == UCC_OK)) {
            status = UCC_OK;
            break;
        }
    }

    task->super.super.status = UCC_INPROGRESS;
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_DPU_TEAM_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }

    status = ucc_tl_dpu_allreduce_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_DPU_TEAM_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_allreduce_init(ucc_tl_dpu_task_t *task)
{
    if (task->args.mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "userdefined reductions are not supported yet");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (!UCC_IS_INPLACE(task->args) && (task->args.src.info.mem_type !=
                                        task->args.dst.info.mem_type)) {
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "assymetric src/dst memory types are not supported yetpp");
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_dpu_allreduce_start;
    task->super.progress = ucc_tl_dpu_allreduce_progress;
    return UCC_OK;
}

static ucc_status_t ucc_tl_dpu_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_dpu_task_t *task = ucc_derived_of(coll_task, ucc_tl_dpu_task_t);
    tl_info(task->team->super.super.context->lib, "finalizing task %p", task);
    ucc_tl_dpu_free_task(task);
    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_coll_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t      *team,
                                         ucc_coll_task_t     **task_h)
{
    ucc_tl_dpu_team_t    *tl_team = ucc_derived_of(team, ucc_tl_dpu_team_t);
    ucc_tl_dpu_task_t    *task    = ucc_tl_dpu_alloc_task();
    ucc_status_t          status  = UCC_OK;

    ucc_coll_task_init(&task->super);
    tl_info(team->context->lib, "task %p initialized", task);

    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));

    task->sync.coll_id          = tl_team->coll_id;
    task->sync.dtype            = coll_args->args.src.info.datatype;
    task->sync.count_total      = coll_args->args.src.info.count;
    task->sync.count_in         = coll_args->args.src.info.count;
    task->sync.op               = coll_args->args.reduce.predefined_op;
    task->team                  = tl_team;
    task->super.finalize        = ucc_tl_dpu_coll_finalize;
    task->super.triggered_post  = NULL;
    tl_team->coll_id++;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        status = ucc_tl_dpu_allreduce_init(task);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (status != UCC_OK) {
        ucc_tl_dpu_free_task(task);
        return status;
    }

    tl_info(team->context->lib, "init coll req %p", task);
    *task_h = &task->super;
    return status;
}
