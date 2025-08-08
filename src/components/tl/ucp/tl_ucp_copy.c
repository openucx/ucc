/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_tag.h"
#include "tl_ucp_ep.h"
#include "tl_ucp_sendrecv.h"
#include "tl_ucp_copy.h"

ucc_status_t ucc_tl_ucp_mc_copy_post(void *dst, ucc_memory_type_t dst_mtype,
                                     ucp_mem_h dst_memh, //NOLINT
                                     void *src, ucc_memory_type_t src_mtype,
                                     ucp_mem_h src_memh, //NOLINT
                                     size_t size,
                                     ucc_tl_ucp_task_t *coll_task, //NOLINT
                                     ucc_tl_ucp_copy_task_t **copy_task) //NOLINT
{
    return ucc_mc_memcpy(dst, src, size, dst_mtype, src_mtype);
}

ucc_status_t ucc_tl_ucp_mc_copy_test(ucc_tl_ucp_context_t *ctx, //NOLINT
                                     ucc_tl_ucp_copy_task_t *copy_task) //NOLINT
{
    /* mc copy is blocking, test always returns UCC_OK */
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_mc_copy_finalize(ucc_tl_ucp_copy_task_t *copy_task) //NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_ec_copy_post(void *dst, ucc_memory_type_t dst_mtype, //NOLINT
                                     ucp_mem_h dst_memh, //NOLINT
                                     void *src, ucc_memory_type_t src_mtype, //NOLINT
                                     ucp_mem_h src_memh, //NOLINT
                                     size_t size,
                                     ucc_tl_ucp_task_t *coll_task,
                                     ucc_tl_ucp_copy_task_t **copy_task)
{
    ucc_ee_executor_task_args_t eargs = {0};
    ucc_ee_executor_task_t     **eee_task = (ucc_ee_executor_task_t **)copy_task;
    ucc_status_t status;
    ucc_ee_executor_t *exec;

    status = ucc_coll_task_get_executor(&coll_task->super, &exec);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }

    eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    eargs.copy.dst  = dst;
    eargs.copy.src  = src;
    eargs.copy.len  = size;

    return ucc_ee_executor_task_post(exec, &eargs, eee_task);
}

ucc_status_t ucc_tl_ucp_ec_copy_test(ucc_tl_ucp_context_t *ctx, //NOLINT
                                     ucc_tl_ucp_copy_task_t *copy_task)
{
    ucc_ee_executor_task_t *eee_task = (ucc_ee_executor_task_t *)copy_task;

    return ucc_ee_executor_task_test(eee_task);
}

ucc_status_t ucc_tl_ucp_ec_copy_finalize(ucc_tl_ucp_copy_task_t *copy_task)
{
    ucc_ee_executor_task_t *eee_task = (ucc_ee_executor_task_t *)copy_task;

    return ucc_ee_executor_task_finalize(eee_task);
}

void ucc_tl_ucp_copy_send_completion_cb(void *request, ucs_status_t status,
                                        void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in copy send completion %s",
                 ucs_status_string(status));
    }
    ucp_request_free(request);
}

void ucc_tl_ucp_copy_recv_completion_cb(void *request, ucs_status_t status, //NOLINT
                                        const ucp_tag_recv_info_t *info, /* NOLINT */
                                        void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in copy recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    /* no request free, it will be called by user */
}

ucc_status_t ucc_tl_ucp_ucp_copy_post(void *dst, ucc_memory_type_t dst_mtype,
                                      ucp_mem_h dst_memh,
                                      void *src, ucc_memory_type_t src_mtype,
                                      ucp_mem_h src_memh,
                                      size_t size, ucc_tl_ucp_task_t *task,
                                      ucc_tl_ucp_copy_task_t **copy_task)
{
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         trank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         dest_group_rank = trank;
    ucp_request_param_t req_param;
    ucc_status_t        status;
    ucp_ep_h            ep;
    ucp_tag_t           ucp_tag, ucp_tag_mask;
    ucs_status_ptr_t    ucp_status;

    UCC_TL_UCP_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask,
                                (args->mask & UCC_COLL_ARGS_FIELD_TAG),
                                task->tagged.tag, trank,
                                team->super.super.params.id,
                                team->super.super.params.scope_id,
                                team->super.super.params.scope);
    req_param.op_attr_mask =
        UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
        UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    req_param.datatype    = ucp_dt_make_contig(size);
    req_param.memory_type = ucc_memtype_to_ucs[dst_mtype];
    req_param.cb.recv     = ucc_tl_ucp_copy_recv_completion_cb;
    req_param.user_data   = task;
    if (dst_memh) {
        req_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
        req_param.memh         = dst_memh;
    }
    ucp_status = ucp_tag_recv_nbx(team->worker->ucp_worker, dst, 1, ucp_tag,
                                  ucp_tag_mask, &req_param);
    UCC_TL_UCP_CHECK_REQ_STATUS();
    (*copy_task)= ucp_status;

    status = ucc_tl_ucp_get_ep(team, trank, &ep);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    ucp_tag = UCC_TL_UCP_MAKE_SEND_TAG((args->mask & UCC_COLL_ARGS_FIELD_TAG),
        task->tagged.tag, trank, team->super.super.params.id,
        team->super.super.params.scope_id, team->super.super.params.scope);
    req_param.op_attr_mask =
        UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
        UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    req_param.datatype    = ucp_dt_make_contig(size);
    req_param.memory_type = ucc_memtype_to_ucs[src_mtype];
    req_param.cb.send     = ucc_tl_ucp_copy_send_completion_cb;
    req_param.user_data   = task;
    if (src_memh) {
        req_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;
        req_param.memh         = src_memh;
    }
    ucp_status = ucp_tag_send_nbx(ep, src, 1, ucp_tag, &req_param);
    UCC_TL_UCP_CHECK_REQ_STATUS();

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_ucp_copy_test(ucc_tl_ucp_context_t *ctx,
                                      ucc_tl_ucp_copy_task_t *copy_task)
{
    ucs_status_ptr_t req_status = (ucs_status_ptr_t)copy_task;

    ucp_worker_progress(ctx->worker.ucp_worker);
    return ucs_status_to_ucc_status(ucp_request_check_status(req_status));
}

ucc_status_t ucc_tl_ucp_ucp_copy_finalize(ucc_tl_ucp_copy_task_t *copy_task)
{
    ucs_status_ptr_t req_status = (ucs_status_ptr_t)copy_task;
    ucp_request_free(req_status);
    return UCC_OK;
}
