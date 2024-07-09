/**
 * Copyright(c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp_ep.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_dpu_offload.h"

void
ucc_tl_ucp_dpu_alltoall_linear_xgvmi_rdma_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t    *task           = ucc_derived_of(coll_task,
                                                          ucc_tl_ucp_task_t);
    ucc_datatype_t        dtype          = TASK_ARGS(task).src.info.datatype;
    size_t                dt_size        = ucc_dt_size(dtype);
    ucc_count_t           count          = coll_task->bargs.args.src.info.count;
    ucc_base_team_t      *base_team      = coll_task->team;
    ucc_tl_ucp_team_t    *tl_team        = ucc_derived_of(base_team,
                                                          ucc_tl_ucp_team_t);
    ucc_rank_t            host_team_size = UCC_TL_TEAM_SIZE(tl_team);
    ucc_coll_task_t      *allgather_task = task->dpu_xgvmi.allgather_task;
    ucc_tl_ucp_context_t *tl_ctx         = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucp_request_param_t   req_param      = {0};
    int                   i              = 0;
    ucc_rank_t            rank           = UCC_TL_TEAM_RANK(tl_team);
    size_t                data_size      = (count * dt_size) / host_team_size;
    ucs_status_ptr_t     *requests       = task->dpu_xgvmi.requests;
    int                  *posted         = &task->dpu_xgvmi.gets_posted;
    int                  *completed      = &task->dpu_xgvmi.gets_completed;
    void                 *src_addr;
    void                 *dst_addr;
    ucp_rkey_h            rkey;
    ucp_ep_h              ep;
    ucc_rank_t            offset;

    if (allgather_task != NULL) {
        ucc_tl_ucp_dpu_xgvmi_key_exchange_progress(coll_task);
        return;
    }

    req_param.op_attr_mask |= UCP_OP_ATTR_FIELD_MEMH;

    for (i = *posted; i < host_team_size; i++) {
        offset = (i + rank) % host_team_size;
        req_param.memh = task->dpu_xgvmi.bufs->dst_ebuf->memh;
        src_addr = PTR_OFFSET(task->dpu_xgvmi.bufs->sbufs[offset],
                              rank * data_size);
        dst_addr = PTR_OFFSET(task->dpu_xgvmi.bufs->rbufs[rank],
                              offset * data_size);
        rkey = task->dpu_xgvmi.bufs->src_rkeys[offset];
        ucc_tl_ucp_get_ep(tl_team, offset, &ep);

        requests[i] = ucp_get_nbx(
                ep, dst_addr,
                data_size, (uint64_t)src_addr,
                rkey, &req_param);

        *posted += 1;
    }

    ucp_worker_progress(tl_ctx->worker.ucp_worker);

    for (i = *completed; i < *posted; i++) {
        if (ucc_tl_ucp_dpu_xgvmi_req_test(requests[i], task) == UCC_OK) {
            if (requests[i]) ucp_request_free(requests[i]);
            *completed += 1;
        } else {
            break;
        }
    }

    if (*completed == host_team_size) {
        task->super.status = UCC_OK;
    }
}
