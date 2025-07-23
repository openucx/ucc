/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "mcast/tl_mlx5_mcast_service_coll.h"
#include "core/ucc_service_coll.h"
 
ucc_status_t ucc_tl_mlx5_mcast_service_bcast_post(void *arg, void *buf, size_t size, ucc_rank_t root,
                                                  ucc_service_coll_req_t **bcast_req)
{
    ucc_tl_mlx5_mcast_oob_p2p_context_t *ctx    = (ucc_tl_mlx5_mcast_oob_p2p_context_t *)arg;
    ucc_status_t                         status = UCC_OK;
    ucc_team_t                          *team   = ctx->base_team;
    ucc_subset_t                         subset = ctx->subset;
    ucc_service_coll_req_t              *req    = NULL;

    status = ucc_service_bcast(team, buf, size, root, subset, &req);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(ctx->base_ctx->lib, "tl service mcast bcast failed");
        return status;
    }

    *bcast_req = req;

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_service_allgather_post(void *arg, void *sbuf, void *rbuf, size_t size,
                                                      ucc_service_coll_req_t **ag_req)
{
    ucc_tl_mlx5_mcast_oob_p2p_context_t *ctx    = (ucc_tl_mlx5_mcast_oob_p2p_context_t *)arg;
    ucc_status_t                         status = UCC_OK;
    ucc_team_t                          *team   = ctx->base_team;
    ucc_subset_t                         subset = ctx->subset;
    ucc_service_coll_req_t              *req    = NULL;

    status =  ucc_service_allgather(team, sbuf, rbuf, size, subset, &req);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(ctx->base_ctx->lib, "tl service mcast allgather failed");
        return status;
    }

    *ag_req = req;

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_service_barrier_post(void *arg, ucc_service_coll_req_t **barrier_req)
{
    ucc_tl_mlx5_mcast_oob_p2p_context_t *ctx    = (ucc_tl_mlx5_mcast_oob_p2p_context_t *)arg;
    ucc_status_t                         status = UCC_OK;
    ucc_team_t                          *team   = ctx->base_team;
    ucc_subset_t                         subset = ctx->subset;
    ucc_service_coll_req_t              *req    = NULL;
    ctx->tmp_sbuf                               = 0;
    ctx->tmp_rbuf                               = 0;

    status =  ucc_service_allreduce(team, &ctx->tmp_sbuf, &ctx->tmp_rbuf, UCC_DT_INT8, 1,
                                    UCC_OP_SUM, subset, &req);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(ctx->base_ctx->lib, "tl service mcast barrier failed");
        return status;
    }

    *barrier_req = req;

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_service_allreduce_post(void *arg, void *sendbuf, void *recvbuf,
                                                       size_t count, ucc_datatype_t dt,
                                                       ucc_reduction_op_t op,
                                                       ucc_service_coll_req_t **allred_req)
{
    ucc_tl_mlx5_mcast_oob_p2p_context_t *ctx    = (ucc_tl_mlx5_mcast_oob_p2p_context_t *)arg;
    ucc_status_t                         status = UCC_OK;
    ucc_team_t                          *team   = ctx->base_team;
    ucc_subset_t                         subset = ctx->subset;
    ucc_service_coll_req_t              *req    = NULL;

    status = ucc_service_allreduce(team, sendbuf, recvbuf, dt, count, op, subset, &req);
    if (UCC_OK != status) {
        tl_error(ctx->base_ctx->lib, "tl service mcast allreduce failed");
        return status;
    }
    *allred_req = req;
    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_service_coll_test(ucc_service_coll_req_t *req)
{
    ucc_status_t status = UCC_OK;

    status = ucc_collective_test(&req->task->super);
    if (UCC_INPROGRESS != status) {
        if (status < 0) {
            ucc_error("oob service coll progress failed");
        }
        ucc_service_coll_finalize(req);
    }

    return status;
}
