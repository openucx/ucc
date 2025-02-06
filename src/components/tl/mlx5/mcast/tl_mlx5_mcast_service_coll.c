/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#include "mcast/tl_mlx5_mcast_service_coll.h"
 
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

    status =  ucc_service_allreduce(team, &ctx->tmp_buf, &ctx->tmp_buf, UCC_DT_INT8, 1,
                                    UCC_OP_SUM, subset, &req);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(ctx->base_ctx->lib, "tl service mcast barrier failed");
        return status;
    }

    *barrier_req = req;

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_service_coll_test(ucc_service_coll_req_t *req)
{
    ucc_status_t status = UCC_OK;

    status = ucc_service_coll_test(req);

    if (UCC_INPROGRESS != status) {
        if (status < 0) {
            ucc_error("oob service coll progress failed");
        }
        ucc_service_coll_finalize(req);
    }

    return status;
}
