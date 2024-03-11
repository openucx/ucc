/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_tl_mlx5_mcast_p2p.h"

static inline void ucc_tl_mlx5_mcast_p2p_completion_cb(void* context)
{
    ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj =
        (ucc_tl_mlx5_mcast_p2p_completion_obj_t *)context;

    ucc_assert(obj != NULL && obj->compl_cb != NULL);

    obj->compl_cb(obj);

    ucc_assert(obj->req != NULL);

    ucc_collective_finalize(obj->req);
}

void ucc_tl_mlx5_mcast_completion_cb(void* context, ucc_status_t status) //NOLINT
{
    ucc_tl_mlx5_mcast_p2p_completion_cb(context);
}

static inline ucc_status_t ucc_tl_mlx5_mcast_do_p2p_bcast_nb(void *buf, size_t
                                                             len, ucc_rank_t my_team_rank, ucc_rank_t dest,
                                                             ucc_team_h team, ucc_context_h ctx,
                                                             ucc_coll_callback_t *callback,
                                                             ucc_coll_req_h *p2p_req, int is_send)
{
    ucc_status_t    status = UCC_OK;
    ucc_coll_req_h  req    = NULL;
    ucc_coll_args_t args;

    args.mask              = UCC_COLL_ARGS_FIELD_ACTIVE_SET |
                             UCC_COLL_ARGS_FIELD_CB;
    args.coll_type         = UCC_COLL_TYPE_BCAST;
    args.src.info.buffer   = buf;
    args.src.info.count    = len;
    args.src.info.datatype = UCC_DT_INT8;
    args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    args.root              = is_send ? my_team_rank : dest;
    args.cb.cb             = callback->cb;
    args.cb.data           = callback->data;
    args.active_set.size   = 2;
    args.active_set.start  = my_team_rank;
    args.active_set.stride = dest - my_team_rank;

    status = ucc_collective_init(&args, &req, team);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(ctx->lib, "nonblocking p2p init failed");
        return status;
    }

    ((ucc_tl_mlx5_mcast_p2p_completion_obj_t *)args.cb.data)->req = req;

    status = ucc_collective_post(req);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(ctx->lib, "nonblocking p2p post failed");
        return status;
    }

    *p2p_req = req;

    return status;
}

static inline ucc_status_t do_send_nb(void *sbuf, size_t len, ucc_rank_t
                                      my_team_rank, ucc_rank_t dest, ucc_team_h team,
                                      ucc_context_h ctx, ucc_coll_callback_t
                                      *callback, ucc_coll_req_h *req)
{
    return ucc_tl_mlx5_mcast_do_p2p_bcast_nb(sbuf, len, my_team_rank, dest,
                                             team, ctx, callback, req, 1);
}

static inline ucc_status_t do_recv_nb(void *rbuf, size_t len, ucc_rank_t
                                      my_team_rank, ucc_rank_t dest, ucc_team_h team,
                                      ucc_context_h ctx, ucc_coll_callback_t
                                      *callback, ucc_coll_req_h *req)
{
    return ucc_tl_mlx5_mcast_do_p2p_bcast_nb(rbuf, len, my_team_rank, dest,
                                             team, ctx, callback, req, 0);
}

ucc_status_t ucc_tl_mlx5_mcast_p2p_send_nb(void* src, size_t size, ucc_rank_t
                                           rank, void *context,
                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t
                                           *obj)
{
    ucc_tl_mlx5_mcast_oob_p2p_context_t *oob_p2p_ctx  =
                                   (ucc_tl_mlx5_mcast_oob_p2p_context_t *)context;
    ucc_status_t                         status       = UCC_OK;
    ucc_coll_req_h                       req          = NULL;
    ucc_rank_t                           my_team_rank = oob_p2p_ctx->my_team_rank;
    ucc_team_h                           team         = oob_p2p_ctx->base_team;
    ucc_context_h                        ctx          = oob_p2p_ctx->base_ctx;
    ucc_coll_callback_t                  callback;

    callback.cb   = ucc_tl_mlx5_mcast_completion_cb;
    callback.data = obj;

    status = do_send_nb(src, size, my_team_rank, rank, team, ctx, &callback, &req);

    if (status < 0) {
        tl_error(ctx->lib, "nonblocking p2p send failed");
        return status;
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_p2p_recv_nb(void *dst, size_t size, ucc_rank_t
                                           rank, void *context,
                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t
                                           *obj)
{
    ucc_tl_mlx5_mcast_oob_p2p_context_t *oob_p2p_ctx  =
                                   (ucc_tl_mlx5_mcast_oob_p2p_context_t *)context;
    ucc_status_t                         status       = UCC_OK;
    ucc_coll_req_h                       req          = NULL;
    ucc_rank_t                           my_team_rank = oob_p2p_ctx->my_team_rank;
    ucc_team_h                           team         = oob_p2p_ctx->base_team;
    ucc_context_h                        ctx          = oob_p2p_ctx->base_ctx;
    ucc_coll_callback_t                  callback;

    callback.cb   = ucc_tl_mlx5_mcast_completion_cb;
    callback.data = obj;

    status = do_recv_nb(dst, size, my_team_rank, rank, team, ctx, &callback, &req);

    if (status < 0) {
        tl_error(ctx->lib, "nonblocking p2p recv failed");
        return status;
    }

    return status;
}
