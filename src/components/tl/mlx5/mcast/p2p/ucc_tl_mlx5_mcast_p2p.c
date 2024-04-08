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
                                                             ucc_team_h team, ucc_coll_callback_t *callback,
                                                             ucc_coll_req_h *p2p_req, int is_send,
                                                             ucc_base_lib_t *lib)
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
    args.active_set.stride = (int)dest - (int)my_team_rank;

    status = ucc_collective_init(&args, &req, team);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(lib, "nonblocking p2p init failed");
        return status;
    }

    ((ucc_tl_mlx5_mcast_p2p_completion_obj_t *)args.cb.data)->req = req;

    status = ucc_collective_post(req);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(lib, "nonblocking p2p post failed");
        return status;
    }

    *p2p_req = req;

    return status;
}

static inline ucc_status_t do_send_nb(void *sbuf, size_t len, ucc_rank_t
                                      my_team_rank, ucc_rank_t dest, ucc_team_h team,
                                      ucc_coll_callback_t *callback,
                                      ucc_coll_req_h *req, ucc_base_lib_t *lib)
{
    return ucc_tl_mlx5_mcast_do_p2p_bcast_nb(sbuf, len, my_team_rank, dest,
                                             team, callback, req, 1, lib);
}

static inline ucc_status_t do_recv_nb(void *rbuf, size_t len, ucc_rank_t
                                      my_team_rank, ucc_rank_t dest, ucc_team_h team,
                                      ucc_coll_callback_t *callback,
                                      ucc_coll_req_h *req, ucc_base_lib_t *lib)
{
    return ucc_tl_mlx5_mcast_do_p2p_bcast_nb(rbuf, len, my_team_rank, dest,
                                             team, callback, req, 0, lib);
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
    ucc_coll_callback_t                  callback;

    callback.cb   = ucc_tl_mlx5_mcast_completion_cb;
    callback.data = obj;

    tl_trace(oob_p2p_ctx->lib, "P2P: SEND to %d Msg Size %ld", rank, size);
    status = do_send_nb(src, size, my_team_rank, rank, team, &callback, &req, oob_p2p_ctx->lib);

    if (status < 0) {
        tl_error(oob_p2p_ctx->lib, "nonblocking p2p send failed");
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
    ucc_coll_callback_t                  callback;

    callback.cb   = ucc_tl_mlx5_mcast_completion_cb;
    callback.data = obj;

    tl_trace(oob_p2p_ctx->lib, "P2P: RECV to %d Msg Size %ld", rank, size);
    status = do_recv_nb(dst, size, my_team_rank, rank, team, &callback, &req, oob_p2p_ctx->lib);

    if (status < 0) {
        tl_error(oob_p2p_ctx->lib, "nonblocking p2p recv failed");
        return status;
    }

    return status;
}
