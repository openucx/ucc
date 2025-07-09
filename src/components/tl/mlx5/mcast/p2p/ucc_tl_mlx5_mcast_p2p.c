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
                                                             ucc_memory_type_t mem_type,
                                                             ucc_team_h team, ucc_coll_callback_t *callback,
                                                             ucc_coll_req_h *p2p_req, int is_send,
                                                             ucc_base_lib_t *lib)
{
    ucc_status_t    status = UCC_OK;
    ucc_coll_req_h  req    = NULL;
    ucc_coll_args_t args   = {0};

    args.mask              = UCC_COLL_ARGS_FIELD_ACTIVE_SET |
                             UCC_COLL_ARGS_FIELD_CB;
    args.coll_type         = UCC_COLL_TYPE_BCAST;
    args.src.info.buffer   = buf;
    args.src.info.count    = len;
    args.src.info.datatype = UCC_DT_INT8;
    args.src.info.mem_type = mem_type;
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
                                      my_team_rank, ucc_rank_t dest,
                                      ucc_memory_type_t mem_type, ucc_team_h team,
                                      ucc_coll_callback_t *callback,
                                      ucc_coll_req_h *req, ucc_base_lib_t *lib)
{
    return ucc_tl_mlx5_mcast_do_p2p_bcast_nb(sbuf, len, my_team_rank, dest, mem_type,
                                             team, callback, req, 1, lib);
}

static inline ucc_status_t do_recv_nb(void *rbuf, size_t len, ucc_rank_t
                                      my_team_rank, ucc_rank_t dest,
                                      ucc_memory_type_t mem_type, ucc_team_h team,
                                      ucc_coll_callback_t *callback,
                                      ucc_coll_req_h *req, ucc_base_lib_t *lib)
{
    return ucc_tl_mlx5_mcast_do_p2p_bcast_nb(rbuf, len, my_team_rank, dest, mem_type,
                                             team, callback, req, 0, lib);
}

ucc_status_t ucc_tl_mlx5_mcast_p2p_send_nb(void* src, size_t size, ucc_rank_t
                                           rank, ucc_memory_type_t mem_type, void *context,
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
    status = do_send_nb(src, size, my_team_rank, rank, mem_type, team, &callback, &req, oob_p2p_ctx->lib);

    if (status < 0) {
        tl_error(oob_p2p_ctx->lib, "nonblocking p2p send failed");
        return status;
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_p2p_recv_nb(void *dst, size_t size, ucc_rank_t
                                           rank, ucc_memory_type_t mem_type, void *context,
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
    status = do_recv_nb(dst, size, my_team_rank, rank, mem_type, team, &callback, &req, oob_p2p_ctx->lib);

    if (status < 0) {
        tl_error(oob_p2p_ctx->lib, "nonblocking p2p recv failed");
        return status;
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_one_sided_p2p_put(void* src, void* remote_addr, size_t length,
                                           uint32_t lkey, uint32_t rkey, ucc_rank_t target_rank,
                                           uint64_t wr_id, ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct ibv_send_wr  swr = {0};
    struct ibv_sge      ssg = {0};
    struct ibv_send_wr *bad_wr;
    int                 rc;

    if (UINT32_MAX < length) {
        tl_error(comm->lib, "msg too large for p2p put");
        return UCC_ERR_NOT_SUPPORTED;
    }

    ssg.addr                = (uint64_t)src;
    ssg.length              = (uint32_t)length;
    ssg.lkey                = lkey;
    swr.sg_list             = &ssg;
    swr.num_sge             = 1;

    swr.opcode              = IBV_WR_RDMA_WRITE;
    swr.wr_id               = wr_id;
    swr.send_flags          = IBV_SEND_SIGNALED;
    swr.wr.rdma.remote_addr = (uint64_t)remote_addr;
    swr.wr.rdma.rkey        = rkey;
    swr.next                = NULL;

    tl_trace(comm->lib, "RDMA WRITE to rank %d size length %ld remote address %p rkey %d lkey %d src %p",
             target_rank, length, remote_addr, rkey, lkey, src);

    if (0 != (rc = ibv_post_send(comm->mcast.rc_qp[target_rank], &swr, &bad_wr))) {
        tl_error(comm->lib, "RDMA Write failed rc %d rank %d remote addresss %p rkey %d",
                 rc, target_rank, remote_addr, rkey);
        return UCC_ERR_NO_MESSAGE;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_one_sided_p2p_get(void* src, void* remote_addr, size_t length,
                                           uint32_t lkey, uint32_t rkey, ucc_rank_t target_rank,
                                           uint64_t wr_id, ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct ibv_send_wr  swr = {0};
    struct ibv_sge      ssg = {0};
    struct ibv_send_wr *bad_wr;
    int                 rc;

    if (UINT32_MAX < length) {
        tl_error(comm->lib, "msg too large for p2p get");
        return UCC_ERR_NOT_SUPPORTED;
    }

    ssg.addr                = (uint64_t)src;
    ssg.length              = (uint32_t)length;
    ssg.lkey                = lkey;
    swr.sg_list             = &ssg;
    swr.num_sge             = 1;

    swr.opcode              = IBV_WR_RDMA_READ;
    swr.wr_id               = wr_id;
    swr.send_flags          = IBV_SEND_SIGNALED;
    swr.wr.rdma.remote_addr = (uint64_t)remote_addr;
    swr.wr.rdma.rkey        = rkey;
    swr.next                = NULL;

    tl_trace(comm->lib, "RDMA READ to rank %d size length %ld remote address %p rkey %d lkey %d src %p",
             target_rank, length, remote_addr, rkey, lkey, src);

    if (0 != (rc = ibv_post_send(comm->mcast.rc_qp[target_rank], &swr, &bad_wr))) {
        tl_error(comm->lib, "RDMA Read failed rc %d rank %d remote addresss %p rkey %d",
                 rc, target_rank, remote_addr, rkey);
        return UCC_ERR_NO_MESSAGE;
    }

    return UCC_OK;
}
