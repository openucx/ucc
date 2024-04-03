/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast.h"
#include "tl_mlx5_mcast_helper.h"

#ifndef TL_MLX5_MCAST_PROGRESS_H_
#define TL_MLX5_MCAST_PROGRESS_H_

#define TO_VIRTUAL(_rank, _size, _root) ((_rank + _size - _root) % _size)

#define TO_ORIGINAL(_rank, _size, _root) ((_rank + _root) % _size)

#define ACK 1

#define GET_COMPL_OBJ(_comm, _compl_fn, _pkt_id, _req)                          \
    ({                                                                          \
        void* item;                                                             \
        ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj;                            \
        item = ucc_mpool_get(&(_comm)->ctx->compl_objects_mp);                  \
        obj  = (ucc_tl_mlx5_mcast_p2p_completion_obj_t *)item;                  \
                                                                                \
        obj->data[0]  = (uintptr_t)_comm;                                       \
        obj->compl_cb = _compl_fn;                                              \
        obj->data[1]  = (uintptr_t)_pkt_id;                                     \
        obj->data[2]  = (uintptr_t)_req;                                        \
        obj;                                                                    \
    })

#define GET_NACK_REQ(_comm, _pkt_id)                                            \
    ({                                                                          \
        void* item;                                                             \
        ucc_tl_mlx5_mcast_nack_req_t *_req;                                     \
        item = ucc_mpool_get(&(_comm)->ctx->nack_reqs_mp);                      \
                                                                                \
        _req         = (ucc_tl_mlx5_mcast_nack_req_t *)item;                    \
        _req->comm   = _comm;                                                   \
        _req->pkt_id = _pkt_id;                                                 \
        _req;                                                                   \
    })

ucc_status_t ucc_tl_mlx5_mcast_prepare_reliable(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                ucc_tl_mlx5_mcast_coll_req_t *req,
                                                ucc_rank_t root);

ucc_status_t ucc_tl_mlx5_mcast_bcast_check_drop(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                ucc_tl_mlx5_mcast_coll_req_t *req);

ucc_status_t ucc_tl_mlx5_mcast_process_packet(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                              ucc_tl_mlx5_mcast_coll_req_t *req,
                                              struct pp_packet* pp);

ucc_status_t ucc_tl_mlx5_mcast_check_nack_requests(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                   uint32_t psn);

ucc_status_t ucc_tl_mlx5_mcast_reliable_send(ucc_tl_mlx5_mcast_coll_comm_t* comm);

ucc_status_t ucc_tl_mlx5_mcast_check_nack_requests(ucc_tl_mlx5_mcast_coll_comm_t* comm, uint32_t psn);

#endif /* ifndef TL_MLX5_MCAST_PROGRESS_H_ */

