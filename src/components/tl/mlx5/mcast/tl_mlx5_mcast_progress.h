/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#include "tl_mlx5_mcast.h"
#include "tl_mlx5_mcast_helper.h"


#ifndef RELIABLE_H_
#define RELIABLE_H_

#define TO_VIRTUAL(_rank, _size, _root) ((_rank + _size - _root) % _size)
#define TO_ORIGINAL(_rank, _size, _root) ((_rank + _root) % _size)

#define ACK 1
#define GET_COMPL_OBJ(_comm, _compl_fn, _pkt_id, _req)              \
    ({                                                              \
        void* item;                                                 \
        ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj;                \
        item = ucc_mpool_get(&(_comm)->ctx->compl_objects_mp);      \
        obj  = (ucc_tl_mlx5_mcast_p2p_completion_obj_t *)item;      \
                                                                    \
        obj->data[0]  = (uintptr_t)_comm;                           \
        obj->compl_cb = _compl_fn;                                  \
        obj->data[1]  = (uintptr_t)_pkt_id;                         \
        obj->data[2]  = (uintptr_t)_req;                            \
        obj;                                                        \
    })

#define GET_NACK_REQ(_comm, _pkt_id)                                \
    ({                                                              \
        void* item;                                                 \
        ucc_tl_mlx5_mcast_nack_req_t *_req;                         \
        item = ucc_mpool_get(&(_comm)->ctx->nack_reqs_mp);          \
                                                                    \
        _req         = (ucc_tl_mlx5_mcast_nack_req_t *)item;        \
        _req->comm   = _comm;                                       \
        _req->pkt_id = _pkt_id;                                     \
        _req;                                                       \
    })


#define P2P_OP(_comm, _buf, _size, _rank, _tag, _ctx, _op, _op_name, _compl_obj) do{                \
        ucc_tl_mlx5_mcast_p2p_interface_t *p2p = &(_comm)->p2p;                                     \
        int ret =  _op ## _nb((_buf), (_size), (_rank), (_tag), (_ctx), _compl_obj);                \
        if (UCC_OK != ret) {                                                                        \
            tl_error(_comm->lib, "FAILED to do p2p "_op_name": comm %p, size %zu, tag %d, rank %d", \
                      (_comm), (size_t)(_size), (_tag), (_rank) );                                  \
            return UCC_ERR_NO_MESSAGE;                                                              \
        }                                                                                           \
    }while(0)

#define SEND_NB(_comm, _buf, _size, _rank, _tag, _ctx, _compl_fn, _pkt_id, _req)                    \
                P2P_OP(_comm, _buf, _size, _rank, _tag, _ctx, p2p->send, "send",                    \
                GET_COMPL_OBJ(_comm, _compl_fn, _pkt_id, NULL))

#define SEND_NB_NO_COMPL(_comm, _buf, _size, _rank, _tag, _ctx)                                     \
                        P2P_OP(_comm, _buf, _size, _rank, _tag, _ctx, p2p->send, "send",            \
                        (&dummy_completion_obj))

#define RECV_NB(_comm, _buf, _size, _rank, _tag, _ctx, _compl_fn, _pkt_id, _req)                    \
                P2P_OP(_comm, _buf, _size, _rank, _tag, _ctx, p2p->recv, "recv",                    \
                GET_COMPL_OBJ(_comm, _compl_fn, _pkt_id, _req))

#define GET_RELIABLE_TAG(_comm) (_comm->last_acked % 1024)

int ucc_tl_mlx5_mcast_prepare_reliable(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                       ucc_tl_mlx5_mcast_coll_req_t *req,
                                       int root);

ucc_status_t ucc_tl_mlx5_mcast_bcast_check_drop(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                ucc_tl_mlx5_mcast_coll_req_t *req);

void ucc_tl_mlx5_mcast_process_packet(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                      ucc_tl_mlx5_mcast_coll_req_t *req,
                                      struct pp_packet* pp);

int ucc_tl_mlx5_mcast_check_nack_requests(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                          uint32_t psn);

int ucc_tl_mlx5_mcast_reliable_send(ucc_tl_mlx5_mcast_coll_comm_t* comm);

int ucc_tl_mlx5_mcast_check_nack_requests_all(ucc_tl_mlx5_mcast_coll_comm_t* comm);

#endif /* ifndef RELIABLE_H_ */

