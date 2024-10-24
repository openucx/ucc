/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "tl_mlx5_mcast_helper.h"
#include "tl_mlx5_mcast_rcache.h"

static inline ucc_status_t ucc_tl_mlx5_mcast_r_window_recycle(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                              ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    ucc_status_t     status        = UCC_OK;
    int              wsize         = comm->wsize;
    int              num_free_win  = wsize - (comm->psn - comm->last_acked);
    int              req_completed = (req->to_send == 0 && req->to_recv == 0);
    struct pp_packet *pp           = NULL;

    ucc_assert(comm->recv_drop_packet_in_progress == false);
    ucc_assert(req->to_send >= 0);

    /* When do we need to perform reliability protocol:
       1. Always in the end of the window
       2. For the zcopy case: in the end of collective, because we can't signal completion
       before made sure that children received the data - user can modify buffer */

    ucc_assert(num_free_win >= 0);

    if (!num_free_win || (req->proto == MCAST_PROTO_ZCOPY && req_completed)) {
        status = ucc_tl_mlx5_mcast_reliable(comm);
        if (UCC_OK != status) {
            return status;
        }

        comm->n_mcast_reliable++;

        for (;comm->last_acked < comm->psn; comm->last_acked++) {
            pp = comm->r_window[comm->last_acked & (wsize-1)];
            ucc_assert(pp != &comm->dummy_packet);
            comm->r_window[comm->last_acked & (wsize-1)] = &comm->dummy_packet;

            pp->context = 0;
            ucc_list_add_tail(&comm->bpool, &pp->super);
        }

        if (!req_completed) {
            status = ucc_tl_mlx5_mcast_prepare_reliable(comm, req, req->root);
            if (ucc_unlikely(UCC_OK != status)) {
                return status;
            }
        }
    }

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_do_bcast(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_status_t                   status = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm   = req->comm;
    int                            zcopy  = req->proto != MCAST_PROTO_EAGER;
    int                            wsize  = comm->wsize;
    int                            num_free_win;
    int                            num_sent;
    int                            to_send;
    int                            to_recv;
    int                            to_recv_left;
    int                            pending_q_size;


    status = ucc_tl_mlx5_mcast_check_nack_requests(comm, UINT32_MAX);
    if (status < 0) {
        return status;
    }

    if (ucc_unlikely(comm->recv_drop_packet_in_progress)) {
        /* wait till parent resend the dropped packet */
        return UCC_INPROGRESS;
    }


    if (req->to_send || req->to_recv) {
        num_free_win = wsize - (comm->psn - comm->last_acked);

        /* Send data if i'm root and there is a space in the window */
        if (num_free_win && req->am_root) {
            num_sent = req->num_packets - req->to_send;
			ucc_assert(req->to_send > 0);
			ucc_assert(req->first_send_psn + num_sent < comm->last_acked + wsize);
            if (req->first_send_psn + num_sent < comm->last_acked + wsize &&
                req->to_send) {
                /* How many to send: either all that are left (if they fit into window) or
                   up to the window limit */
                to_send = ucc_min(req->to_send,
                                  comm->last_acked + wsize - (req->first_send_psn + num_sent));
                ucc_tl_mlx5_mcast_send(comm, req, to_send, zcopy);

                num_free_win = wsize - (comm->psn - comm->last_acked);
            }
        }
        
        status = ucc_tl_mlx5_mcast_prepare_reliable(comm, req, req->root);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }

        if (num_free_win && req->to_recv) {
            /* How many to recv: either all that are left or up to the window limit. */
            pending_q_size = 0;
            to_recv        = ucc_min(num_free_win, req->to_recv);
            to_recv_left   = ucc_tl_mlx5_mcast_recv(comm, req, to_recv, &pending_q_size);

            if (to_recv == to_recv_left) {
                /* We didn't receive anything: increase the stalled counter and get ready for
                   drop event */
                if (comm->stalled++ >= DROP_THRESHOLD) {

                    tl_trace(comm->lib, "Did not receive the packet with psn in"
                            " current window range, so get ready for drop"
                            " event. pending_q_size %d current comm psn %d"
                            " last_acked psn %d stall threshold %d ",
                            pending_q_size, comm->psn, comm->last_acked,
                            DROP_THRESHOLD);

                    status = ucc_tl_mlx5_mcast_bcast_check_drop(comm, req);
                    if (UCC_INPROGRESS == status) {
                        return status;
                    }
                }
            } else if (to_recv_left < 0) {
                /* a failure happend during cq polling */
                return UCC_ERR_NO_MESSAGE;
            } else {
                comm->stalled = 0;
                comm->timer   = 0;
            }
        }
    }

    /* This function will check if we have to do a round of reliability protocol */
    status = ucc_tl_mlx5_mcast_r_window_recycle(comm, req);
    if (UCC_OK != status) {
        return status;
    }

    if (req->to_send || req->to_recv || (zcopy && comm->psn != comm->last_acked)) {
        return UCC_INPROGRESS;
    } else {
        return status;
    }
}


ucc_status_t ucc_tl_mlx5_mcast_test(ucc_tl_mlx5_mcast_coll_req_t* req)
{
    ucc_status_t status = UCC_OK;

    ucc_assert(req->comm->psn >= req->start_psn);

    status = ucc_tl_mlx5_mcast_do_bcast(req);
    if (UCC_INPROGRESS != status) {
        ucc_assert(req->comm->ctx != NULL);
        ucc_tl_mlx5_mcast_mem_deregister(req->comm->ctx, req->rreg);
        req->rreg = NULL;
    }

    return status;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_prepare_bcast(void* buf, size_t size, ucc_rank_t root,
                                                           ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                           ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    ucc_status_t             status;
    ucc_tl_mlx5_mcast_reg_t *reg;

    req->comm    = comm;
    req->ptr     = buf;
    req->length  = size;
    req->root    = root;
    req->am_root = (root == comm->rank);
    req->mr      = comm->pp_mr;
    req->rreg    = NULL;
    /* cost of copy is too high in cuda buffers */
    req->proto   = (req->length < comm->max_eager && !comm->cuda_mem_enabled) ?
                            MCAST_PROTO_EAGER : MCAST_PROTO_ZCOPY;

    status = ucc_tl_mlx5_mcast_prepare_reliable(comm, req, req->root);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    if (req->am_root) {
        if (req->proto != MCAST_PROTO_EAGER) {
           status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->ptr, req->length, &reg);
           if (UCC_OK != status) {
                return status;
           }
           req->rreg = reg;
           req->mr   = reg->mr;
        }
    }

    req->offset       = 0;
    req->start_psn    = comm->last_psn;
    req->num_packets  = ucc_max(ucc_div_round_up(req->length, comm->max_per_packet), 1);
    req->last_pkt_len = req->length - (req->num_packets - 1)*comm->max_per_packet;

    ucc_assert(req->last_pkt_len > 0 && req->last_pkt_len <= comm->max_per_packet);

    comm->last_psn     += req->num_packets;
    req->first_send_psn = req->start_psn;
    req->to_send        = req->am_root ? req->num_packets : 0;
    req->to_recv        = req->am_root ? 0 : req->num_packets;

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_coll_do_bcast(void* buf, size_t size, ucc_rank_t root,
                                             ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                             ucc_tl_mlx5_mcast_coll_req_t **task_req_handle)
{
    ucc_status_t                  status;
    ucc_tl_mlx5_mcast_coll_req_t *req;

    tl_trace(comm->lib, "MCAST bcast start, buf %p, size %ld, root %d, comm %d, "
             "comm_size %d, am_i_root %d comm->psn = %d \n",
             buf, size, root, comm->comm_id, comm->commsize, comm->rank ==
             root, comm->psn );

    req = ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_coll_req_t), "mcast_req");
    if (!req) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_tl_mlx5_mcast_prepare_bcast(buf, size, root, comm, req);
    if (UCC_OK != status) {
        ucc_free(req);
        return status;
    }

    status           = UCC_INPROGRESS;
    *task_req_handle = req;

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t            *task      = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_team_t            *mlx5_team = TASK_TEAM(task);
    ucc_tl_mlx5_mcast_team_t      *team      = mlx5_team->mcast;
    ucc_coll_args_t               *args      = &TASK_ARGS(task);
    ucc_datatype_t                 dt        = args->src.info.datatype;
    size_t                         count     = args->src.info.count;
    ucc_rank_t                     root      = args->root;
    ucc_status_t                   status    = UCC_OK;
    size_t                         data_size = ucc_dt_size(dt) * count;
    void                          *buf       = args->src.info.buffer;
    ucc_tl_mlx5_mcast_coll_comm_t *comm      = team->mcast_comm;

    task->bcast_mcast.req_handle = NULL;

    status = ucc_tl_mlx5_mcast_coll_do_bcast(buf, data_size, root, comm,
                                             &task->bcast_mcast.req_handle);
    if (status < 0) {
        tl_error(UCC_TASK_LIB(task), "mcast_coll_do_bcast failed:%d", status);
        coll_task->status = status;
        return ucc_task_complete(coll_task);
    }

    coll_task->status = status;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(mlx5_team)->pq, &task->super);
}

void ucc_tl_mlx5_mcast_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t           *task   = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_mcast_coll_req_t *req    = task->bcast_mcast.req_handle;

    if (req != NULL) {
        coll_task->status = ucc_tl_mlx5_mcast_test(req);
    }
}

static inline ucc_status_t ucc_tl_mlx5_mcast_check_memory_type_cap(ucc_base_coll_args_t *coll_args,
                                                                   ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *mlx5_team = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_mcast_coll_comm_t *comm      = mlx5_team->mcast->mcast_comm;
    ucc_coll_args_t               *args      = &coll_args->args;

    if ((comm->cuda_mem_enabled &&
            args->src.info.mem_type == UCC_MEMORY_TYPE_CUDA) ||
        (!comm->cuda_mem_enabled &&
                args->src.info.mem_type == UCC_MEMORY_TYPE_HOST)) {
        return UCC_OK;
    }
        
    return UCC_ERR_NO_RESOURCE;
}

ucc_status_t ucc_tl_mlx5_mcast_check_support(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *team)
{
    ucc_coll_args_t *args     = &coll_args->args;
    int              buf_size = ucc_dt_size(args->src.info.datatype) * args->src.info.count;

    if (UCC_COLL_ARGS_ACTIVE_SET(args)) {
        tl_trace(team->context->lib, "mcast bcast not supported for active sets");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_OK != ucc_tl_mlx5_mcast_check_memory_type_cap(coll_args, team)) {
        tl_trace(team->context->lib, "mcast bcast not compatible with this memory type");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (args->src.info.mem_type == UCC_MEMORY_TYPE_CUDA &&
            buf_size > CUDA_MEM_MCAST_BCAST_MAX_MSG) {
        /* for large messages (more than one mtu) we need zero-copy design which
         * is not implemented yet */
        tl_trace(team->context->lib, "mcast cuda bcast not supported for large messages");
        return UCC_ERR_NOT_IMPLEMENTED;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_bcast_init(ucc_tl_mlx5_task_t *task)
{
    task->super.post     = ucc_tl_mlx5_mcast_bcast_start;
    task->super.progress = ucc_tl_mlx5_mcast_collective_progress;

    return UCC_OK;
}
