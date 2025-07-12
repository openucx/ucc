/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "tl_mlx5_mcast_helper.h"
#include "tl_mlx5_mcast_rcache.h"

#define MCAST_BCAST_IN_PROGRESS(_req, _comm)                                         \
        (_req->to_send || _req->to_recv || _comm->pending_send ||                    \
         _comm->one_sided.pending_reads || (NULL != _req->bcast_rkeys_req) ||        \
         (_req->ag_schedule != NULL && _req->step != _req->ag_schedule->total_steps))

static inline ucc_status_t ucc_tl_mlx5_mcast_r_window_recycle(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                              ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    ucc_status_t     status        = UCC_OK;
    int              wsize         = comm->bcast_comm.wsize;
    int              num_free_win  = wsize - (comm->psn - comm->bcast_comm.last_acked);
    int              req_completed = (req->to_send == 0 && req->to_recv == 0);
    struct pp_packet *pp           = NULL;

    ucc_assert(comm->bcast_comm.recv_drop_packet_in_progress == false);
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

        while (req->exec_task != NULL) {
            EXEC_TASK_TEST("failed to complete the nb memcpy", req->exec_task, comm->lib);
        }

        comm->bcast_comm.n_mcast_reliable++;

        for (; comm->bcast_comm.last_acked < comm->psn; comm->bcast_comm.last_acked++) {
            pp = comm->r_window[comm->bcast_comm.last_acked & (wsize-1)];
            ucc_assert(pp != &comm->dummy_packet);
            comm->r_window[comm->bcast_comm.last_acked & (wsize-1)] = &comm->dummy_packet;

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

static inline ucc_status_t ucc_tl_mlx5_mcast_do_bcast(void *req_handle)
{
    ucc_status_t                   status = UCC_OK;
    ucc_tl_mlx5_mcast_coll_req_t  *req    = (ucc_tl_mlx5_mcast_coll_req_t *)req_handle;
    ucc_tl_mlx5_mcast_coll_comm_t *comm   = req->comm;
    int                            zcopy  = req->proto != MCAST_PROTO_EAGER;
    int                            wsize  = comm->bcast_comm.wsize;
    int                            num_free_win;
    int                            num_sent;
    int                            to_send;
    int                            to_recv;
    int                            to_recv_left;
    int                            pending_q_size;

    ucc_assert(req->comm->psn >= req->start_psn);

    status = ucc_tl_mlx5_mcast_check_nack_requests(comm, UINT32_MAX);
    if (status < 0) {
        return status;
    }

    if (ucc_unlikely(comm->bcast_comm.recv_drop_packet_in_progress)) {
        /* wait till parent resend the dropped packet */
        return UCC_INPROGRESS;
    }


    if (req->to_send || req->to_recv) {
        num_free_win = wsize - (comm->psn - comm->bcast_comm.last_acked);

        /* Send data if i'm root and there is a space in the window */
        if (num_free_win && req->am_root) {
            num_sent = req->num_packets - req->to_send;
			ucc_assert(req->to_send > 0);
			ucc_assert(req->first_send_psn + num_sent < comm->bcast_comm.last_acked + wsize);
            if (req->first_send_psn + num_sent < comm->bcast_comm.last_acked + wsize &&
                req->to_send) {
                /* How many to send: either all that are left (if they fit into window) or
                   up to the window limit */
                to_send = ucc_min(req->to_send,
                                  comm->bcast_comm.last_acked + wsize - (req->first_send_psn + num_sent));
                ucc_tl_mlx5_mcast_send(comm, req, to_send, zcopy);

                num_free_win = wsize - (comm->psn - comm->bcast_comm.last_acked);
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
                            " bcast_comm.last_acked psn %d stall threshold %d ",
                            pending_q_size, comm->psn, comm->bcast_comm.last_acked,
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

    if (req->to_send || req->to_recv || (zcopy && comm->psn != comm->bcast_comm.last_acked)) {
        return UCC_INPROGRESS;
    } else {
        return status;
    }
}

static inline ucc_status_t
ucc_tl_mlx5_mcast_validate_zero_copy_bcast_params(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                  ucc_tl_mlx5_mcast_coll_req_t  *req)
{

    if (req->num_packets % req->mcast_prepost_bucket_size != 0) {
        tl_debug(comm->lib, "Pipelined mcast bcast not supported: "
                "num_packets (%d) must be a multiple of mcast_prepost_bucket_size (%d) ",
                req->num_packets, req->mcast_prepost_bucket_size);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (comm->commsize % req->concurrency_level != 0) {
        tl_debug(comm->lib, "Pipelined mcast bcast not supported: "
                "team size (%d) must be a multiple of concurrency_level (%d).",
                comm->commsize, req->concurrency_level);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (req->length % comm->max_per_packet != 0) {
        tl_debug(comm->lib, "Pipelined mcast bcast not supported: "
                "length (%ld) must be a multiple of max_per_packet (%d).",
                req->length, comm->max_per_packet);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (req->mcast_prepost_bucket_size * req->concurrency_level * 2 > comm->params.rx_depth) {
        tl_debug(comm->lib, "Pipelined mcast bcast not supported: "
                "we only support the case prepost_bucket_size * concurrency_level * 2 > rx_depth, "
                "but got: prepost_bucket_size=%d, concurrency_level=%d, "
                "rx_depth=%d",
                 req->mcast_prepost_bucket_size, req->concurrency_level,
                 comm->params.rx_depth);
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

static inline ucc_status_t
ucc_tl_mlx5_mcast_prepare_zero_copy_bcast(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                          ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    ucc_tl_mlx5_mcast_reg_t                   *reg    = NULL;
    ucc_rank_t                                 root   = req->root;
    int                                        offset = 0;
    ucc_status_t                               status;
    ucc_rank_t                                 j, i;
    int                                        total_steps;
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *schedule;
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *curr_schedule;
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *prev_schedule;
    ucc_assert(comm->bcast_comm.truly_zero_copy_bcast_enabled);

    if (comm->commsize == 1) return UCC_OK;

    if (comm->mcast_group_count != 2) {
        tl_warn(comm->lib, "need exactly two mcast groups to enable zcopy bcast");
        return UCC_ERR_NOT_SUPPORTED;
    }
    req->concurrency_level = 1;

    req->mcast_prepost_bucket_size =
        ucc_min(req->num_packets, comm->bcast_comm.mcast_prepost_bucket_size);

    status = ucc_tl_mlx5_mcast_validate_zero_copy_bcast_params(comm, req);
    if (status != UCC_OK) {
        return status;
    }

    /* calculate the schedule and details of what we should
     * mcast and prepost to which mcast group at each stage*/
    total_steps = req->num_packets / (req->concurrency_level
                  * req->mcast_prepost_bucket_size) + 1;
    schedule    = ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_pipelined_ag_schedule_t) *
                             total_steps, "sched");
    if (!schedule) {
        tl_warn(comm->lib, "cannot allocate memory for schedule list");
        return UCC_ERR_NO_MEMORY;
    }

    /* generate schedule */
    for (i = 0; i < total_steps; i++) {
        curr_schedule = &(schedule[i]);
        if (i < total_steps - 1) {
            for (j = 0; j < req->concurrency_level; j++) {
                curr_schedule->prepost_buf_op[j].group_id =
                    j + req->concurrency_level * (i % 2);
                curr_schedule->prepost_buf_op[j].offset =
                    (offset + j * req->mcast_prepost_bucket_size) *
                    comm->max_per_packet;
                curr_schedule->prepost_buf_op[j].root = root;
                curr_schedule->prepost_buf_op[j].count =
                    req->mcast_prepost_bucket_size;
            }
        }
        if (i > 0) {
            prev_schedule = &(schedule[i - 1]);
            for (j = 0; j < req->concurrency_level; j++) {
                curr_schedule->multicast_op[j].group_id =
                    prev_schedule->prepost_buf_op[j].group_id;
                curr_schedule->multicast_op[j].offset =
                    prev_schedule->prepost_buf_op[j].offset;
                curr_schedule->multicast_op[j].offset_left =
                    prev_schedule->prepost_buf_op[j].offset;
                curr_schedule->multicast_op[j].root = root;
                if (!req->am_root) {
                    curr_schedule->multicast_op[j].to_recv =
                        prev_schedule->prepost_buf_op[j].count;
                }
                curr_schedule->to_recv += curr_schedule->multicast_op[j].to_recv;
                if (curr_schedule->multicast_op[j].root == comm->rank) {
                    curr_schedule->multicast_op[j].to_send_left =
                        prev_schedule->prepost_buf_op[j].count;
                    curr_schedule->to_send += curr_schedule->multicast_op[j].to_send_left;
                }
            }
        }
        if (req->am_root) {
            curr_schedule->prepost_buf_op_done = 1;
        }
        offset += req->mcast_prepost_bucket_size * req->concurrency_level;
    }
    tl_trace(comm->lib,
             "generated the schedule for pipelined zero copy bcast with total_steps %d",
             total_steps);
    schedule->total_steps = total_steps;
    req->total_steps      = total_steps;
    req->ag_schedule      = schedule;
    req->ag_counter       = comm->bcast_comm.coll_counter;
    comm->bcast_comm.coll_counter++;

    if (!req->am_root) {
        tl_trace(comm->lib, "registering recv buf of size %ld", req->length);
        ucc_assert(req->recv_rreg == NULL);
        status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->ptr, req->length, &reg);
        if (UCC_OK != status) {
             tl_warn(comm->lib, "unable to register receive buffer %p of size %ld",
                      req->ptr, req->length);
             ucc_free(schedule);
             return status;
        }
        req->recv_rreg = reg;
        req->recv_mr   = reg->mr;
    }

    if (comm->one_sided.reliability_enabled) {
        req->one_sided_reliability_scheme = ONE_SIDED_SYNCHRONOUS_PROTO;
    } else {
        req->one_sided_reliability_scheme = ONE_SIDED_NO_RELIABILITY;
    }

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_bcast_reliability_ready(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm = req->comm;
    ucc_tl_mlx5_mcast_reg_t       *reg  = NULL;
    ucc_status_t                   status;

    ucc_assert(req->ag_counter == comm->bcast_comm.under_progress_counter);

    if (!comm->one_sided.reliability_enabled || comm->one_sided.reliability_ready) {
        return UCC_OK;
    }

    if (req->bcast_rkeys_req) {
        status = comm->service_coll.coll_test(req->bcast_rkeys_req);
        if (status == UCC_OK) {
            req->bcast_rkeys_req = NULL;
            tl_trace(comm->lib, "bcast for remote_addr/rkey is completed");
            comm->one_sided.reliability_ready = 1;
        }
        return status;
    }

    /* initialize the structures needed by reliability protocol */
    memset(comm->one_sided.recvd_pkts_tracker, 0, comm->commsize * sizeof(uint32_t));
    memset(comm->one_sided.remote_slot_info, ONE_SIDED_INVALID, comm->commsize * sizeof(int));
    /* local slots state */
    comm->one_sided.slots_state = ONE_SIDED_INVALID;

    /* do nonblocking bcast over remote addresses/keys */
    if (!req->rreg) {
       /* register sbuf if it is not registered before */
       status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->ptr, req->length, &reg);
       if (UCC_OK != status) {
            return status;
       }
       req->rreg = reg;
       req->mr   = reg->mr;
    }
    if (req->am_root) {
        comm->one_sided.sendbuf_memkey_list[comm->rank].rkey        = req->mr->rkey;
        comm->one_sided.sendbuf_memkey_list[comm->rank].remote_addr = (uint64_t)req->ptr;
    }
    tl_trace(comm->lib, "bcast over buffer addresses/rkey: address %p rkey %d",
             req->ptr, req->mr->rkey);
    status = comm->service_coll.bcast_post(comm->p2p_ctx,
                                           &(comm->one_sided.sendbuf_memkey_list[req->root]),
                                           sizeof(ucc_tl_mlx5_mcast_slot_mem_info_t),
                                           req->root,
                                           &req->bcast_rkeys_req);
    if (UCC_OK != status) {
        tl_error(comm->lib, "oob bcast failed during one-sided reliability reset of a collective call");
        return status;
    }

    return UCC_INPROGRESS;
}

static inline ucc_status_t
ucc_tl_mlx5_mcast_check_zcopy_bcast_collective(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                               ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *sched = req->ag_schedule;

    ucc_assert(req->bcast_rkeys_req == NULL);
    if (comm->one_sided.pending_reads) {
        return ucc_tl_mlx5_mcast_progress_one_sided_communication(comm, req);
    }
    if (req->step == req->total_steps) {
        return UCC_OK;
    }
    if (!sched[req->step].prepost_buf_op_done || !sched[req->step].multicast_op_done) {
        // it is not yet the time to start the reliability protocol
        return UCC_INPROGRESS;
    }
    if (sched[req->step].num_recvd == sched[req->step].to_recv) {
        /* check for out of order packets, if any root sent a out of order
         * packet to us in the current step, go ahead and issue RDMA READ
         * from that root for this specific piece of send buffer */
        return ucc_tl_mlx5_mcast_reliable_zcopy_pipelined_one_sided_get(comm, req, NULL);
    } else if (!comm->timer) {
        if (comm->stalled < DROP_THRESHOLD) {
            comm->stalled++;
        } else {
            // kick the timer
            comm->timer   = ucc_tl_mlx5_mcast_get_timer();
            comm->stalled = 0;
        }
    } else {
        if (comm->stalled < DROP_THRESHOLD) {
            comm->stalled++;
        } else {
            // calcuate the current time and check if it's time to do RDMA READ
            if (ucc_tl_mlx5_mcast_get_timer() - comm->timer >=
                    comm->ctx->params.timeout) {
                comm->timer = 0;
                ucc_assert(sched[req->step].to_recv >= sched[req->step].num_recvd);
                tl_debug(comm->lib, "zcopy bcast timeout %d pending packets to recv %d on step %d",
                         comm->ctx->params.timeout, sched[req->step].to_recv - sched[req->step].num_recvd,
                         req->step);
                return ucc_tl_mlx5_mcast_reliable_zcopy_pipelined_one_sided_get(comm, req, NULL);
            } else {
                comm->stalled = 0;
            }
        }
    }
    return UCC_INPROGRESS;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_do_zero_copy_pipelined_bcast(void *req_handle)
{
    ucc_tl_mlx5_mcast_coll_req_t              *req   = (ucc_tl_mlx5_mcast_coll_req_t *)req_handle;
    ucc_tl_mlx5_mcast_coll_comm_t             *comm  = req->comm;
    const int                                  zcopy = req->proto != MCAST_PROTO_EAGER;
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *sched = req->ag_schedule;
    int                                        root  = req->root;
    int                                        num_recvd, to_send_left,
                                               j, group_id, num_packets, count;
    size_t                                     offset, offset_left;
    ucc_status_t                               status;

    if (req->ag_counter != req->comm->bcast_comm.under_progress_counter) {
        /* it is not this task's turn for progress */
        ucc_assert(req->comm->bcast_comm.under_progress_counter < req->ag_counter);
        return UCC_INPROGRESS;
    }

    status = ucc_tl_mlx5_mcast_bcast_reliability_ready(req);
    if (UCC_OK != status) {
        return status;
    }

    ucc_assert(req->to_recv >= 0 && req->to_send >= 0);
    if (req->barrier_req) {
        status = comm->service_coll.coll_test(req->barrier_req);
        if (status != UCC_OK) {
            return status;
        }
        tl_trace(comm->lib, "barrier at end of req->step %d is completed", req->step);
        req->barrier_req = NULL;
        req->step++;
        if (comm->one_sided.reliability_enabled) {
            memset(comm->one_sided.recvd_pkts_tracker, 0, comm->commsize * sizeof(uint32_t));
        }
    }

    if (req->step < sched->total_steps) {
        if (!sched[req->step].multicast_op_done) {
            for (j = 0; j < req->concurrency_level; j++) {
                ucc_assert(root == sched[req->step].multicast_op[j].root);
                group_id     = sched[req->step].multicast_op[j].group_id;
                to_send_left = sched[req->step].multicast_op[j].to_send_left;
                offset_left  = sched[req->step].multicast_op[j].offset_left;
                num_packets  = ucc_min(comm->bcast_comm.max_push_send - comm->pending_send, to_send_left);
                if (to_send_left &&
                    (comm->bcast_comm.max_push_send - comm->pending_send) > 0) {
                    ucc_assert(req->am_root);
                    status = ucc_tl_mlx5_mcast_send_collective(comm, req, num_packets,
                                                               zcopy, group_id, offset_left);
                    if (UCC_OK != status) {
                        return status;
                    }
                    sched[req->step].multicast_op[j].to_send_left -= num_packets;
                    sched[req->step].multicast_op[j].offset_left  += (num_packets * comm->max_per_packet);
                }
                if (comm->pending_send) {
                    status = ucc_tl_mlx5_mcast_poll_send(comm);
                    if (status != UCC_OK) {
                        return status;
                    }
                }
                if (!sched[req->step].multicast_op[j].to_send_left && !comm->pending_send) {
                    tl_trace(comm->lib, "done with mcast ops step %d group id %d to_send %d",
                             req->step, group_id, sched[req->step].to_send);
                    sched[req->step].multicast_op_done = 1;
                    break;
                }
            }
        }

        if (!sched[req->step].prepost_buf_op_done) {
            /* prepost the user buffers for none roots */
            for (j = 0; j < req->concurrency_level; j++) {
                group_id = sched[req->step].prepost_buf_op[j].group_id;
                count    = sched[req->step].prepost_buf_op[j].count;
                offset   = sched[req->step].prepost_buf_op[j].offset;
                if (count) {
                    ucc_assert(!req->am_root);
                    status   = ucc_tl_mlx5_mcast_post_user_recv_buffers(comm, req, group_id, root,
                                                                        UCC_COLL_TYPE_BCAST, count,
                                                                        offset);
                    if (UCC_OK != status) {
                        return status;
                    }
                }
                /* progress the recvd packets in between */
                if (req->to_recv) {
                    num_recvd = ucc_tl_mlx5_mcast_recv_collective(comm, req,
                                                                  req->to_recv,
                                                                  UCC_COLL_TYPE_BCAST);
                    if (num_recvd < 0) {
                        tl_error(comm->lib, "a failure happend during cq polling");
                        status = UCC_ERR_NO_MESSAGE;
                        return status;
                    }
                    sched[req->step].num_recvd += num_recvd;
                }
            }
            tl_trace(comm->lib, "done with prepost bufs step %d group id %d count %d offset %ld root %d",
                     req->step, group_id, count, offset, root);
            sched[req->step].prepost_buf_op_done = 1;
        }

        if (req->to_recv) {
            num_recvd = ucc_tl_mlx5_mcast_recv_collective(comm, req,
                                                          req->to_recv,
                                                          UCC_COLL_TYPE_BCAST);
            if (num_recvd < 0) {
                tl_error(comm->lib, "a failure happend during cq polling");
                status = UCC_ERR_NO_MESSAGE;
                return status;
            }
            sched[req->step].num_recvd += num_recvd;
        }

        if (sched[req->step].prepost_buf_op_done &&
            sched[req->step].multicast_op_done &&
            sched[req->step].num_recvd == sched[req->step].to_recv) {
            // current step done
            ucc_assert(sched[req->step].prepost_buf_op_done && sched[req->step].multicast_op_done);
            ucc_assert(req->barrier_req == NULL);
            status = comm->service_coll.barrier_post(comm->p2p_ctx, &req->barrier_req);
            if (status != UCC_OK) {
                return status;
            }
            tl_trace(comm->lib, "init global sync req->step %d", req->step);
        }
    }

    if (comm->one_sided.reliability_enabled) {
        status = ucc_tl_mlx5_mcast_check_zcopy_bcast_collective(comm, req);
        if (status < 0) {
            return status;
        }
    }

    if (req->barrier_req != NULL ||
        MCAST_BCAST_IN_PROGRESS(req, comm)) {
        return UCC_INPROGRESS;
    }

    /* all completed */
    assert(req->step == sched->total_steps);
    return UCC_OK;
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

    req->num_packets = ucc_div_round_up(req->length, comm->max_per_packet);
    req->offset      = 0;
    req->to_send     = req->am_root ? req->num_packets : 0;
    req->to_recv     = req->am_root ? 0 : req->num_packets;

    if (comm->bcast_comm.truly_zero_copy_bcast_enabled) {
        status = ucc_tl_mlx5_mcast_prepare_zero_copy_bcast(comm, req);
        if (UCC_OK != status) {
            return status;
        }
        req->progress = ucc_tl_mlx5_mcast_do_zero_copy_pipelined_bcast;
    } else {
        status = ucc_tl_mlx5_mcast_prepare_reliable(comm, req, req->root);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
        req->start_psn    = comm->bcast_comm.last_psn;
        req->last_pkt_len = req->length - (req->num_packets - 1) * comm->max_per_packet;
        ucc_assert(req->last_pkt_len > 0 && req->last_pkt_len <= comm->max_per_packet);
        comm->bcast_comm.last_psn += req->num_packets;
        req->first_send_psn        = req->start_psn;
        req->progress              = ucc_tl_mlx5_mcast_do_bcast;
    }

    if (req->am_root) {
        if (req->proto != MCAST_PROTO_EAGER) {
            status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->ptr, req->length, &reg);
            if (UCC_OK != status) {
                if (req->ag_schedule) {
                    ucc_free(req->ag_schedule);
                }
                return status;
            }
            req->rreg = reg;
            req->mr   = reg->mr;
        }
    }

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

    req = ucc_mpool_get(&comm->ctx->mcast_req_mp);
    if (!req) {
        tl_error(comm->lib, "failed to get mcast req");
        return UCC_ERR_NO_MEMORY;
    }
    memset(req, 0, sizeof(ucc_tl_mlx5_mcast_coll_req_t));

    status = ucc_tl_mlx5_mcast_prepare_bcast(buf, size, root, comm, req);
    if (UCC_OK != status) {
        ucc_mpool_put(req);
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

    task->coll_mcast.req_handle = NULL;

    status = ucc_tl_mlx5_mcast_coll_do_bcast(buf, data_size, root, comm,
                                             &task->coll_mcast.req_handle);
    if (status < 0) {
        tl_error(UCC_TASK_LIB(task), "mcast_coll_do_bcast failed:%d", status);
        coll_task->status = status;
        return ucc_task_complete(coll_task);
    }

    ucc_assert(task->coll_mcast.req_handle != NULL);

    coll_task->status                       = status;
    task->coll_mcast.req_handle->coll_task = coll_task;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(mlx5_team)->pq, &task->super);
}

static inline void ucc_tl_mlx5_mcast_bcast_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t           *task   = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_mcast_coll_req_t *req    = task->coll_mcast.req_handle;

    ucc_assert(req != NULL);

    coll_task->status = (req->progress)(req);
    if (coll_task->status < 0) {
        tl_error(UCC_TASK_LIB(task), "progress mcast bcast failed:%d", coll_task->status);
    }
}

static inline ucc_status_t
ucc_tl_mlx5_mcast_check_comm_level_cap(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *mlx5_team           =
        ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_mcast_coll_comm_t *comm                =
        mlx5_team->mcast->mcast_comm;
    ucc_coll_args_t               *args                = &coll_args->args;
    size_t                         buf_size            =
        ucc_dt_size(args->src.info.datatype) * args->src.info.count;
    ucc_memory_type_t              mem_type            = args->src.info.mem_type;
    ucc_coll_type_t                coll_type           = args->coll_type;
    bool                           mcast_enabled       = false;
    bool                           zero_copy_supported = false;

    /* Check supported memory types */
    if (mem_type != UCC_MEMORY_TYPE_CUDA &&
        mem_type != UCC_MEMORY_TYPE_HOST) {
        tl_trace(comm->lib,
                 "unsupported memory type %d for mlx5 multicast",
                 mem_type);
        return UCC_ERR_NO_RESOURCE;
    }

    /* Check CUDA memory enablement consistency */
    if ((comm->cuda_mem_enabled && mem_type == UCC_MEMORY_TYPE_HOST) ||
        (!comm->cuda_mem_enabled && mem_type == UCC_MEMORY_TYPE_CUDA)) {
        tl_trace(comm->lib,
                 "CUDA memory usage inconsistent with configuration "
                 "(enabled: %d, requested: %d)",
                 comm->cuda_mem_enabled,
                 mem_type == UCC_MEMORY_TYPE_CUDA);
        return UCC_ERR_NO_RESOURCE;
    }

    /* Check if multicast is enabled for the specific collective type */
    if (coll_type == UCC_COLL_TYPE_BCAST &&
        comm->context->mcast_bcast_enabled) {
        mcast_enabled = true;
    } else if (coll_type == UCC_COLL_TYPE_ALLGATHER &&
               comm->context->mcast_allgather_enabled) {
        mcast_enabled = true;
    }

    if (!mcast_enabled) {
        tl_trace(comm->lib,
                 "multicast for collective type %s is disabled",
                 ucc_coll_type_str(coll_type));
        return UCC_ERR_NO_RESOURCE;
    }

    /* Check zero-copy requirements */
    if (coll_type == UCC_COLL_TYPE_BCAST &&
        comm->bcast_comm.truly_zero_copy_bcast_enabled) {
        zero_copy_supported = true;
    } else if (coll_type == UCC_COLL_TYPE_ALLGATHER &&
               comm->allgather_comm.truly_zero_copy_allgather_enabled) {
        zero_copy_supported = true;
    }

    if (zero_copy_supported) {
        if (coll_type == UCC_COLL_TYPE_BCAST &&
            (buf_size < comm->truly_zero_copy_coll_min_msg ||
             buf_size % comm->max_per_packet != 0)) {
            tl_trace(comm->lib,
                     "zero-copy bcast requirements not met "
                     "(buf_size: %zu, min_msg: %d, max_packet: %d)",
                     buf_size,
                     comm->truly_zero_copy_coll_min_msg,
                     comm->max_per_packet);
            return UCC_ERR_NO_RESOURCE;
        } else if (coll_type == UCC_COLL_TYPE_ALLGATHER &&
                   buf_size >= comm->max_per_packet &&
                   buf_size % comm->max_per_packet != 0) {
            /* For allgather, allow small messages (< max_per_packet) and
             * only enforce packet size multiple for large messages */
            tl_trace(comm->lib,
                     "zero-copy allgather requirements not met: "
                     "large message (>= %d bytes) must be multiple of max_per_packet (%d)",
                     comm->max_per_packet, comm->max_per_packet);
            return UCC_ERR_NO_RESOURCE;
        }
    }

    /* Specific checks for CUDA BCAST without zero-copy */
    if (mem_type == UCC_MEMORY_TYPE_CUDA &&
        coll_type == UCC_COLL_TYPE_BCAST &&
        buf_size > CUDA_MEM_MCAST_BCAST_MAX_MSG &&
        !comm->bcast_comm.truly_zero_copy_bcast_enabled) {
        tl_trace(comm->lib,
                 "mcast cuda bcast with size %zu not supported without "
                 "zero copy enabled (max size: %d)",
                 buf_size, CUDA_MEM_MCAST_BCAST_MAX_MSG);
        return UCC_ERR_NO_RESOURCE;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_check_support(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *team)
{
    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args) ||
        ((coll_args->args.coll_type == UCC_COLL_TYPE_ALLGATHER) &&
         (UCC_IS_INPLACE(coll_args->args) || UCC_IS_PERSISTENT(coll_args->args)))) {
        tl_trace(team->context->lib, "mcast collective not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_OK != ucc_tl_mlx5_mcast_check_comm_level_cap(coll_args, team)) {
        tl_trace(team->context->lib, "mcast collective not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_bcast_init(ucc_tl_mlx5_task_t *task)
{
    task->super.post     = ucc_tl_mlx5_mcast_bcast_start;
    task->super.progress = ucc_tl_mlx5_mcast_bcast_progress;
    task->super.flags    = UCC_COLL_TASK_FLAG_EXECUTOR;

    return UCC_OK;
}
