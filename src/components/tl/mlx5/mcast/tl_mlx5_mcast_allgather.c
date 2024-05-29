/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_helper.h"
#include "tl_mlx5_coll.h"
#include "tl_mlx5_mcast_rcache.h"
#include "tl_mlx5_mcast_progress.h"
#include "tl_mlx5_mcast_allgather.h"
#include <inttypes.h>

static inline ucc_status_t ucc_tl_mlx5_mcast_check_zcopy_collective(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                                    ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    //  1. check if any out of order packet has arrvied in
    //  the current req->step
    //  2. if so, wait until all the expected packets have
    //  arrived for this step or timeout has passed
    //  3. it is also possible that some packets are dropped so we
    //  wait until timeout to issue RDMA Get
    //  4. issue RDMA read for the current req->step's offset
    //  5. only the synchronous one-sided reliablity scheme is supported
    
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *sched     = req->ag_schedule;
    int                                        completed = 0;
    ucc_status_t                               status;
    
    ucc_assert(sched != NULL);
    ucc_assert(comm->one_sided.reliability_ready);

    if (comm->one_sided.rdma_read_in_progress) {
        return ucc_tl_mlx5_mcast_progress_one_sided_communication(comm, req);
    }

    /* check if remote rkey/address have arrived - applicable for sync design */
    if (req->allgather_rkeys_req && UCC_OK ==
        ucc_collective_test(&((ucc_service_coll_req_t *)req->allgather_rkeys_req)->task->super)) {
        ucc_assert(ONE_SIDED_SYNCHRONOUS_PROTO == req->one_sided_reliability_scheme);
        ucc_service_coll_finalize(req->allgather_rkeys_req);
        req->allgather_rkeys_req = NULL;
        tl_trace(comm->lib, "Allgather for remote_addr/rkey is completed");
    }

    if (!sched[req->step].prepost_buf_op_done ||
            !sched[req->step].multicast_op_done) {
        // it is not yet the time to start the reliability protocol
        return UCC_INPROGRESS;
    }

    if (sched[req->step].num_recvd == sched[req->step].to_recv && NULL == req->allgather_rkeys_req) {
        /* check for out of order packets, if any root sent a out of order
         * packet to us in the current step, go ahead and issue RDMA READ
         * from that root for this specific piece of send buffer */
        status = ucc_tl_mlx5_mcast_reliable_zcopy_pipelined_one_sided_get(comm, req, &completed);
        if (status < 0) {
            return status;
        } else if (completed == req->concurreny_level) {
            // we recv'd all the packets from this step
            return UCC_OK;
        } else {
            return UCC_INPROGRESS;
        }
    } else if (!comm->timer) {
        if (comm->stalled < DROP_THRESHOLD) {
            comm->stalled++;
        } else {
            // kick the timer
            comm->timer   = ucc_tl_mlx5_mcast_get_timer();
            comm->stalled = 0;
        }
    } else {
        if (comm->stalled < DROP_THRESHOLD || (NULL != req->allgather_rkeys_req)) {
            comm->stalled++;
        } else {
            // calcuate the current time and check if it's time to do RDMA READ
            if (ucc_tl_mlx5_mcast_get_timer() - comm->timer >=
                    comm->ctx->params.timeout) {
                tl_debug(comm->lib, "[REL] time out");
                status =
                    ucc_tl_mlx5_mcast_reliable_zcopy_pipelined_one_sided_get(comm,
                                                                             req, &completed);
                if (status < 0) {
                    return status;
                } else if (completed == req->concurreny_level) {
                    // we recv'd all the packets from this step
                    return UCC_OK;
                } else {
                    return UCC_INPROGRESS;
                }
            } else {
                comm->stalled = 0;
            }
        }
    }

    return UCC_INPROGRESS;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_check_collective(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                              ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    ucc_status_t status;

    ucc_assert(comm->one_sided.reliability_ready);

    if (comm->one_sided.rdma_read_in_progress) {
        return ucc_tl_mlx5_mcast_progress_one_sided_communication(comm, req);
    }

    /* check if remote rkey/address have arrived - applicable for sync design */
    if (req->allgather_rkeys_req && UCC_OK ==
        ucc_collective_test(&((ucc_service_coll_req_t *)req->allgather_rkeys_req)->task->super)) {
        ucc_assert(ONE_SIDED_SYNCHRONOUS_PROTO == req->one_sided_reliability_scheme);
        ucc_service_coll_finalize(req->allgather_rkeys_req);
        req->allgather_rkeys_req = NULL;
        tl_trace(comm->lib, "Allgather for remote_addr/rkey is completed");
    }

    if (!req->to_send && !req->to_recv) {
        // all have been received, nothing to do
        return UCC_OK;

    } else if (req->to_send) {
        // it is not yet the time to start the reliability protocol
        return UCC_INPROGRESS;
    }

    if (!comm->timer) {
        if (comm->stalled < DROP_THRESHOLD) {
            comm->stalled++;
        } else {
            // kick the timer
            comm->timer = ucc_tl_mlx5_mcast_get_timer();
            comm->stalled = 0;
        }
    } else {
        if (comm->stalled < DROP_THRESHOLD || (NULL != req->allgather_rkeys_req)) {
            comm->stalled++;
        } else {
            // calcuate the current time and check if it's time to do RDMA READ
            if (ucc_tl_mlx5_mcast_get_timer() - comm->timer >=
                    comm->ctx->params.timeout) {
                tl_debug(comm->lib, "[REL] time out req->to_recv %d left out of total of %d packets",
                         req->to_recv, req->num_packets * comm->commsize);
                status = ucc_tl_mlx5_mcast_reliable_one_sided_get(comm, req, NULL);
                if (UCC_OK != status) {
                    return status;
                }
            } else {
                comm->stalled = 0;
            }
        }
    }

    return UCC_INPROGRESS;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_reset_reliablity(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm = req->comm;
    ucc_tl_mlx5_mcast_reg_t       *reg  = NULL;
    ucc_status_t                   status;

    ucc_assert(req->ag_counter == comm->ag_under_progress_counter);

    if (comm->one_sided.reliability_enabled && !comm->one_sided.reliability_ready) {
        /* initialize the structures needed by reliablity protocol */ 
        memset(comm->one_sided.recvd_pkts_tracker, 0, comm->commsize * sizeof(uint32_t));
        memset(comm->one_sided.remote_slot_info, ONE_SIDED_INVALID, comm->commsize * sizeof(uint32_t));
        /* local slots state */
        comm->one_sided.slots_state = ONE_SIDED_INVALID;

        if (ONE_SIDED_SYNCHRONOUS_PROTO == req->one_sided_reliability_scheme) {
            /* do nonblocking allgather over remote addresses/keys */
            if (!req->rreg) {
               /* register sbuf if it is not registered before */
               status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->ptr, req->length, &reg);
               if (UCC_OK != status) {
                    return status;
               }
               req->rreg = reg;
               req->mr   = reg->mr;
            }

            comm->one_sided.sendbuf_memkey_list[comm->rank].rkey        = req->mr->rkey;
            comm->one_sided.sendbuf_memkey_list[comm->rank].remote_addr = (uint64_t)req->ptr;

            tl_trace(comm->lib, "Allgather over sendbuf addresses/rkey: address %p rkey %d",
                     req->ptr, req->mr->rkey);

            status = comm->service_coll.allgather_post(comm->p2p_ctx, NULL /* in-place */,
                                                       comm->one_sided.sendbuf_memkey_list,
                                                       sizeof(ucc_tl_mlx5_mcast_slot_mem_info_t),
                                                       &req->allgather_rkeys_req);
            if (UCC_OK != status) {
                tl_error(comm->lib, "oob allgather failed during one-sided reliability reset of a collective call");
                return status;
            }
        }

        memset(comm->pending_recv_per_qp, 0, sizeof(int) * MAX_GROUP_COUNT);
        comm->one_sided.reliability_ready = 1;
    }

    return UCC_OK;
}

static inline void ucc_tl_mlx5_mcast_init_async_reliability_slots(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm = req->comm;
    void                          *dest;

    ucc_assert(req->ag_counter == comm->ag_under_progress_counter);

    if (ONE_SIDED_ASYNCHRONOUS_PROTO == req->one_sided_reliability_scheme &&
                   ONE_SIDED_INVALID == comm->one_sided.slots_state) {
        /* copy the sendbuf and seqnum to the internal temp buf in case other processes need
         * to read from it */
        ucc_assert(req->length <= comm->one_sided.reliability_scheme_msg_threshold);
        dest = comm->one_sided.slots_buffer + (req->ag_counter % ONE_SIDED_SLOTS_COUNT)
               * comm->one_sided.slot_size;
    
        /* both user buffer and reliablity slots are on host */
        memcpy(dest + ONE_SIDED_SLOTS_INFO_SIZE, req->ptr, req->length);
        memcpy(dest, &req->ag_counter, ONE_SIDED_SLOTS_INFO_SIZE);

        comm->one_sided.slots_state = ONE_SIDED_VALID;
    }
}

static inline ucc_status_t ucc_tl_mlx5_mcast_do_zero_copy_pipelined_allgather(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_tl_mlx5_mcast_coll_comm_t             *comm                = req->comm;
    const int                                  zcopy               = req->proto != MCAST_PROTO_EAGER;
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *sched               = req->ag_schedule;
    int                                        ready_for_next_step = 0;
    int                                        num_recvd, root, to_send_left, j, group_id, num_packets, count;
    size_t                                     offset, offset_left;
    ucc_status_t                               status;


    status = ucc_tl_mlx5_mcast_reset_reliablity(req);
    if (UCC_OK != status) {
        return status;
    }
    
    ucc_assert(ONE_SIDED_ASYNCHRONOUS_PROTO != req->one_sided_reliability_scheme);
    ucc_assert(req->to_recv>=0 && req->to_send >=0);
    ucc_assert(sched != NULL);

    if (req->barrier_req) {
        // global sync is in progress
        status = ucc_collective_test(&((ucc_service_coll_req_t *)req->barrier_req)->task->super);
        if (status < 0 || UCC_INPROGRESS == status) {
            return status;
        } else {
            ucc_service_coll_finalize(req->barrier_req);
            req->barrier_req = NULL;
            tl_trace(comm->lib, "barrier at end of req->step %d is completed", req->step);
        }
    
        // a new step is going to start

        if (comm->one_sided.reliability_enabled) {
            //reset the one_sided.recvd_pkts_tracker[] for next step
            memset(comm->one_sided.recvd_pkts_tracker, 0, sizeof(int) * comm->commsize);
        }

       // ucc_assert(!comm->pending_send && !comm->pending_recv);

        req->step++;

        if (req->step == sched->total_steps) {
            ucc_assert(!req->to_send && !req->to_recv);
            return UCC_OK;
        }
    }

    ucc_assert(req->step < sched->total_steps);

    if (!sched[req->step].multicast_op_done) {
        for (j = 0; j < req->concurreny_level; j++) {
            root = sched[req->step].multicast_op[j].root;
            if (comm->rank == root) {
                /* it's my turn to place mcast packets on wire */ 
                group_id     = sched[req->step].multicast_op[j].group_id;
                to_send_left = sched[req->step].multicast_op[j].to_send_left;
                offset_left  = sched[req->step].multicast_op[j].offset_left;
                num_packets  = ucc_min(comm->max_push_send - comm->pending_send, to_send_left);
                if (to_send_left &&
                    (comm->max_push_send - comm->pending_send) > 0) {

                    status = ucc_tl_mlx5_mcast_send_collective(comm, req, num_packets,
                                                               zcopy, UCC_COLL_TYPE_ALLGATHER,
                                                               group_id, offset_left);
                    if (UCC_OK != status) {
                        return status;
                    }

                    sched[req->step].multicast_op[j].to_send_left -= num_packets;
                    sched[req->step].multicast_op[j].offset_left  += (num_packets * comm->max_per_packet);
                }

                if (comm->pending_send) {
                    if (ucc_tl_mlx5_mcast_poll_send(comm) < 0) {
                        return UCC_ERR_NO_MESSAGE;
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
    }

    if (!sched[req->step].prepost_buf_op_done) {
        /* prepost the user buffers for a set of processes */
        for (j = 0; j < req->concurreny_level; j++) {
            root     = sched[req->step].prepost_buf_op[j].root;
            group_id = sched[req->step].prepost_buf_op[j].group_id;
            count    = sched[req->step].prepost_buf_op[j].count;
            offset   = sched[req->step].prepost_buf_op[j].offset;

            status   = ucc_tl_mlx5_mcast_post_user_recv_buffers(comm, req, group_id, root,
                                                                UCC_COLL_TYPE_ALLGATHER, count,
                                                                offset);
            if (UCC_OK != status) {
                return status;
            }

            if (req->to_recv) {
                num_recvd = ucc_tl_mlx5_mcast_recv_collective(comm, req,
                                                              req->to_recv,
                                                              UCC_COLL_TYPE_ALLGATHER);

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
                                                      UCC_COLL_TYPE_ALLGATHER);

        if (num_recvd < 0) {
            tl_error(comm->lib, "a failure happend during cq polling");
            status = UCC_ERR_NO_MESSAGE;
            return status;
        }

        sched[req->step].num_recvd += num_recvd;
    }

    if (comm->one_sided.reliability_enabled) {
        status = ucc_tl_mlx5_mcast_check_zcopy_collective(comm, req);
        if (UCC_INPROGRESS != status && UCC_OK != status) {
            return status;
        }
        if (UCC_OK == status) {
            ready_for_next_step = 1;
        }
    } else if (sched[req->step].prepost_buf_op_done &&
               sched[req->step].multicast_op_done &&
               sched[req->step].num_recvd == sched[req->step].to_recv) {
        ready_for_next_step = 1;
    }

    if (ready_for_next_step) {
        // go to barrier
        // after barrier, increase req->step
        ucc_assert(sched[req->step].prepost_buf_op_done &&
                sched[req->step].multicast_op_done);
        ucc_assert(req->barrier_req == NULL);
        status = comm->service_coll.barrier_post(comm->p2p_ctx, &req->barrier_req);
        if (status != UCC_OK) {
            return status;
        }
        tl_trace(comm->lib, "init global sync req->step %d", req->step);
    }

    return UCC_INPROGRESS;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_do_allgather(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_status_t                   status = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm   = req->comm;
    const int                      zcopy  = req->proto != MCAST_PROTO_EAGER;
    int                            num_recvd;

    ucc_assert(req->to_recv>=0 && req->to_send >=0);

    status = ucc_tl_mlx5_mcast_reset_reliablity(req);
    if (UCC_OK != status) {
        return status;
    }

    if (req->to_send || req->to_recv) {
        ucc_assert(comm->max_push_send >= comm->pending_send);
        if (req->to_send &&
            (comm->max_push_send - comm->pending_send) > 0) {
            ucc_tl_mlx5_mcast_send_collective(comm, req, ucc_min(comm->max_push_send -
                                              comm->pending_send, req->to_send),
                                              zcopy, UCC_COLL_TYPE_ALLGATHER, -1, SIZE_MAX);
        }

        ucc_tl_mlx5_mcast_init_async_reliability_slots(req);

        if (req->to_recv) {
            num_recvd = ucc_tl_mlx5_mcast_recv_collective(comm, req, req->to_recv, UCC_COLL_TYPE_ALLGATHER);
            if (num_recvd < 0) {
                tl_error(comm->lib, "a failure happend during cq polling");
                status = UCC_ERR_NO_MESSAGE;
                return status;
            }
        }
    }

    if (comm->pending_send) {
        if (ucc_tl_mlx5_mcast_poll_send(comm) < 0) {
            return UCC_ERR_NO_MESSAGE;
        }
    }

    if (comm->one_sided.reliability_enabled) {
        status = ucc_tl_mlx5_mcast_check_collective(comm, req);
        if (UCC_INPROGRESS != status && UCC_OK != status) {
            return status;
        }
    }

    if (MCAST_ALLGATHER_IN_PROGRESS(req, comm)) {
        return UCC_INPROGRESS;
    } else {
        if (ONE_SIDED_SYNCHRONOUS_PROTO == req->one_sided_reliability_scheme) {
            if (!req->barrier_req) {
                // mcast operations are done and now go to barrier
               status = comm->service_coll.barrier_post(comm->p2p_ctx, &req->barrier_req);
               if (status != UCC_OK) {
                   return status;
               }
               tl_trace(comm->lib, "mcast operations are done and now go to barrier");
               return UCC_INPROGRESS;
            } else if (UCC_OK ==
                ucc_collective_test(&((ucc_service_coll_req_t *)req->barrier_req)->task->super)) {
                ucc_service_coll_finalize(req->barrier_req);
                req->barrier_req = NULL;
                tl_trace(comm->lib, "Barrier at the end of allgather is completed");
            } else {
                // still progressing barrier
                return UCC_INPROGRESS;
            }
        }

      /* this task is completed */
      return UCC_OK;
    }
}

ucc_status_t ucc_tl_mlx5_mcast_test_allgather(ucc_tl_mlx5_mcast_coll_req_t* req)
{
    ucc_status_t status = UCC_OK;
    
    if (req->comm->truly_zero_copy_allgather_enabled) {
        status = ucc_tl_mlx5_mcast_do_zero_copy_pipelined_allgather(req);
    } else {
        status = ucc_tl_mlx5_mcast_do_allgather(req);
    }
    if (UCC_OK == status) {
        ucc_assert(req->comm->ctx != NULL);
        ucc_tl_mlx5_mcast_mem_deregister(req->comm->ctx, req->rreg);
        req->rreg = NULL;
        ucc_tl_mlx5_mcast_mem_deregister(req->comm->ctx, req->recv_rreg);
        req->recv_rreg = NULL;
        if (req->ag_schedule) {
            ucc_free(req->ag_schedule);
            req->ag_schedule = NULL;
        }
         /* reset the reliability structures so that it gets initialized again for the next
        * allgather */
        req->comm->one_sided.reliability_ready = 0;
        req->comm->stalled                     = 0;
        req->comm->timer                       = 0;
    }

    return status;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_prepare_zero_copy_allgather(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                                         ucc_tl_mlx5_mcast_coll_req_t  *req)
{
    ucc_tl_mlx5_mcast_reg_t                   *reg    = NULL;
    ucc_rank_t                                 root   = 0;
    int                                        offset = 0;
    ucc_status_t                               status;
    ucc_rank_t                                 j, i;
    int                                        total_steps;
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *new_sched;

    ucc_assert(comm->truly_zero_copy_allgather_enabled);

    req->concurreny_level = comm->mcast_group_count; // this must be equal or
                                                     // less than number of groups
    /*
     * at each stage half of the mcast groups are
     * ready for receiving mcast packets while the
     * other half are getting prepared by preposting
     * recv buffers
     */
    if (req->concurreny_level > 1) {
        req->concurreny_level /= 2;
    } else {
        tl_warn(comm->lib, "not enough mcast groups to enable zcopy pipeline allgather");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (req->concurreny_level > comm->commsize) {
        req->concurreny_level = comm->commsize;
    }

    ucc_assert(req->concurreny_level <= ONE_SIDED_MAX_CONCURRENT_LEVEL);

    if (comm->mcast_prepost_bucket_size > req->num_packets) {
        req->mcast_prepost_bucket_size = req->num_packets;
    } else {
        req->mcast_prepost_bucket_size = comm->mcast_prepost_bucket_size;
    }

    if ((req->concurreny_level % 2 == 0 &&
        req->num_packets % req->mcast_prepost_bucket_size != 0) ||
        (comm->commsize % req->concurreny_level != 0) ||
        (req->length % comm->max_per_packet != 0)) {
        tl_warn(comm->lib, "Pipelined mcast allgather not supported: "
                "num_packets %d mcast_prepost_bucket_size %d "
                "length %ld max_per_packet %d "
                "team size %d concurreny_level %d",
                req->num_packets, req->mcast_prepost_bucket_size, req->length,
                comm->max_per_packet, comm->commsize, req->concurreny_level);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (req->mcast_prepost_bucket_size*req->concurreny_level*2 >
            comm->params.rx_depth) {
        tl_warn(comm->lib, "either reduce prepost_bucket_size or mcast group "
                "count or increase recv queue size "
                "mcast_prepost_bucket_size %d concurreny_level %d "
                "rx_depth %d",
                 req->mcast_prepost_bucket_size, req->concurreny_level,
                 comm->params.rx_depth);
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* now calculate the schedule and details of what we should
     * mcast and prepost to which mcast group at each stage*/
    total_steps = req->num_packets * (comm->commsize / req->concurreny_level)
                / req->mcast_prepost_bucket_size + 1;

    new_sched = ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_pipelined_ag_schedule_t) * total_steps, "sched");
    if (!new_sched) {
        tl_warn(comm->lib, "cannot allocate memory for schedule list");
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < total_steps; i++) {
        ucc_assert(root < comm->commsize);

        if (i < total_steps - 1) {
            for (j = 0; j < req->concurreny_level; j++) {
                new_sched[i].prepost_buf_op[j].group_id = j + req->concurreny_level * (i % 2);
                new_sched[i].prepost_buf_op[j].offset   = offset * comm->max_per_packet;
                new_sched[i].prepost_buf_op[j].root     = root + j;
                new_sched[i].prepost_buf_op[j].count    = req->mcast_prepost_bucket_size;
            }
        } else {
            new_sched[i].prepost_buf_op_done = 1;
        }

        if (i > 0) {
            for (j = 0; j < req->concurreny_level; j++) {
                new_sched[i].multicast_op[j].group_id     = new_sched[i - 1].prepost_buf_op[j].group_id;
                new_sched[i].multicast_op[j].offset       = new_sched[i - 1].prepost_buf_op[j].offset;
                new_sched[i].multicast_op[j].offset_left  = new_sched[i - 1].prepost_buf_op[j].offset;
                new_sched[i].multicast_op[j].root         = new_sched[i - 1].prepost_buf_op[j].root;
                new_sched[i].multicast_op[j].to_send_left = new_sched[i - 1].prepost_buf_op[j].count;
                new_sched[i].multicast_op[j].to_recv      = new_sched[i - 1].prepost_buf_op[j].count;
                new_sched[i].to_recv                     += new_sched[i].multicast_op[j].to_recv;
                if (new_sched[i].multicast_op[j].root == comm->rank) {
                    new_sched[i].to_send += new_sched[i].multicast_op[j].to_send_left;
                }
            }
        }
        
        if (!new_sched[i].to_send || !new_sched[i].to_recv) {
            new_sched[i].multicast_op_done = 1;
        }

        offset += req->mcast_prepost_bucket_size;

        if (offset == req->num_packets) {
            offset = 0;
            root   = (root + req->concurreny_level) % comm->commsize;
        }
    }

    tl_trace(comm->lib,
             "generated the schedule for pipelined zero copy allgather with total_steps %d",
             total_steps);
    
    new_sched->total_steps = total_steps;
    req->total_steps       = total_steps;
    req->ag_schedule       = new_sched;

    tl_trace(comm->lib, "registering recv buf of size %ld", req->length * comm->commsize);

    ucc_assert(req->recv_rreg == NULL);

    status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->rptr, req->length *
                                            comm->commsize, &reg);
    if (UCC_OK != status) {
         tl_warn(comm->lib, "unable to register receive buffer %p of size %ld",
                  req->rptr, req->length * comm->commsize);
         ucc_free(new_sched);
         return status;
    }

    req->recv_rreg = reg;
    req->recv_mr   = reg->mr;

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_prepare_allgather(void* sbuf, void *rbuf, int size,
                                                               ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                               ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_tl_mlx5_mcast_reg_t *reg = NULL;
    ucc_status_t             status;

    req->comm   = comm;
    req->ptr    = sbuf;
    req->rptr   = rbuf;
    req->length = size;
    req->mr     = comm->pp_mr;
    req->rreg   = NULL;
    /* - zero copy protocol only provides zero copy design at sender side
     * - truly zero copy protocol provides zero copy design at receiver side as well
     * here we select the sender side protocol */
    req->proto  = (req->length < comm->max_eager) ? MCAST_PROTO_EAGER :
                                                    MCAST_PROTO_ZCOPY;

    if (comm->commsize > ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE) {
        tl_warn(comm->lib,
                "team size is %d but max supported team size of mcast allgather is %d",
                comm->commsize, ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE);
        return UCC_ERR_NOT_SUPPORTED;
    }

    req->offset      = 0;
    req->num_packets = (req->length + comm->max_per_packet - 1)/comm->max_per_packet;

    if (req->num_packets == 0) {
        req->num_packets = 1;
    }

    ONE_SIDED_MAX_PACKET_COUNT(comm->ag_max_num_packets);

    if (comm->ag_max_num_packets < req->num_packets) {
        tl_warn(comm->lib,
                "msg size is %ld but max supported msg size of mcast allgather is %d",
                req->length, comm->ag_max_num_packets * comm->max_per_packet);
        return UCC_ERR_NOT_SUPPORTED;
    }

    req->last_pkt_len = req->length - (req->num_packets - 1)*comm->max_per_packet;

    ucc_assert(req->last_pkt_len > 0 && req->last_pkt_len <= comm->max_per_packet);

    if (req->proto == MCAST_PROTO_ZCOPY) {
        /* register the send buffer */
       status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->ptr, req->length, &reg);
       if (UCC_OK != status) {
            return status;
       }
       req->rreg = reg;
       req->mr   = reg->mr;
    }

    if (comm->one_sided.reliability_enabled) {
        req->one_sided_reliability_scheme = (req->length <
                comm->one_sided.reliability_scheme_msg_threshold) ?
                ONE_SIDED_ASYNCHRONOUS_PROTO : ONE_SIDED_SYNCHRONOUS_PROTO;
        if (comm->truly_zero_copy_allgather_enabled) {
            req->one_sided_reliability_scheme = ONE_SIDED_SYNCHRONOUS_PROTO;
        }
    } else {
        req->one_sided_reliability_scheme = ONE_SIDED_NO_RELIABILITY;
    }

    req->ag_counter = comm->ag_counter;
    req->to_send    = req->num_packets;
    req->to_recv    = comm->commsize * req->num_packets;

    if (comm->truly_zero_copy_allgather_enabled) {
        status = ucc_tl_mlx5_mcast_prepare_zero_copy_allgather(comm, req);
        if (UCC_OK != status) {
            return status;
        }
    }

    comm->ag_counter++;
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_coll_do_allgather(void* sbuf, void *rbuf, int size,
                                                               ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                               ucc_tl_mlx5_mcast_coll_req_t **task_req_handle)
{
    ucc_tl_mlx5_mcast_coll_req_t *req;
    ucc_status_t                  status;

    tl_trace(comm->lib, "MCAST allgather start, sbuf %p, rbuf %p, size %d, comm %d, "
             "comm_size %d, counter %d truly zero copy pipelined %s\n",
             sbuf, rbuf, size, comm->comm_id, comm->commsize, comm->ag_counter,
             comm->truly_zero_copy_allgather_enabled?"enabled":"disabled");

    req = ucc_calloc(1, sizeof(ucc_tl_mlx5_mcast_coll_req_t), "mcast_req");
    if (!req) {
        tl_error(comm->lib, "malloc failed");
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_tl_mlx5_mcast_prepare_allgather(sbuf, rbuf, size, comm, req);
    if (UCC_OK != status) {
        tl_warn(comm->lib, "prepare mcast allgather failed");
        ucc_free(req);
        return status;
    }

    status = UCC_INPROGRESS;

    *task_req_handle = req;

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_allgather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t            *task      = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_team_t            *mlx5_team = TASK_TEAM(task);
    ucc_tl_mlx5_mcast_team_t      *team      = mlx5_team->mcast;
    ucc_coll_args_t               *args      = &TASK_ARGS(task);
    ucc_datatype_t                 dt        = args->src.info.datatype;
    size_t                         count     = args->src.info.count;
    ucc_status_t                   status    = UCC_OK;
    size_t                         data_size = ucc_dt_size(dt) * count;
    void                          *sbuf      = args->src.info.buffer;
    void                          *rbuf      = args->dst.info.buffer;
    ucc_tl_mlx5_mcast_coll_comm_t *comm      = team->mcast_comm;

    task->coll_mcast.req_handle = NULL;

    status = ucc_tl_mlx5_mcast_coll_do_allgather(sbuf, rbuf, data_size, comm, &task->coll_mcast.req_handle);
    if (status < 0) {
        tl_warn(UCC_TASK_LIB(task), "do mcast allgather failed:%d", status);
        coll_task->status = status;
        return ucc_task_complete(coll_task);
    }

    coll_task->status = status;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(mlx5_team)->pq, &task->super);
}

void ucc_tl_mlx5_mcast_allgather_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t           *task      = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);
    ucc_tl_mlx5_mcast_coll_req_t *req       = task->coll_mcast.req_handle;
    ucc_status_t                  status;

    if (task->coll_mcast.req_handle != NULL) {
        req = task->coll_mcast.req_handle;
        if (req->ag_counter != req->comm->ag_under_progress_counter) {
            /* it is not this task's turn for progress */
            ucc_assert(req->comm->ag_under_progress_counter < req->ag_counter);
            return;
        }

        status = ucc_tl_mlx5_mcast_test_allgather(task->coll_mcast.req_handle);
        if (UCC_INPROGRESS == status) {
            return;
        } else if (UCC_OK == status) {
            coll_task->status = UCC_OK;
            req->comm->ag_under_progress_counter++;
            ucc_free(req);
            task->coll_mcast.req_handle = NULL;
        } else {
            tl_error(UCC_TASK_LIB(task), "progress mcast allgather failed:%d", status);
            coll_task->status = status;
            ucc_task_complete(coll_task);
        }
    }

    return;
}

ucc_status_t ucc_tl_mlx5_mcast_allgather_init(ucc_tl_mlx5_task_t *task)
{
    task->super.post     = ucc_tl_mlx5_mcast_allgather_start;
    task->super.progress = ucc_tl_mlx5_mcast_allgather_progress;

    return UCC_OK;
}

