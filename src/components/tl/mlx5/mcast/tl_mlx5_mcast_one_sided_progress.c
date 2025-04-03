/**
 * Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_one_sided_progress.h"
#include <inttypes.h>
#include "tl_mlx5_mcast_rcache.h"

ucc_status_t ucc_tl_mlx5_mcast_reliable_one_sided_get(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                      ucc_tl_mlx5_mcast_coll_req_t  *req,
                                                      int *completed)
{
    int                      target_completed = 0;
    int                      issued = 0;
    ucc_status_t             status;
    ucc_rank_t               target;
    ucc_tl_mlx5_mcast_reg_t *reg;
    void                    *src_addr;
    void                    *remote_addr;
    uint32_t                 rkey;
    uint32_t                 lkey;
    size_t                   size;
    uint64_t                 wr;

    /* in sync design this function is only called once */
    ucc_assert(!(ONE_SIDED_SYNCHRONOUS_PROTO == req->one_sided_reliability_scheme &&
                 comm->one_sided.pending_reads));
    for (target = 0; target < comm->commsize; target++) {
        if (comm->one_sided.recvd_pkts_tracker[target] == req->num_packets) {
            target_completed++;
            continue;
        }
        if (NULL == req->recv_rreg) {
            tl_debug(comm->lib, "registering recv buf of size %ld", comm->commsize * req->length);
            status = ucc_tl_mlx5_mcast_mem_register(comm->ctx, req->rptr, comm->commsize * req->length, &reg);
            if (UCC_OK != status) {
                 return status;
            }
            req->recv_rreg = reg;
            req->recv_mr   = reg->mr;
        }
        switch(req->one_sided_reliability_scheme) {
            case ONE_SIDED_ASYNCHRONOUS_PROTO:
                /* first check if the remote slot is valid in this design, if
                 * reliability protocol is kicked, allgather is completed once
                 * all the values in one_sided.recvd_pkts_tracker[] is set to
                 * req->num_packets and comm->pending_reads is set to 0 */
                if (req->ag_counter == comm->one_sided.remote_slot_info[target]) {
                    /* read remote data from remote slot
                     * the content of this data is copied from send buffer by remote
                     * process */
                    src_addr    = PTR_OFFSET(req->rptr, (req->length * target));
                    remote_addr = PTR_OFFSET(comm->one_sided.info[target].slot_mem.remote_addr,
                                             ((req->ag_counter %
                                              ONE_SIDED_SLOTS_COUNT) *
                                              comm->one_sided.slot_size +
                                              ONE_SIDED_SLOTS_INFO_SIZE));
                    lkey        = req->recv_mr->lkey;
                    rkey        = comm->one_sided.info[target].slot_mem.rkey;
                    size        = req->length;
                    wr          = MCAST_AG_RDMA_READ_WR;
                    comm->one_sided.pending_reads++;
                    target_completed++;
                    comm->one_sided.remote_slot_info[target]   = ONE_SIDED_PENDING_DATA;
                    comm->one_sided.recvd_pkts_tracker[target] = req->num_packets;
                } else if (ONE_SIDED_PENDING_INFO != comm->one_sided.remote_slot_info[target] &&
                           ONE_SIDED_PENDING_DATA != comm->one_sided.remote_slot_info[target]) {
                    /* remote slot is not valid yet. Need to do an rdma
                     * read to check the latest state */
                    src_addr    = &comm->one_sided.remote_slot_info[target];
                    remote_addr = PTR_OFFSET(comm->one_sided.info[target].slot_mem.remote_addr,
                                             ((req->ag_counter % ONE_SIDED_SLOTS_COUNT) *
                                             comm->one_sided.slot_size));
                    lkey        = comm->one_sided.remote_slot_info_mr->lkey;
                    rkey        = comm->one_sided.info[target].slot_mem.rkey;
                    size        = ONE_SIDED_SLOTS_INFO_SIZE;
                    wr          = MCAST_AG_RDMA_READ_INFO_WR;
                    comm->one_sided.pending_reads++;
                    comm->one_sided.remote_slot_info[target] = ONE_SIDED_PENDING_INFO;
                } else {
                    /* rdma read to remote info or data has already been issue but it
                     * has not been completed */
                    continue;
                }
                break;
            case ONE_SIDED_SYNCHRONOUS_PROTO:
                /* read the whole remote send buffer */
                src_addr    = PTR_OFFSET(req->rptr, (req->length * target));
                remote_addr = (void *)comm->one_sided.sendbuf_memkey_list[target].remote_addr;
                rkey        = comm->one_sided.sendbuf_memkey_list[target].rkey;
                lkey        = req->recv_mr->lkey;
                size        = req->length;
                wr          = MCAST_AG_RDMA_READ_WR;
                comm->one_sided.pending_reads++;
                target_completed++;
                break;
            default:
                return UCC_ERR_NOT_IMPLEMENTED;
        }
        issued++;
        status = ucc_tl_mlx5_one_sided_p2p_get(src_addr, remote_addr, size,
                                              lkey, rkey, target, wr, comm);
        if (UCC_OK != status) {
            return status;
        }
    }
    if (completed) {
        *completed = target_completed;
    }
    if (issued) {
        tl_debug(comm->lib,
                 "issued %d RDMA READ to remote INFO/DATA. Number of target ranks completed: %d",
                 issued, target_completed);
    }
    return UCC_OK;
}

ucc_status_t
ucc_tl_mlx5_mcast_progress_one_sided_communication(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                   ucc_tl_mlx5_mcast_coll_req_t *req)
{
    int          completed = 0;
    ucc_status_t status;
    
    ucc_assert(comm->one_sided.pending_reads);

    if (!req->to_send && !req->to_recv) {
        // need to wait until all the rdma reads are done to avoid data invalidation 
        tl_trace(comm->lib,
                "all the mcast packets arrived during the reliability protocol- current timeout is %d usec",
                comm->ctx->params.timeout);
    }

    if (ucc_tl_mlx5_mcast_poll_send(comm) < 0) {
        return UCC_ERR_NO_MESSAGE;
    }

    // check if all the rdma read have been completed and return UCC_OK if so
    switch(req->one_sided_reliability_scheme) {
        case ONE_SIDED_ASYNCHRONOUS_PROTO:
            status = ucc_tl_mlx5_mcast_reliable_one_sided_get(comm, req, &completed);
            if (UCC_OK != status) {
                return status;
            }

            if (!comm->one_sided.pending_reads && (completed == comm->commsize)) {
                tl_debug(comm->lib,
                         "all the pending RDMA READ are comepleted in async reliability protocol");
                req->to_recv = 0;
                return UCC_OK;
            }
            break;

        case ONE_SIDED_SYNCHRONOUS_PROTO:
            if (!comm->one_sided.pending_reads) {
                tl_debug(comm->lib,
                         "all the pending RDMA READ are comepleted in sync reliability protocol");
                req->to_recv = 0;
                return UCC_OK;
            } 
            break;

        default:
            return UCC_ERR_NOT_IMPLEMENTED;
    }

    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_mlx5_mcast_process_packet_collective(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                         ucc_tl_mlx5_mcast_coll_req_t  *req,
                                                         struct pp_packet  *pp,
                                                         int coll_type)
{
    int          out_of_order_recvd = 0;
    void        *dest;
    int          offset;
    int          source_rank;
    uint32_t     ag_counter;

    ucc_assert(pp->context == 0); // making sure it's a recv packet not send
    ucc_assert(UCC_COLL_TYPE_ALLGATHER == coll_type);

    // process the immediate value saved in pp->psn
    source_rank = pp->psn % ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE;
    ag_counter  = (pp->psn / ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE) % ONE_SIDED_MAX_ALLGATHER_COUNTER;
    offset      = (pp->psn / (ONE_SIDED_MAX_ALLGATHER_COUNTER * ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE));

    tl_trace(comm->lib, "processing a recvd packet with length %d source_rank"
             " %d ag_counter %d offset %d", pp->length, source_rank,
             ag_counter, offset);

    ucc_assert(offset < req->num_packets);
    // there are scenarios where we receive a packet with same offset/rank  more than one time
    // this means that a packet which was considered dropped in previous run has not just arrived
    // need to check the allgather call counter and ignore this packet if it does not match

    if (ag_counter == (req->ag_counter % ONE_SIDED_MAX_ALLGATHER_COUNTER)) {
        if (comm->allgather_comm.truly_zero_copy_allgather_enabled) {
            /* no need for memcopy - packet must be delivered by hca into
             * the user buffer double check that ordering is correct */
            if (offset != pp->packet_counter) {
                /* recevied out of order packet */
                tl_trace(comm->lib, "recevied out of order: packet counter %d expected recv counter %d",
                         offset, pp->packet_counter);
                out_of_order_recvd = 1;
            }
        } else if (pp->length) {
            /* staging based allgather */
            if (pp->length == comm->max_per_packet) {
                dest = PTR_OFFSET(req->rptr, (offset * pp->length + source_rank * req->length));
            } else {
                dest = PTR_OFFSET(req->rptr, ((req->length - pp->length) + source_rank * req->length));
            }
            memcpy(dest, (void*) pp->buf, pp->length);
        }

        if (comm->one_sided.reliability_enabled) {
            /* out of order recv'd packet that happen that is fatal in zero-copy
             * design is considered just like dropped packet */
            if (out_of_order_recvd == 0) {
                comm->one_sided.recvd_pkts_tracker[source_rank]++;
            }
            if (comm->one_sided.recvd_pkts_tracker[source_rank] > req->num_packets) {
                tl_error(comm->lib, "reliability failed: comm->one_sided.recvd_pkts_tracker[%d] %d"
                        " req->num_packets %d offset %d"
                        " comm->allgather_comm.under_progress_counter %d req->ag_counter"
                        " %d \n", source_rank, comm->one_sided.recvd_pkts_tracker[source_rank],
                        req->num_packets, offset,
                        comm->allgather_comm.under_progress_counter, req->ag_counter);
                return UCC_ERR_NO_MESSAGE;
            }
        }
        req->to_recv--;
        comm->psn++;
        pp->context = 0;
        ucc_list_add_tail(&comm->bpool, &pp->super);
        comm->pending_recv_per_qp[pp->qp_id]--;
    } else if (ag_counter > (req->ag_counter % ONE_SIDED_MAX_ALLGATHER_COUNTER)) {
        /* received out of order allgather packet - add it to queue for future
         * processing */
        ucc_list_add_tail(&comm->pending_q, &pp->super);
    } else {
        /* received a packet which was left from previous iterations
         * it is due to the fact that reliability protocol was initiated.
         * return the posted receive buffer back to the pool */
        ucc_assert(comm->one_sided.reliability_enabled);
        pp->context = 0;
        ucc_list_add_tail(&comm->bpool, &pp->super);
    }

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_drain_recv_wr(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                           ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                                           int qp_id)
{
    struct ibv_qp_attr attr = {
        .qp_state   = IBV_QPS_ERR,
        .pkey_index = ctx->pkey_index,
        .port_num   = ctx->ib_port,
        .qkey       = DEF_QKEY
    };

    /* set the qp state to ERR to flush the posted recv buffers */
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr, IBV_QP_STATE)) {
        tl_error(ctx->lib, "failed to move mcast qp to ERR, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_RESET;
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr, IBV_QP_STATE)) {
        tl_error(ctx->lib, "failed to move mcast qp to RESET, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_INIT;
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
        tl_error(ctx->lib, "failed to move mcast qp to INIT, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr, IBV_QP_STATE)) {
        tl_error(ctx->lib, "failed to modify QP to RTR, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn   = DEF_PSN;
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        tl_error(ctx->lib, "failed to modify QP to RTS, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }
    return UCC_OK;
}

/* TODO handle packet size < MTU */
ucc_status_t
ucc_tl_mlx5_mcast_reliable_zcopy_pipelined_one_sided_get(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                         ucc_tl_mlx5_mcast_coll_req_t *req,
                                                         int *completed)
{
    ucc_tl_mlx5_mcast_pipelined_ag_schedule_t *sched = req->ag_schedule;
    ucc_status_t                               status;
    struct pp_packet                          *pp;
    struct pp_packet                          *next;
    void                                      *src_addr;
    void                                      *remote_addr;
    uint32_t                                   rkey;
    uint32_t                                   lkey;
    size_t                                     size;
    uint64_t                                   wr;
    int                                        target_completed = 0;
    int                                        issued = 0, j, root;

    ucc_assert(!comm->one_sided.pending_reads);
    // cancel all the preposted recvs regarding this target before RDMA READ
    for (j = 0; j < comm->mcast_group_count; j++) {
        if (comm->pending_recv_per_qp[j] != 0) {
            status = ucc_tl_mlx5_mcast_drain_recv_wr(comm, comm->ctx, j);
            if (UCC_OK != status) {
                tl_error(comm->lib, "unable to drain the posted recv wr on qp %d", j);
                return status;
            }
            comm->pending_recv          -= comm->pending_recv_per_qp[j];
            req->to_recv                -= comm->pending_recv_per_qp[j];
            comm->pending_recv_per_qp[j] = 0;
            ucc_assert(comm->pending_recv >= 0 && req->to_recv >= 0);
        }
    }
    // return the recv pp back to free pool
    ucc_list_for_each_safe(pp, next, &comm->posted_q, super) {
        ucc_list_del(&pp->super);
        pp->context = 0;
        ucc_list_add_tail(&comm->bpool, &pp->super);
    }
    ucc_assert(!comm->pending_recv);
    for (j = 0; j < req->concurrency_level; j++) {
        root = sched[req->step].multicast_op[j].root;
        /* comm->one_sided.recvd_pkts_tracker[root] will not be incremented if there is out of
         order packet */
        if (comm->one_sided.recvd_pkts_tracker[root] == sched[req->step].multicast_op[j].to_recv) {
            target_completed++;
            continue;
        }
        /* issue RDMA Read to this root and read a piece of sendbuf
         * from related to this step*/
        src_addr    = PTR_OFFSET(req->rptr, req->length * root +
                                 sched[req->step].multicast_op[j].offset);
        remote_addr = PTR_OFFSET(comm->one_sided.sendbuf_memkey_list[root].remote_addr,
                                 sched[req->step].multicast_op[j].offset);
        rkey        = comm->one_sided.sendbuf_memkey_list[root].rkey;
        lkey        = req->recv_mr->lkey;
        size        = sched[req->step].multicast_op[j].to_recv * comm->max_per_packet;
        wr          = MCAST_AG_RDMA_READ_WR;
        ucc_assert(size != 0);
        comm->one_sided.pending_reads++;
        issued++;
        status = ucc_tl_mlx5_one_sided_p2p_get(src_addr, remote_addr, size,
                                               lkey, rkey, root, wr, comm);
        if (UCC_OK != status) {
             return status;
        }
        tl_trace(comm->lib, "RDMA READ for step %d total steps %d",
                 req->step, req->total_steps);
    }
    if (completed) {
        *completed = target_completed;
    }
    if (issued) {
        tl_debug(comm->lib, "issued %d RDMA READ to remote DATA for step %d. Number of"
                 " target ranks completed: %d",
                 req->step, issued, target_completed);
    }
    return UCC_OK;
}
