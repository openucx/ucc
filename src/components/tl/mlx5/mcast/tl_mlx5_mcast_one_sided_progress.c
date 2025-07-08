/**
 * Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_one_sided_progress.h"
#include "tl_mlx5_mcast_rcache.h"
#include "tl_mlx5_mcast_hca_copy.h"
#include <inttypes.h>

ucc_status_t
ucc_tl_mlx5_mcast_staging_allgather_reliable_one_sided_get(ucc_tl_mlx5_mcast_coll_comm_t *comm,
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

    /* When reliability protocol starts, copy scratch buffer to user buffer before RDMA reads.
     * This preserves received multicast packets while missing packets are retrieved via RDMA. */
    if (req->scratch_buf && req->scratch_packets_received > 0) {
        tl_trace(comm->lib,
                 "reliability protocol starting - copying scratch buffer (%d packets received) "
                 "to user buffer before RDMA reads", req->scratch_packets_received);

        /* Copy the entire scratch buffer to user buffer. Successfully received packets
         * will be preserved, and missing packets will be overwritten by RDMA reads. */
        status = ucc_tl_mlx5_mcast_memcpy(req->rptr, UCC_MEMORY_TYPE_CUDA,
                                          req->scratch_buf, UCC_MEMORY_TYPE_HOST,
                                          req->length * comm->commsize, comm);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(comm->lib,
                     "failed to copy scratch buffer to user buffer before reliability protocol");
            return status;
        }

        /* Mark scratch buffer as copied to avoid further operations on it */
        req->scratch_packets_received = -1;
        tl_trace(comm->lib,
                 "successfully copied scratch buffer to user buffer before reliability protocol");
    }

    for (target = 0; target < comm->commsize; target++) {
        if (comm->one_sided.recvd_pkts_tracker[target] == req->num_packets) {
            target_completed++;
            continue;
        }
        if (NULL == req->recv_rreg) {
            /* For reliability protocol, always register the user buffer to avoid memory access issues */
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
                    /* Always read to user buffer for reliability protocol */
                    src_addr = PTR_OFFSET(req->rptr, (req->length * target));
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
                /* Always read to user buffer for reliability protocol */
                src_addr = PTR_OFFSET(req->rptr, (req->length * target));
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
                 "all the mcast packets arrived during the reliability protocol "
                 "current timeout is %d usec",
                 comm->ctx->params.timeout);
    }

    if (ucc_tl_mlx5_mcast_poll_send(comm) < 0) {
        return UCC_ERR_NO_MESSAGE;
    }

    // check if all the rdma read have been completed and return UCC_OK if so
    switch(req->one_sided_reliability_scheme) {
        case ONE_SIDED_ASYNCHRONOUS_PROTO:
            /* only applicable for allgather collectives */
            status = ucc_tl_mlx5_mcast_staging_allgather_reliable_one_sided_get(comm, req, &completed);
            if (UCC_OK != status) {
                return status;
            }

            if (!comm->one_sided.pending_reads && (completed == comm->commsize)) {
                tl_trace(comm->lib,
                         "all the pending RDMA READ are comepleted in async reliability protocol");
                req->to_recv = 0;
                return UCC_OK;
            }
            break;

        case ONE_SIDED_SYNCHRONOUS_PROTO:
            if (!comm->one_sided.pending_reads) {
                tl_trace(comm->lib,
                         "all the pending RDMA READ are comepleted in sync reliability protocol");

                if (!comm->allgather_comm.truly_zero_copy_allgather_enabled &&
                    !comm->bcast_comm.truly_zero_copy_bcast_enabled) {
                    req->to_recv = 0;
                }
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
    int               out_of_order_recvd = 0;
    void             *dest;
    int               offset;
    int               source_rank;
    uint32_t          ag_counter;
    ucc_memory_type_t dst_mem_type;
    ucc_memory_type_t src_mem_type;
    ucc_status_t      status;

    ucc_assert(pp->context == 0); // making sure it's a recv packet not send

    // process the immediate value saved in pp->psn
    source_rank = pp->psn % ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE;
    ag_counter  = (pp->psn / ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE) % ONE_SIDED_MAX_ZCOPY_COLL_COUNTER;
    offset      = (pp->psn / (ONE_SIDED_MAX_ZCOPY_COLL_COUNTER * ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE));

    tl_trace(comm->lib, "processing a recvd packet with length %d source_rank"
             " %d ag_counter %d offset %d", pp->length, source_rank,
             ag_counter, offset);

    ucc_assert(offset < req->num_packets);
    // there are scenarios where we receive a packet with same offset/rank  more than one time
    // this means that a packet which was considered dropped in previous run has not just arrived
    // need to check the allgather call counter and ignore this packet if it does not match

    if (ag_counter == (req->ag_counter % ONE_SIDED_MAX_ZCOPY_COLL_COUNTER)) {
        /* Update reliability tracking FIRST before any data processing */
        if (comm->one_sided.reliability_enabled) {
            /* out of order recv'd packet that happen that is fatal in zero-copy
             * design is considered just like dropped packet */
            if (offset != pp->packet_counter) {
                out_of_order_recvd = 1;
            }

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
            if (comm->allgather_comm.truly_zero_copy_allgather_enabled ||
                    comm->bcast_comm.truly_zero_copy_bcast_enabled) {
                /* remove pp from posted_recv_bufs queue */
                ucc_list_del(&pp->super);
            }
        }

        if (comm->allgather_comm.truly_zero_copy_allgather_enabled ||
            comm->bcast_comm.truly_zero_copy_bcast_enabled) {
            /* no need for memcopy - packet must be delivered by hca into
             * the user buffer double check that ordering is correct */
            if (offset != pp->packet_counter) {
                /* recevied out of order packet */
                tl_trace(comm->lib, "recevied out of order: packet counter %d expected recv counter %d with pp %p",
                         offset, pp->packet_counter, pp);
            }
        } else if (pp->length) {
            /* staging based allgather */
            ucc_assert(coll_type == UCC_COLL_TYPE_ALLGATHER);

            /* Use scratch buffer optimization when available for CUDA memory.
             * Both reliability and non-reliability paths coordinate properly with scratch buffer. */
            if (req->scratch_buf && comm->cuda_mem_enabled) {
                /* CUDA staging with scratch buffer optimization */
                if (pp->length == comm->max_per_packet) {
                    dest = req->scratch_buf + (offset * pp->length + source_rank * req->length);
                } else {
                    dest = req->scratch_buf + ((req->length - pp->length) + source_rank * req->length);
                }

                /* Fast HOST-to-HOST memcpy to scratch buffer */
                memcpy(dest, (void*) pp->buf, pp->length);

                /* Only increment counter if we haven't already completed the copy */
                if (req->scratch_packets_received >= 0) {
                    req->scratch_packets_received++;
                }

                /* Check if all packets received - if so, copy scratch buffer to user buffer */
                if (req->scratch_packets_received == (req->comm->commsize * req->num_packets)) {
                    status = ucc_tl_mlx5_mcast_memcpy(req->rptr, UCC_MEMORY_TYPE_CUDA,
                                                      req->scratch_buf, UCC_MEMORY_TYPE_HOST,
                                                      req->length * req->comm->commsize, comm);
                    if (ucc_unlikely(status != UCC_OK)) {
                        tl_error(comm->lib, "failed to copy scratch buffer to user buffer");
                        return status;
                    }
                    req->scratch_packets_received = -1;
                    tl_trace(comm->lib, "all packets received - copied scratch buffer to user buffer");
                }
            } else {
                /* Staging logic fallback - used when scratch buffer is not available or CUDA is disabled */
                if (pp->length == comm->max_per_packet) {
                    dest = PTR_OFFSET(req->rptr, (offset * pp->length + source_rank * req->length));
                } else {
                    dest = PTR_OFFSET(req->rptr,
                                      ((req->length - pp->length) + source_rank * req->length));
                }

                dst_mem_type = comm->cuda_mem_enabled ? UCC_MEMORY_TYPE_CUDA : UCC_MEMORY_TYPE_HOST;
                src_mem_type = UCC_MEMORY_TYPE_HOST; // staging buffer is always HOST
                status       = ucc_tl_mlx5_mcast_memcpy(dest, dst_mem_type, (void*) pp->buf,
                                                         src_mem_type, pp->length, comm);
                if (ucc_unlikely(status != UCC_OK)) {
                    tl_error(comm->lib, "failed to copy buffer");
                    return status;
                }
            }
        }

        req->to_recv--;
        comm->psn++;
        pp->context = 0;
        ucc_list_add_tail(&comm->bpool, &pp->super);
        comm->one_sided.posted_recv[pp->qp_id].posted_recvs_count--;
    } else if (ag_counter > (req->ag_counter % ONE_SIDED_MAX_ZCOPY_COLL_COUNTER)) {
        /* received out of order allgather/bcast packet - add it to queue for future
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

/* QP drain and reattach functionality - tested and working properly */
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
        tl_error(comm->lib, "failed to move mcast qp to ERR, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_RESET;
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr, IBV_QP_STATE)) {
        tl_error(comm->lib, "failed to move mcast qp to RESET, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_INIT;
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr,
                      IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY)) {
        tl_error(comm->lib, "failed to move mcast qp to INIT, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_RTR;
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr, IBV_QP_STATE)) {
        tl_error(comm->lib, "failed to modify QP to RTR, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn   = DEF_PSN;
    if (ibv_modify_qp(comm->mcast.groups[qp_id].qp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN)) {
        tl_error(comm->lib, "failed to modify QP to RTS, errno %d", errno);
        return UCC_ERR_NO_RESOURCE;
    }

    if (ibv_attach_mcast(comm->mcast.groups[qp_id].qp, &comm->mcast.groups[qp_id].mgid,
                         comm->mcast.groups[qp_id].lid)) {
        tl_error(comm->lib, "failed to attach QP %d to the mcast group with mcast_lid %d, errno %d",
                 qp_id, comm->mcast.groups[qp_id].lid, errno);
        return UCC_ERR_NO_RESOURCE;
    }

    tl_trace(comm->lib, "drained recv queue of QP %d", qp_id);
    return UCC_OK;
}

/* RDMA read for pipelined zero-copy reliability protocol */
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
    int                                        issued           = 0;
    int                                        j;
    int                                        root;
    int                                        qp_id;

    ucc_assert(!comm->one_sided.pending_reads);
    for (j = 0; j < req->concurrency_level; j++) {
        root = sched[req->step].multicast_op[j].root;
        /* comm->one_sided.recvd_pkts_tracker[root] will not be incremented if there is out of
         order packet */
        if (comm->one_sided.recvd_pkts_tracker[root] == sched[req->step].multicast_op[j].to_recv) {
            target_completed++;
            continue;
        }
        qp_id  = sched[req->step].prepost_buf_op[j].group_id;
        status = ucc_tl_mlx5_mcast_drain_recv_wr(comm, comm->ctx, qp_id);
        if (UCC_OK != status) {
            tl_error(comm->lib, "unable to drain the posted recv wr on qp %d", qp_id);
            return status;
        }
        ucc_list_for_each_safe(pp, next, &comm->one_sided.posted_recv[qp_id].posted_recv_bufs,
                               super) {
            /* return the recv pp this list back to free pool */
            ucc_list_del(&pp->super);
            pp->context = 0;
            ucc_list_add_tail(&comm->bpool, &pp->super);
        }
        tl_trace(comm->lib, "RDMA READ for step %d total steps %d"
                 " posted_recvs_count %d recvd_pkts_tracker for root=%d is %d "
                 "to_recv for this step %d req->to_recv %d comm->pending_recv %d QP %d",
                 req->step, req->total_steps,
                 comm->one_sided.posted_recv[qp_id].posted_recvs_count,
                 root, comm->one_sided.recvd_pkts_tracker[root],
                 sched[req->step].multicast_op[j].to_recv,
                 req->to_recv, comm->pending_recv, qp_id);

        comm->pending_recv                                   -=
            comm->one_sided.posted_recv[qp_id].posted_recvs_count;
        req->to_recv                                         -=
            comm->one_sided.posted_recv[qp_id].posted_recvs_count;
        comm->one_sided.recvd_pkts_tracker[root]              =
            sched[req->step].multicast_op[j].to_recv;
        comm->one_sided.posted_recv[qp_id].posted_recvs_count = 0;
        ucc_assert(comm->pending_recv >= 0 && req->to_recv >= 0);

        /* issue RDMA Read to this root and read a piece of sendbuf
         * from related to this step*/
        if (comm->bcast_comm.truly_zero_copy_bcast_enabled) {
            src_addr = PTR_OFFSET(req->ptr, req->length * root +
                                  sched[req->step].multicast_op[j].offset);
        } else {
            ucc_assert(comm->allgather_comm.truly_zero_copy_allgather_enabled);
            src_addr = PTR_OFFSET(req->rptr, req->length * root +
                                  sched[req->step].multicast_op[j].offset);
        }
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
    }
    sched[req->step].num_recvd = sched[req->step].to_recv;
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
