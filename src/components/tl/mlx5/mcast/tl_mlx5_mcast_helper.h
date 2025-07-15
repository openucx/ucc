/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef TL_MLX5_MCAST_HELPER_H_
#define TL_MLX5_MCAST_HELPER_H_
#include "tl_mlx5_mcast_progress.h"
#include "tl_mlx5_mcast_one_sided_progress.h"
#include "utils/ucc_math.h"
#include "tl_mlx5.h"
#include "tl_mlx5_mcast_hca_copy.h"

static inline ucc_status_t ucc_tl_mlx5_mcast_poll_send(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    int               num_comp;
    struct pp_packet *pp;
    struct ibv_wc     wc[POLL_PACKED];

    num_comp = ibv_poll_cq(comm->mcast.scq, POLL_PACKED, &wc[0]);
    if (num_comp < 0) {
        tl_error(comm->lib, "send queue poll completion failed %d", num_comp);
        return UCC_ERR_NO_MESSAGE;
    } else if (num_comp > 0) {
        tl_trace(comm->lib, "polled send completions: %d", num_comp);
        for (int i = 0 ; i < num_comp ; i++) {
            if (IBV_WC_SUCCESS != wc[i].status) {
                tl_warn(comm->lib, "mcast_poll_send: %s err %d num_comp %d op %ld wr_id\n",
                        ibv_wc_status_str(wc[i].status), num_comp, wc[i].opcode, wc[i].wr_id);
                return UCC_ERR_NO_MESSAGE;
            }
            switch (wc[i].wr_id) {
                case MCAST_AG_RDMA_READ_WR:
                    /* completion of a RDMA Read to remote send buffer during
                     * reliability protocol */
                    comm->one_sided.pending_reads--;
                    tl_trace(comm->lib, "RDMA READ completion, pending reads %d",
                             comm->one_sided.pending_reads);
                    break;
                case MCAST_AG_RDMA_READ_INFO_WR:
                    tl_trace(comm->lib, "RDMA READ remote slot info completion, pending reads %d",
                             comm->one_sided.pending_reads);
                    comm->one_sided.pending_reads--;
                    break;
                case MCAST_BCASTSEND_WR:
                    tl_trace(comm->lib, "completion of mcast send for bcast, opcode %d",
                             wc[i].opcode);
                    comm->pending_send--;
                    break;
                default:
                    tl_trace(comm->lib, "completion of mcast send for zero-copy collective, opcode %d",
                             wc[i].opcode);
                    comm->pending_send--;
                    pp = (struct pp_packet*)wc[i].wr_id;
                    assert(pp != 0);
                    pp->context = 0;
                    ucc_list_add_tail(&comm->bpool, &pp->super);
                    break;
            }
        }
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_send(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                  ucc_tl_mlx5_mcast_coll_req_t *req,
                                                  int num_packets, const int zcopy)
{
    struct ibv_send_wr *swr            = &comm->mcast.swr;
    struct ibv_sge     *ssg            = &comm->mcast.ssg;
    int                 max_per_packet = comm->max_per_packet;
    int                 offset = req->offset;
    int                 i;
    struct ibv_send_wr *bad_wr;
    struct pp_packet   *pp;
    int                 rc;
    int                 length;
    ucc_status_t        status;
    ucc_memory_type_t   mem_type = comm->cuda_mem_enabled ? UCC_MEMORY_TYPE_CUDA
                                                          : UCC_MEMORY_TYPE_HOST;

    for (i = 0; i < num_packets; i++) {
        if (comm->params.sx_depth <=
               (comm->pending_send * comm->params.scq_moderation + comm->tx)) {
            status = ucc_tl_mlx5_mcast_poll_send(comm);
            if (UCC_OK != status) {
                return status;
            }
            break;
        }

        if (NULL == (pp = ucc_tl_mlx5_mcast_buf_get_free(comm))) {
            break;
        }

        ucc_assert(pp->context == 0);

        __builtin_prefetch((void*) pp->buf);
        __builtin_prefetch(PTR_OFFSET(req->ptr, offset));

        length      = req->to_send == 1 ? req->last_pkt_len : max_per_packet;
        pp->length  = length;
        pp->psn     = comm->psn;
        ssg[0].addr = (uintptr_t) PTR_OFFSET(req->ptr, offset);

        if (zcopy) {
            pp->context = (uintptr_t) PTR_OFFSET(req->ptr, offset);
        } else {
            status = ucc_mc_memcpy((void*) pp->buf, PTR_OFFSET(req->ptr, offset), length,
                                   mem_type, mem_type);
            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(comm->lib, "failed to copy cuda buffer");
                return status;
            }
            ssg[0].addr = (uint64_t) pp->buf;
        }

        ssg[0].length     = length;
        ssg[0].lkey       = zcopy ? req->mr->lkey : comm->pp_mr->lkey;
        swr[0].wr.ud.ah   = comm->mcast.groups[0].ah;
        swr[0].wr_id      = MCAST_BCASTSEND_WR;
        swr[0].imm_data   = htonl(pp->psn);
        swr[0].send_flags = (length <= comm->max_inline) ? IBV_SEND_INLINE : 0;

        comm->r_window[pp->psn & (comm->bcast_comm.wsize-1)] = pp;
        comm->psn++;
        req->to_send--;
        offset += length;
        comm->tx++;

        if (comm->tx == comm->params.scq_moderation) {
            swr[0].send_flags |= IBV_SEND_SIGNALED;
            comm->tx           = 0;
            comm->pending_send++;
        }

        tl_trace(comm->lib, "post_send, psn %d, length %d, zcopy %d, signaled %d",
                 pp->psn, pp->length, zcopy, swr[0].send_flags & IBV_SEND_SIGNALED);

        if (0 != (rc = ibv_post_send(comm->mcast.groups[0].qp, &swr[0], &bad_wr))) {
            tl_error(comm->lib, "post send failed: ret %d, start_psn %d, to_send %d, "
                    "to_recv %d, length %d, psn %d, inline %d",
                     rc, req->start_psn, req->to_send, req->to_recv,
                     length, pp->psn, length <= comm->max_inline);
            return UCC_ERR_NO_MESSAGE;
        }

        status = ucc_tl_mlx5_mcast_check_nack_requests(comm, pp->psn);
        if (UCC_OK != status) {
            return status;
        }
    }

    req->offset = offset;

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_process_pp(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                        ucc_tl_mlx5_mcast_coll_req_t *req,
                                                        struct pp_packet *pp,
                                                        int *num_left, int in_pending_queue)
{
    ucc_status_t status = UCC_OK;

    if (PSN_RECEIVED(pp->psn, comm) || pp->psn < comm->bcast_comm.last_acked) {
        /* This psn was already received */
        ucc_assert(pp->context == 0);
        if (in_pending_queue) {
            /* this pp belongs to pending queue so remove it */
            ucc_list_del(&pp->super);
        }
        /* add pp to the free pool */
        ucc_list_add_tail(&comm->bpool, &pp->super);
    } else if (PSN_IS_IN_RANGE(pp->psn, req, comm)) {
        if (*num_left <= 0 && !in_pending_queue) {
            /* we just received this packet and it is in order, but there is no
             * more space in window so we need to place this packet in the
             * pending queue for future processings */
            ucc_list_add_tail(&comm->pending_q, &pp->super);
        } else {
            __builtin_prefetch(PTR_OFFSET(req->ptr, PSN_TO_RECV_OFFSET(pp->psn, req, comm)));
            __builtin_prefetch((void*) pp->buf);
            if (in_pending_queue) {
                ucc_list_del(&pp->super);
            }
            status = ucc_tl_mlx5_mcast_process_packet(comm, req, pp);
            if (UCC_OK != status) {
                return status;
            }
            (*num_left)--;
        }
    } else if (!in_pending_queue) {
        /* add pp to the pending queue as it is out of order */
        ucc_list_add_tail(&comm->pending_q, &pp->super);
    }

    return status;
}

/* this function return the number of mcast recv packets that
 * are left or -1 in case of error */
static inline int ucc_tl_mlx5_mcast_recv(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                         ucc_tl_mlx5_mcast_coll_req_t *req,
                                         int num_left, int *pending_q_size)
{
    struct pp_packet *pp;
    struct pp_packet *next;
    uint64_t          id;
    struct ibv_wc    *wc;
    int               num_comp;
    int               i;
    int               real_num_comp;
    ucc_status_t      status;

    /* check if we have already received something */
    ucc_list_for_each_safe(pp, next, &comm->pending_q, super) {
        status = ucc_tl_mlx5_mcast_process_pp(comm, req, pp, &num_left, 1);
        if (UCC_OK != status) {
            return -1;
        }
        (*pending_q_size)++;
    }

    wc = ucc_malloc(sizeof(struct ibv_wc) * POLL_PACKED, "WC");
    if (!wc) {
        tl_error(comm->lib, "ucc_malloc failed");
        return -1;
    }

    while (num_left > 0)
    {
        memset(wc, 0, sizeof(struct ibv_wc) * POLL_PACKED);
        num_comp = ibv_poll_cq(comm->mcast.rcq, POLL_PACKED, wc);

        if (num_comp < 0) {
            tl_error(comm->lib, "recv queue poll completion failed %d", num_comp);
            ucc_free(wc);
            return -1;
        } else if (num_comp == 0) {
            break;
        }

        real_num_comp = num_comp;

        for (i = 0; i < real_num_comp; i++) {
            if (IBV_WC_SUCCESS != wc[i].status) {
                tl_error(comm->lib, "mcast_recv: %s err pending_recv %d wr_id %ld"
                         " num_comp %d byte_len %d",
                         ibv_wc_status_str(wc[i].status), comm->pending_recv,
                         wc[i].wr_id, num_comp, wc[i].byte_len);
                ucc_free(wc);
                return -1;
            }

            id         = wc[i].wr_id;
            pp         = (struct pp_packet*) (id);
            pp->length = wc[i].byte_len - GRH_LENGTH;
            pp->psn    = ntohl(wc[i].imm_data);

            tl_trace(comm->lib, "completion: psn %d, length %d, already_received %d, "
                                " psn in req %d, req_start %d, req_num packets"
                                " %d, to_send %d, to_recv %d, num_left %d",
                                pp->psn, pp->length, PSN_RECEIVED(pp->psn,
                                comm) > 0, PSN_IS_IN_RANGE(pp->psn, req,
                                comm), req->start_psn, req->num_packets,
                                req->to_send, req->to_recv, num_left);

            status = ucc_tl_mlx5_mcast_process_pp(comm, req, pp, &num_left, 0);
            if (UCC_OK != status) {
                return -1;
            }
        }

        comm->pending_recv -= num_comp;
        status = ucc_tl_mlx5_mcast_post_recv_buffers(comm);
        if (UCC_OK != status) {
            return -1;
        }
    }

    ucc_free(wc);
    return num_left;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_send_collective(ucc_tl_mlx5_mcast_coll_comm_t*
                                                             comm, ucc_tl_mlx5_mcast_coll_req_t *req,
                                                             int num_packets, const int zcopy,
                                                             int mcast_group_index,
                                                             size_t send_offset)
{
    struct ibv_send_wr *swr            = &comm->mcast.swr;
    struct ibv_sge     *ssg            = &comm->mcast.ssg;
    size_t              offset         = (send_offset == SIZE_MAX) ? req->offset : send_offset;
    int                 max_commsize   = ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE;
    int                 max_ag_counter = ONE_SIDED_MAX_ZCOPY_COLL_COUNTER;
    int                 i;
    struct ibv_send_wr *bad_wr;
    struct pp_packet   *pp;
    int                 rc;
    int                 length;
    ucc_status_t        status;
    int                 use_zcopy;
    ucc_memory_type_t   src_mem_type;
    ucc_memory_type_t   dst_mem_type;

    ucc_assert(mcast_group_index <= comm->mcast_group_count);

    swr->num_sge           = 1;
    swr->sg_list           = & comm->mcast.ssg;
    swr->opcode            = IBV_WR_SEND_WITH_IMM;
    swr->wr.ud.remote_qpn  = MULTICAST_QPN;
    swr->wr.ud.remote_qkey = DEF_QKEY;
    swr->next              = NULL;

    for (i = 0; i < num_packets; i++) {
        if (NULL == (pp = ucc_tl_mlx5_mcast_buf_get_free(comm))) {
            break;
        }
        ucc_assert(pp->context == 0);

        __builtin_prefetch((void*) pp->buf);
        __builtin_prefetch(req->ptr + offset);

        length = (req->to_send == 1) ? (req->length - offset) : comm->max_per_packet;
        pp->length  = length;

        // generate psn to be used as immediate data
        /* example: encapsulate packet counter (top 16 bits), collective counter (middle 8 bits),
         * and source rank (low 8 bits) - assuming max_commsize and
         * max_ag_counter are 256 */
        pp->psn = (req->num_packets - req->to_send)*max_commsize*max_ag_counter
                   + (req->ag_counter % max_ag_counter)*max_commsize + comm->rank;

        ssg[0].addr = (uintptr_t)req->ptr + offset;

        /* Enable zero-copy if we have registered CUDA memory, even with staging protocol */
        use_zcopy = zcopy || (comm->cuda_mem_enabled && req->mr && req->mr != comm->pp_mr);

        if (use_zcopy && comm->cuda_mem_enabled && req->mr != comm->pp_mr) {
            tl_trace(comm->lib, "using zero-copy sending for CUDA memory, length %d", length);
        }

        if (!use_zcopy) {
            src_mem_type = comm->cuda_mem_enabled ? UCC_MEMORY_TYPE_CUDA : UCC_MEMORY_TYPE_HOST;
            dst_mem_type = UCC_MEMORY_TYPE_HOST; // staging buffer is always HOST
            status       = ucc_tl_mlx5_mcast_memcpy((void*) pp->buf, dst_mem_type,
                                                     req->ptr + offset, src_mem_type, length, comm);
            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(comm->lib, "failed to copy buffer to staging area");
                return status;
            }
            ssg[0].addr = (uint64_t) pp->buf;
            ssg[0].lkey = comm->pp_mr->lkey;
        } else {
            pp->context = (uintptr_t)req->ptr + offset;
            ssg[0].lkey = req->mr->lkey;
        }

        ssg[0].length     = length;
        swr[0].wr_id      = (uint64_t) pp;
        swr[0].imm_data   = htonl(pp->psn);
        swr[0].send_flags = (length <= comm->max_inline) ? IBV_SEND_INLINE : 0;

        comm->psn    ++;
        req->to_send --;
        offset += length;

        swr[0].send_flags |= IBV_SEND_SIGNALED;
        comm->pending_send++;

        swr[0].wr.ud.ah = comm->mcast.groups[mcast_group_index].ah;

        tl_trace(comm->lib,
                 "mcast  post_send, psn %d, length %d, zcopy %d, use_zcopy %d, signaled %d "
                 "qp->state %d qp->qp_num %d qp->pd %p mcast_group_index %d",
                 pp->psn, pp->length, zcopy, use_zcopy, swr[0].send_flags & IBV_SEND_SIGNALED,
                 comm->mcast.groups[mcast_group_index].qp->state,
                 comm->mcast.groups[mcast_group_index].qp->qp_num,
                 comm->mcast.groups[mcast_group_index].qp->pd, mcast_group_index);

        if (0 != (rc = ibv_post_send(comm->mcast.groups[mcast_group_index].qp, &swr[0], &bad_wr))) {
            tl_error(comm->lib, "post send failed: ret %d, start_psn %d, to_send %d, "
                      "to_recv %d, length %d, psn %d, inline %d",
                      rc, req->start_psn, req->to_send, req->to_recv,
                      length, pp->psn, length <= comm->max_inline);
            return UCC_ERR_NO_MESSAGE;
        }
    }

    if (send_offset == SIZE_MAX) {
        req->offset = offset;
    }

    return UCC_OK;
}

static inline int ucc_tl_mlx5_mcast_recv_collective(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                    ucc_tl_mlx5_mcast_coll_req_t *req, int
                                                    num_left, int coll_type)
{
    int               max_commsize   = ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE;
    int               max_ag_counter = ONE_SIDED_MAX_ZCOPY_COLL_COUNTER;
    struct pp_packet *pp;
    struct pp_packet *next;
    uint64_t          id;
    struct ibv_wc    *wc;
    int               num_comp;
    int               i;
    int               real_num_comp;
    int               recv_progressed = 0;
    int               ag_counter;
    ucc_status_t      status;

    /* check if we have already received something */
    ucc_list_for_each_safe(pp, next, &comm->pending_q, super) {
        ag_counter = (pp->psn / max_commsize) %
                      max_ag_counter;
        if (ag_counter == (req->ag_counter % max_ag_counter)) {
            ucc_list_del(&pp->super);
            status = ucc_tl_mlx5_mcast_process_packet_collective(comm, req, pp, coll_type);
            if (UCC_OK != status) {
                tl_error(comm->lib, "process mcast packet failed, status %d",
                         status);
                return -1;
            }
            recv_progressed++;
        }
    };

    wc = ucc_malloc(sizeof(struct ibv_wc) * POLL_PACKED, "WC");
    if (!wc) {
        recv_progressed = -1;
        goto exit;
    }

    while (num_left >  recv_progressed)
    {
        memset(wc, 0, sizeof(struct ibv_wc) * POLL_PACKED);
        num_comp = ibv_poll_cq(comm->mcast.rcq, POLL_PACKED, &wc[0]);

        if (num_comp < 0) {
            tl_error(comm->lib, "recv queue poll completion failed %d", num_comp);
            recv_progressed = -1;
            goto exit;
        } else if (num_comp == 0) {
            break;
        }

        if (IBV_WC_SUCCESS != wc[0].status) {
            tl_error(comm->lib, "mcast_recv: %s err pending_recv %d wr_id %ld num_comp %d byte_len %d\n",
                     ibv_wc_status_str(wc[0].status), comm->pending_recv, wc[0].wr_id, num_comp, wc[0].byte_len);
            recv_progressed = -1;
            goto exit;
        }

        real_num_comp = num_comp;

        for (i = 0; i < real_num_comp; i++) {
            ucc_assert(wc[i].status == IBV_WC_SUCCESS);
            id         = wc[i].wr_id;
            pp         = (struct pp_packet*) (id);
            pp->length = wc[i].byte_len - GRH_LENGTH;
            pp->psn    = ntohl(wc[i].imm_data);

            tl_trace(comm->lib, "%d collective pkt completion: psn %d, length %d, "
                                "req_num packets %d, to_send %d, to_recv %d, num_left %d \n",
                                 coll_type, pp->psn, pp->length, req->num_packets,
                                 req->to_send, req->to_recv, num_left);

            status = ucc_tl_mlx5_mcast_process_packet_collective(comm, req, pp, coll_type);
            if (UCC_OK != status) {
                tl_error(comm->lib, "process mcast packet failed, status %d",
                         status);
                recv_progressed = -1;
                goto exit;
            }

            recv_progressed++;
            ucc_assert(pp->qp_id < MAX_GROUP_COUNT);
        }

        comm->pending_recv -= num_comp;
        status = ucc_tl_mlx5_mcast_post_recv_buffers(comm);
        if (UCC_OK != status) {
            recv_progressed = -1;
            goto exit;
        }
    }

exit:
    ucc_free(wc);
    return recv_progressed;
}


static inline ucc_status_t ucc_tl_mlx5_mcast_poll_recv(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_status_t      status = UCC_OK;
    struct pp_packet *pp;
    struct ibv_wc     wc;
    int               num_comp;
    uint64_t          id;
    int               length;
    uint32_t          psn;

    do {
        num_comp = ibv_poll_cq(comm->mcast.rcq, 1, &wc);

        if (num_comp > 0) {
            if (IBV_WC_SUCCESS != wc.status) {
                tl_error(comm->lib, "mcast_poll_recv: %s err %d num_comp",
                        ibv_wc_status_str(wc.status), num_comp);
                status = UCC_ERR_NO_MESSAGE;
                return status;
            }

            // Make sure we received all in order.
            id     = wc.wr_id;
            length = wc.byte_len - GRH_LENGTH;
            psn    = ntohl(wc.imm_data);
            pp     = (struct pp_packet*) id;

            if (psn >= comm->psn) {
                ucc_assert(!PSN_RECEIVED(psn, comm));
                pp->psn    = psn;
                pp->length = length;
                ucc_list_add_tail(&comm->pending_q, &pp->super);
            } else {
                ucc_assert(pp->context == 0);
                ucc_list_add_tail(&comm->bpool, &pp->super);
            }

            comm->pending_recv--;
            status = ucc_tl_mlx5_mcast_post_recv_buffers(comm);
            if (UCC_OK != status) {
                return status;
            }
        } else if (num_comp != 0) {
            tl_error(comm->lib, "mcast_poll_recv: %d num_comp", num_comp);
            status = UCC_ERR_NO_MESSAGE;
            return status;
        }
    } while (num_comp);

    return status;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_reliable(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_status_t status = UCC_OK;

    if (comm->bcast_comm.racks_n != comm->bcast_comm.child_n ||
            comm->bcast_comm.sacks_n != comm->bcast_comm.parent_n ||
            comm->bcast_comm.nack_requests) {
        if (comm->pending_send) {
            status = ucc_tl_mlx5_mcast_poll_send(comm);
            if (UCC_OK != status) {
                return status;
            }
        }

        if (comm->bcast_comm.parent_n) {
            status = ucc_tl_mlx5_mcast_poll_recv(comm);
            if (UCC_OK != status) {
                return status;
            }
        }

        status = ucc_tl_mlx5_mcast_check_nack_requests(comm, UINT32_MAX);
        if (UCC_OK != status) {
            return status;
        }
    }

    if (comm->bcast_comm.parent_n && !comm->bcast_comm.reliable_in_progress) {
        status = ucc_tl_mlx5_mcast_reliable_send(comm);
        if (UCC_OK != status) {
            return status;
        }
    }

    if (!comm->bcast_comm.reliable_in_progress) {
        comm->bcast_comm.reliable_in_progress = 1;
    }

    if (comm->bcast_comm.racks_n == comm->bcast_comm.child_n &&
            comm->bcast_comm.sacks_n == comm->bcast_comm.parent_n && 0 ==
            comm->bcast_comm.nack_requests) {
        // Reset for next round.
        memset(comm->bcast_comm.parents,  0, sizeof(comm->bcast_comm.parents));
        memset(comm->bcast_comm.children, 0, sizeof(comm->bcast_comm.children));

        comm->bcast_comm.racks_n              = comm->bcast_comm.child_n  = 0;
        comm->bcast_comm.sacks_n              = comm->bcast_comm.parent_n = 0;
        comm->bcast_comm.reliable_in_progress = 0;

        return UCC_OK;
    }

    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_mlx5_probe_ip_over_ib(char* ib_dev_list,
                                          struct sockaddr_storage *addr);

ucc_status_t ucc_tl_mlx5_setup_mcast(ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_init_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                        ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_setup_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                         ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_create_rc_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                             ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_modify_rc_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                             ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_clean_mcast_comm(ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_join_mcast_post(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                               struct sockaddr_in6 *net_addr,
                                               struct mcast_group *group,
                                               int is_root);

ucc_status_t ucc_tl_mlx5_mcast_join_mcast_get_event(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                                    struct rdma_cm_event **event);

ucc_status_t ucc_tl_mlx5_leave_mcast_groups(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                            ucc_tl_mlx5_mcast_coll_comm_t    *comm);

#endif /* TL_MLX5_MCAST_HELPER_H_ */
