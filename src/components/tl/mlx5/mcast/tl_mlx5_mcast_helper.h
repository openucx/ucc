/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <infiniband/verbs.h>
#include <glob.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <ifaddrs.h>
#include <sys/socket.h>
#include "ucc/api/ucc_status.h"
#include "tl_mlx5.h"

ucc_status_t ucc_tl_probe_ip_over_ib(const char* ib_dev_list, struct sockaddr_storage *addr);

#ifndef MCAST_H_
#define MCAST_H_
#include "tl_mlx5_mcast_progress.h"

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

static inline int ucc_tl_mlx5_mcast_poll_send(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct ibv_wc wc;
    int           num_comp;
    
    num_comp = ibv_poll_cq(comm->scq, 1, &wc);
    
    tl_trace(comm->lib, "Polled send completions: %d", num_comp);
    
    if (num_comp < 0) {

        tl_error(comm->lib, "send queue poll completion failed %d", num_comp);

    } else if (num_comp > 0) {

        if (IBV_WC_SUCCESS != wc.status) {
           tl_warn(comm->lib, "mcast_poll_send: %s err %d num_comp\n",
                    ibv_wc_status_str(wc.status), num_comp);
            return -1;
        }

        comm->pending_send -= num_comp;
    }
    
    return num_comp;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_send(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                  ucc_tl_mlx5_mcast_coll_req_t *req,
                                                  int num_packets, const int zcopy)
{
    struct ibv_send_wr *swr            = &comm->mcast.swr;
    struct ibv_sge     *ssg            = &comm->mcast.ssg;
    int                 max_per_packet = comm->max_per_packet;
    int                 offset         = req->offset, i;
    struct ibv_send_wr *bad_wr;
    struct pp_packet   *pp;
    int                 rc;
    int                 length;

    for (i = 0; i < num_packets; i++) {
        pp = ucc_tl_mlx5_mcast_buf_get_free(comm);
        assert(pp->context == 0);

        __builtin_prefetch((void*) pp->buf);
        __builtin_prefetch(req->ptr + offset);

        length      = req->to_send == 1 ? req->last_pkt_len : max_per_packet;
        pp->length  = length;
        pp->psn     = comm->psn;
        ssg[0].addr = (uintptr_t)req->ptr + offset;

        if (!zcopy) {
            memcpy((void*) pp->buf, req->ptr + offset, length);
            ssg[0].addr = (uint64_t) pp->buf;
        } else {
            pp->context = (uintptr_t)req->ptr + offset;
        }

        ssg[0].length     = length;
        ssg[0].lkey       = req->mr->lkey;
        swr[0].wr_id      = MCAST_BCASTSEND_WR;
        swr[0].imm_data   = htonl(pp->psn);
        swr[0].send_flags = (length <= comm->max_inline) ? IBV_SEND_INLINE : 0;

        comm->r_window[pp->psn & (comm->wsize-1)] = pp;
        comm->psn    ++;
        req->to_send --;
        offset       += length;
        comm->tx     ++;

        if (comm->tx == comm->params.scq_moderation) {
            swr[0].send_flags |= IBV_SEND_SIGNALED;
            comm->tx           = 0;
            comm->pending_send++;
        }

        while (comm->params.sx_depth <=
               (comm->pending_send * comm->params.scq_moderation + comm->tx)) {
            if (ucc_tl_mlx5_mcast_poll_send(comm) < 0) {
                return UCC_ERR_NO_MESSAGE;
            }
        }

        tl_trace(comm->lib, "post_send, psn %d, length %d, zcopy %d, signaled %d",
                             pp->psn, pp->length, zcopy, swr[0].send_flags & IBV_SEND_SIGNALED);

        if (0 != (rc = ibv_post_send(comm->mcast.qp, &swr[0], &bad_wr))) {
            tl_error(comm->lib, "Post send failed: ret %d, start_psn %d, to_send %d, "
                      "to_recv %d, length %d, psn %d, inline %d",
                      rc, req->start_psn, req->to_send, req->to_recv,
                      length, pp->psn, length <= comm->max_inline);
            return UCC_ERR_NO_MESSAGE;
        }

        if (UCC_OK != ucc_tl_mlx5_mcast_check_nack_requests(comm, pp->psn)) {
            return UCC_ERR_NO_MESSAGE;
        }
    }

    req->offset = offset;

    return UCC_OK;
}

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

    /* check if we have already received something */
    ucc_list_for_each_safe(pp, next, &comm->pending_q, super) {
        if (PSN_IS_IN_RANGE(pp->psn, req, comm)) {
            __builtin_prefetch(req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm));
            __builtin_prefetch((void*) pp->buf);
            ucc_list_del(&pp->super);
            ucc_tl_mlx5_mcast_process_packet(comm, req, pp);
            num_left --;
        } else if (pp->psn < comm->last_acked){
            ucc_list_del(&pp->super);
            ucc_list_add_tail(&comm->bpool, &pp->super);
        }

        (*pending_q_size)++;
    };

    wc = ucc_malloc(sizeof(struct ibv_wc) * POLL_PACKED, "WC");
    if (!wc) {
        return -1;
    }

    while (num_left > 0) 
    {
        memset(wc, 0, sizeof(sizeof(struct ibv_wc) * POLL_PACKED));
        num_comp = ibv_poll_cq(comm->rcq, POLL_PACKED, &wc[0]);

        if (num_comp < 0) {
            tl_error(comm->lib, "recv queue poll completion failed %d", num_comp);
            ucc_free(wc);
            return -1;
        } else if (num_comp == 0) {
            break;
        }

        if (IBV_WC_SUCCESS != wc[0].status) {
            tl_error(comm->lib, "mcast_recv: %s err pending_recv %d wr_id %ld num_comp %d byte_len %d\n",
                    ibv_wc_status_str(wc[0].status), comm->pending_recv, wc[0].wr_id, num_comp, wc[0].byte_len);
            return -1;
        }

        real_num_comp = num_comp;

        for (i = 0; i < real_num_comp; i++) {
            
            assert(wc[i].status == IBV_WC_SUCCESS);

            id         = wc[i].wr_id;
            pp         = (struct pp_packet*) (id);
            pp->length = wc[i].byte_len - GRH_LENGTH;
            pp->psn    = ntohl(wc[i].imm_data);

            tl_trace(comm->lib, "completion: psn %d, length %d, already_received %d, "
                                "psn in req %d, req_start %d, req_num packets %d, to_send %d, to_recv %d, num_left %d \n", 
                                 pp->psn, pp->length, PSN_RECEIVED(pp->psn, comm) > 0,
                                 PSN_IS_IN_RANGE(pp->psn, req, comm), req->start_psn, req->num_packets,
                                 req->to_send, req->to_recv, num_left);

            if (PSN_RECEIVED(pp->psn, comm) || pp->psn < comm->last_acked) {
                /* This psn was already received */
                assert(pp->context == 0);
                ucc_list_add_tail(&comm->bpool, &pp->super);
            } else {
                if (num_left > 0 && PSN_IS_IN_RANGE(pp->psn, req, comm)) {
                    __builtin_prefetch(req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm));
                    __builtin_prefetch((void*) pp->buf);
                    ucc_tl_mlx5_mcast_process_packet(comm, req, pp);
                    num_left--;
                } else {
                    ucc_list_add_tail(&comm->pending_q, &pp->super);
                }
            }
        }

        comm->pending_recv -= num_comp;
        ucc_tl_mlx5_mcast_post_recv_buffers(comm);
    }

    ucc_free(wc);
    return num_left;
}


static inline int ucc_tl_mlx5_mcast_poll_recv(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct pp_packet *pp;
    struct ibv_wc     wc;
    int               num_comp;
    uint64_t          id;
    int               length;
    uint32_t          psn;

    do {
        num_comp = ibv_poll_cq(comm->rcq, 1, &wc);

        if (num_comp > 0) {
            
            if (IBV_WC_SUCCESS != wc.status) {
                tl_warn(comm->lib, "mcast_poll_recv: %s err %d num_comp \n",
                        ibv_wc_status_str(wc.status), num_comp);
                return -1;
            }

            // Make sure we received all in order.
            id     = wc.wr_id;
            length = wc.byte_len - GRH_LENGTH;
            psn    = ntohl(wc.imm_data);
            pp     = (struct pp_packet*) id;

            if (psn >= comm->psn) {
                assert(!PSN_RECEIVED(psn, comm));
                pp->psn    = psn;
                pp->length = length;
                ucc_list_add_tail(&comm->pending_q, &pp->super);
            } else {
                assert(pp->context == 0);
                ucc_list_add_tail(&comm->bpool, &pp->super);
            }

            comm->pending_recv--;
            ucc_tl_mlx5_mcast_post_recv_buffers(comm);
        }
        
    } while (num_comp);

    return num_comp;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_reliable(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{


    if (comm->racks_n != comm->child_n || comm->sacks_n != comm->parent_n ||
           comm->nack_requests) {
        
        if (comm->pending_send) {
            ucc_tl_mlx5_mcast_poll_send(comm);
        }
        
        if (comm->parent_n) {
            ucc_tl_mlx5_mcast_poll_recv(comm);
        }
        
        ucc_tl_mlx5_mcast_check_nack_requests_all(comm);
    }

    if (comm->parent_n && !comm->reliable_in_progress) {
        ucc_tl_mlx5_mcast_reliable_send(comm);
    }

    if (!comm->reliable_in_progress) {
        comm->reliable_in_progress = true;
    }

    if (comm->racks_n == comm->child_n && comm->sacks_n == comm->parent_n &&
           0 == comm->nack_requests) {
        
        // Reset for next round.
        memset(comm->parents,  0, sizeof(comm->parents));
        memset(comm->children, 0, sizeof(comm->children));

        comm->racks_n = comm->child_n  = 0;
        comm->sacks_n = comm->parent_n = 0;
        comm->reliable_in_progress     = false;

        return UCC_OK;

    }

    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_setup_mcast(ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_init_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                        ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_setup_qps(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                         ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_clean_mcast_comm(ucc_tl_mlx5_mcast_coll_comm_t *comm);

#endif
