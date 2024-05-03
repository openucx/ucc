/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_progress.h"

static ucc_status_t ucc_tl_mlx5_mcast_recv_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj);

static ucc_status_t ucc_tl_mlx5_mcast_send_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj);

static ucc_status_t ucc_tl_mlx5_mcast_reliability_send_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *comp_obj)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm   = (ucc_tl_mlx5_mcast_coll_comm_t*)comp_obj->data[0];
    unsigned int                   pkt_id = comp_obj->data[2];
    struct packet                 *p      = (struct packet *)comp_obj->data[1];
    ucc_status_t                   status;

    if (p != NULL) {
        /* it was a nack packet to our parent */
        ucc_free(p);
    }

    if (pkt_id != UINT_MAX) {
        /* we sent the real data to our child so reduce the nack reqs */
        ucc_assert(comm->nack_requests > 0);
        ucc_assert(comm->p2p_pkt[pkt_id].type == MCAST_P2P_NACK_SEND_PENDING);
        comm->p2p_pkt[pkt_id].type = MCAST_P2P_ACK;
        comm->nack_requests--;
        status = comm->params.p2p_iface.recv_nb(&comm->p2p_pkt[pkt_id],
                                                sizeof(struct packet), comm->p2p_pkt[pkt_id].from,
                                                comm->p2p_ctx, GET_COMPL_OBJ(comm,
                                                ucc_tl_mlx5_mcast_recv_completion, pkt_id, NULL));
        if (status <  0) {
            return status;
        }
    }

    ucc_mpool_put(comp_obj);

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_resend_packet_reliable(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                                    int p2p_pkt_id)
{
    uint32_t          psn = comm->p2p_pkt[p2p_pkt_id].psn;
    struct pp_packet *pp  = comm->r_window[psn % comm->wsize];
    ucc_status_t      status;

    ucc_assert(pp->psn == psn);
    ucc_assert(comm->p2p_pkt[p2p_pkt_id].type == MCAST_P2P_NEED_NACK_SEND);

    comm->p2p_pkt[p2p_pkt_id].type = MCAST_P2P_NACK_SEND_PENDING;
    
    tl_trace(comm->lib, "[comm %d, rank %d] Send data NACK: to %d, psn %d, context %ld nack_requests %d \n",
                         comm->comm_id, comm->rank,
                         comm->p2p_pkt[p2p_pkt_id].from, psn, pp->context, comm->nack_requests);

    status = comm->params.p2p_iface.send_nb((void*) (pp->context ? pp->context : pp->buf),
                                            pp->length, comm->p2p_pkt[p2p_pkt_id].from,
                                            comm->p2p_ctx, GET_COMPL_OBJ(comm,
                                            ucc_tl_mlx5_mcast_reliability_send_completion, NULL, p2p_pkt_id));
    if (status <  0) {
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_check_nack_requests(ucc_tl_mlx5_mcast_coll_comm_t *comm, uint32_t psn)
{
    ucc_status_t status = UCC_OK;
    int          i;
    struct pp_packet *pp;

    if (!comm->nack_requests) {
        return UCC_OK;
    }

    if (psn != UINT32_MAX) {
        for (i=0; i<comm->child_n; i++) {
            if (psn == comm->p2p_pkt[i].psn &&
                comm->p2p_pkt[i].type == MCAST_P2P_NEED_NACK_SEND) {
                status = ucc_tl_mlx5_mcast_resend_packet_reliable(comm, i);
                if (status != UCC_OK) {
                    break;
                }
            }
        }
    } else {
        for (i=0; i<comm->child_n; i++){
            if (comm->p2p_pkt[i].type == MCAST_P2P_NEED_NACK_SEND) {
                psn = comm->p2p_pkt[i].psn;
                pp  = comm->r_window[psn % comm->wsize];
                if (psn == pp->psn) {
                    status = ucc_tl_mlx5_mcast_resend_packet_reliable(comm, i);
                    if (status < 0) {
                        break;
                    }
                }
            }
        }
    }

    return status;
}

static inline int ucc_tl_mlx5_mcast_find_nack_psn(ucc_tl_mlx5_mcast_coll_comm_t* comm,
                                                  ucc_tl_mlx5_mcast_coll_req_t *req)
{
    int psn            = ucc_max(comm->last_acked, req->start_psn);
    int max_search_psn = ucc_min(req->start_psn + req->num_packets,
                             comm->last_acked + comm->wsize + 1);

    for (; psn < max_search_psn; psn++) {
        if (!PSN_RECEIVED(psn, comm)) {
            break;
        }
    }

    ucc_assert(psn < max_search_psn);
    
    return psn;
}

static inline ucc_rank_t ucc_tl_mlx5_mcast_get_nack_parent(ucc_tl_mlx5_mcast_coll_req_t *req)
{
    return req->parent;
}

/* When parent resend the lost packet to a child, this function is called at child side */
static ucc_status_t ucc_tl_mlx5_mcast_recv_data_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj)
{
    ucc_status_t                   status = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm   = (ucc_tl_mlx5_mcast_coll_comm_t *)obj->data[0];
    struct pp_packet               *pp    = (struct pp_packet *)obj->data[1];
    ucc_tl_mlx5_mcast_coll_req_t   *req   = (ucc_tl_mlx5_mcast_coll_req_t *)obj->data[2];
    void                           *dest;

    tl_trace(comm->lib, "[comm %d, rank %d] Recved data psn %d", comm->comm_id, comm->rank, pp->psn);

    dest = req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm);
    memcpy(dest, (void*) pp->buf, pp->length);
    req->to_recv--;
    comm->r_window[pp->psn % comm->wsize] = pp;
    
    status = ucc_tl_mlx5_mcast_check_nack_requests(comm, pp->psn);
    if (status < 0) {
        return status;
    }

    comm->psn++;
    comm->recv_drop_packet_in_progress = false;

    return status;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_reliable_send_NACK(ucc_tl_mlx5_mcast_coll_comm_t* comm,
                                                                ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_status_t      status = UCC_OK;
    uint32_t          psn    = ucc_tl_mlx5_mcast_find_nack_psn(comm, req);
    struct pp_packet *pp;
    ucc_rank_t        parent;
    struct packet    *p;

    p          = ucc_calloc(1, sizeof(struct packet));
    p->type    = MCAST_P2P_NACK;
    p->psn     = psn;
    p->from    = comm->rank;
    p->comm_id = comm->comm_id;

    parent = ucc_tl_mlx5_mcast_get_nack_parent(req);

    comm->nacks_counter++;

    status = comm->params.p2p_iface.send_nb(p, sizeof(struct packet), parent,
                                            comm->p2p_ctx, GET_COMPL_OBJ(comm,
                                            ucc_tl_mlx5_mcast_reliability_send_completion, p, UINT_MAX));
    if (status <  0) {
        return status;
    }
   
    tl_trace(comm->lib, "[comm %d, rank %d] Sent NAK : parent %d, psn %d",
             comm->comm_id, comm->rank, parent, psn);

    // Prepare to obtain the data.
    pp         = ucc_tl_mlx5_mcast_buf_get_free(comm);
    pp->psn    = psn;
    pp->length = PSN_TO_RECV_LEN(pp->psn, req, comm);

    comm->recv_drop_packet_in_progress = true;

    status = comm->params.p2p_iface.recv_nb((void*) pp->buf,
                                            pp->length, parent,
                                            comm->p2p_ctx, GET_COMPL_OBJ(comm,
                                            ucc_tl_mlx5_mcast_recv_data_completion, pp, req));
    if (status <  0) {
        return status;
    }

    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_mlx5_mcast_reliable_send(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    ucc_rank_t   i;
    ucc_rank_t   parent;
    ucc_status_t status;

    tl_trace(comm->lib, "comm %p, psn %d, last_acked %d, n_parent %d",
                         comm, comm->psn, comm->last_acked, comm->parent_n);

    ucc_assert(!comm->reliable_in_progress);

    for (i=0; i<comm->parent_n; i++) {
        parent                    = comm->parents[i];
        comm->p2p_spkt[i].type    = MCAST_P2P_ACK;
        comm->p2p_spkt[i].psn     = comm->last_acked + comm->wsize;
        comm->p2p_spkt[i].comm_id = comm->comm_id;
        
        tl_trace(comm->lib, "rank %d, Posting SEND to parent %d, n_parent %d,  psn %d",
                 comm->rank, parent, comm->parent_n, comm->psn);

        status = comm->params.p2p_iface.send_nb(&comm->p2p_spkt[i],
                                                sizeof(struct packet), parent,
                                                comm->p2p_ctx, GET_COMPL_OBJ(comm,
                                                ucc_tl_mlx5_mcast_send_completion, i, NULL));
        if (status <  0) {
            return status;
        }
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_mcast_recv_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm   = (ucc_tl_mlx5_mcast_coll_comm_t*)obj->data[0];
    int                            pkt_id = (int)obj->data[1];
    uint32_t                       psn;
    struct pp_packet              *pp;
    ucc_status_t                   status;

    ucc_assert(comm->comm_id == comm->p2p_pkt[pkt_id].comm_id);

    if (comm->p2p_pkt[pkt_id].type != MCAST_P2P_ACK) {
        ucc_assert(comm->p2p_pkt[pkt_id].type == MCAST_P2P_NACK);
        psn = comm->p2p_pkt[pkt_id].psn;
        pp  = comm->r_window[psn % comm->wsize];
        
        tl_trace(comm->lib, "[comm %d, rank %d] Got NACK: from %d, psn %d, avail %d pkt_id %d",
                             comm->comm_id, comm->rank,
                             comm->p2p_pkt[pkt_id].from, psn, pp->psn == psn, pkt_id);

        comm->p2p_pkt[pkt_id].type = MCAST_P2P_NEED_NACK_SEND;
        comm->nack_requests++;

        if (pp->psn == psn) {
            /* parent already has this packet so it is ready to forward it to its child */
            status = ucc_tl_mlx5_mcast_resend_packet_reliable(comm, pkt_id);
            if (status != UCC_OK) {
                return status;
            }
        }

    } else {
        ucc_assert(comm->p2p_pkt[pkt_id].type == MCAST_P2P_ACK);
        comm->racks_n++;
    }

    ucc_mpool_put(obj); /* return the completion object back to the mem pool compl_objects_mp */

    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_mcast_send_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm = (ucc_tl_mlx5_mcast_coll_comm_t*)obj->data[0];
 
    comm->sacks_n++;
    ucc_mpool_put(obj);
    return UCC_OK;
}

static inline int add_uniq(ucc_rank_t *arr, uint32_t *len, ucc_rank_t value)
{
    int i;
    
    for (i=0; i<(*len); i++) {
        if (arr[i] == value) {
            return 0;
        }
    }
    
    arr[*len] = value;
    (*len)++;
    return 1;
}

ucc_status_t ucc_tl_mlx5_mcast_prepare_reliable(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                ucc_tl_mlx5_mcast_coll_req_t *req,
                                                ucc_rank_t root)
{
    ucc_rank_t   mask  = 1;
    ucc_rank_t   vrank = TO_VIRTUAL(comm->rank, comm->commsize, root);
    ucc_rank_t   child;
    ucc_status_t status;

    ucc_assert(comm->commsize <= pow(2, MAX_COMM_POW2));

    while (mask < comm->commsize) {
        if (vrank & mask) {
            req->parent = TO_ORIGINAL((vrank ^ mask), comm->commsize, root);
            add_uniq(comm->parents, &comm->parent_n, req->parent);
            break;
        } else {
            child = vrank ^ mask;
            if (child < comm->commsize) {
                child = TO_ORIGINAL(child, comm->commsize, root);
                if (add_uniq(comm->children, &comm->child_n, child)) {
                    tl_trace(comm->lib, "rank %d, Posting RECV from child %d, n_child %d,  psn %d",
                             comm->rank, child, comm->child_n, comm->psn);

                    status = comm->params.p2p_iface.recv_nb(&comm->p2p_pkt[comm->child_n - 1],
                                                            sizeof(struct packet), child,
                                                            comm->p2p_ctx, GET_COMPL_OBJ(comm,
                                                            ucc_tl_mlx5_mcast_recv_completion, comm->child_n - 1, req));
                    if (status <  0) {
                        return status;
                    }
                }
            }
        }

        mask <<= 1;
    }

    return UCC_OK;
}

static inline uint64_t ucc_tl_mlx5_mcast_get_timer(void)
{
    double t_second = ucc_get_time();
    return (uint64_t) (t_second * 1000000);
}

ucc_status_t ucc_tl_mlx5_mcast_bcast_check_drop(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                ucc_tl_mlx5_mcast_coll_req_t *req)
{
    ucc_status_t status = UCC_OK;

    if (comm->timer == 0) {
        comm->timer = ucc_tl_mlx5_mcast_get_timer();
    } else {
        if (ucc_tl_mlx5_mcast_get_timer() - comm->timer >= comm->ctx->params.timeout) {
            tl_trace(comm->lib, "[REL] time out %d", comm->psn);
            status = ucc_tl_mlx5_mcast_reliable_send_NACK(comm, req);
            comm->timer = 0;
        }
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_process_packet(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                              ucc_tl_mlx5_mcast_coll_req_t *req,
                                              struct pp_packet* pp)
{
    ucc_status_t status = UCC_OK;
    void *dest;
    ucc_assert(pp->psn >= req->start_psn &&
           pp->psn < req->start_psn + req->num_packets);

    ucc_assert(pp->length == PSN_TO_RECV_LEN(pp->psn, req, comm));
    ucc_assert(pp->context == 0);

    if (pp->length > 0 ) {
        dest = req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm);
        memcpy(dest, (void*) pp->buf, pp->length);
    }

    comm->r_window[pp->psn & (comm->wsize-1)] = pp;
    status = ucc_tl_mlx5_mcast_check_nack_requests(comm, pp->psn);
    if (status < 0) {
        return status;
    }

    req->to_recv--;
    comm->psn++;
    ucc_assert(comm->recv_drop_packet_in_progress == false);

    return status;
}

