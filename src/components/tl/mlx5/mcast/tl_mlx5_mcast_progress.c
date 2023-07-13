/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_progress.h"

static int ucc_tl_mlx5_mcast_recv_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj);

static int ucc_tl_mlx5_mcast_send_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj);

static ucc_status_t ucc_tl_mlx5_mcast_dummy_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj)
{
    return UCC_OK;
}

static ucc_tl_mlx5_mcast_p2p_completion_obj_t dummy_completion_obj = {
    .compl_cb = ucc_tl_mlx5_mcast_dummy_completion,
};

static inline int ucc_tl_mlx5_mcast_resend_packet_reliable(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                                           int p2p_pkt_id)
{
    uint32_t          psn = comm->p2p_pkt[p2p_pkt_id].psn;
    struct pp_packet *pp  = comm->r_window[psn % comm->wsize];

    assert(pp->psn == psn);
    
    tl_trace(comm->lib, "[comm %d, rank %d] Send data NACK: to %d, psn %d, context %ld\n",
                         comm->comm_id, comm->rank,
                         comm->p2p_pkt[p2p_pkt_id].from, psn, pp->context);

    SEND_NB_NO_COMPL(comm,(void*) (pp->context ? pp->context : pp->buf), pp->length,
                     comm->p2p_pkt[p2p_pkt_id].from, 2703, comm->p2p_ctx);
    
    RECV_NB(comm,&comm->p2p_pkt[p2p_pkt_id], sizeof(struct packet), comm->p2p_pkt[p2p_pkt_id].from,
            GET_RELIABLE_TAG(comm), comm->p2p_ctx, ucc_tl_mlx5_mcast_recv_completion, p2p_pkt_id, NULL);
    
    return UCC_OK;
}

int ucc_tl_mlx5_mcast_check_nack_requests(ucc_tl_mlx5_mcast_coll_comm_t *comm, uint32_t psn)
{
    int i;

    if (!comm->nack_requests)
        return UCC_OK;

    for (i=0; i<comm->child_n; i++) {
        if (psn == comm->p2p_pkt[i].psn &&
            comm->p2p_pkt[i].type == MCAST_P2P_NEED_NACK_SEND) {
            ucc_tl_mlx5_mcast_resend_packet_reliable(comm, i);
            comm->p2p_pkt[i].type = MCAST_P2P_ACK;
            comm->nack_requests--;
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_check_nack_requests_all(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    struct pp_packet *pp;
    uint32_t          psn;
    int               i;

    if (!comm->nack_requests) {
        return UCC_OK;
    }

    for (i=0; i<comm->child_n; i++){
        if (comm->p2p_pkt[i].type == MCAST_P2P_NEED_NACK_SEND) {
            
            psn = comm->p2p_pkt[i].psn;
            pp  = comm->r_window[psn % comm->wsize];

            if (psn == pp->psn) {
                ucc_tl_mlx5_mcast_resend_packet_reliable(comm, i);
                comm->p2p_pkt[i].type = MCAST_P2P_ACK;
                comm->nack_requests--;
            }
        }
    }
    
    return UCC_OK;
}

static inline int ucc_tl_mlx5_mcast_find_nack_psn(ucc_tl_mlx5_mcast_coll_comm_t* comm,
                                                  ucc_tl_mlx5_mcast_coll_req_t *req)
{
    int psn            = MAX(comm->last_acked, req->start_psn);
    int max_search_psn = MIN(req->start_psn + req->num_packets,
                             comm->last_acked + comm->wsize + 1);

    for (; psn < max_search_psn; psn++) {
        if (!PSN_RECEIVED(psn, comm)) {
            break;
        }
    }

    assert(psn < max_search_psn);
    
    return psn;
}

static inline int psn2root(ucc_tl_mlx5_mcast_coll_req_t *req, int psn)
{
    return (psn - req->start_psn) / req->num_packets;
}

static inline int ucc_tl_mlx5_mcast_get_nack_parent(ucc_tl_mlx5_mcast_coll_req_t *req, int psn)
{
    return req->parent;
}

/* When parent resend the lost packet to a child, this function is called at child side */
static ucc_status_t ucc_tl_mlx5_mcast_recv_data_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm = (ucc_tl_mlx5_mcast_coll_comm_t *)obj->data[0];
    struct pp_packet               *pp  = (struct pp_packet *)obj->data[1];
    void                           *dest;

    ucc_tl_mlx5_mcast_coll_req_t *req = (ucc_tl_mlx5_mcast_coll_req_t *)obj->data[2];

    tl_trace(comm->lib, "[comm %d, rank %d] Recved data psn %d", comm->comm_id, comm->rank, pp->psn);

    dest = req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm);

    memcpy(dest, (void*) pp->buf, pp->length);
    req->to_recv--;
    comm->r_window[pp->psn % comm->wsize] = pp;
    
    ucc_tl_mlx5_mcast_check_nack_requests(comm, pp->psn);
    comm->psn++;

    comm->recv_drop_packet_in_progress = false;

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_reliable_send_NACK(ucc_tl_mlx5_mcast_coll_comm_t* comm,
                                                                ucc_tl_mlx5_mcast_coll_req_t *req)
{
    struct pp_packet *pp;
    int               parent;

    struct packet p = {
        .type    = MCAST_P2P_NACK,
        .psn     = ucc_tl_mlx5_mcast_find_nack_psn(comm, req),
        .from    = comm->rank,
        .comm_id = comm->comm_id,
    };

    parent = ucc_tl_mlx5_mcast_get_nack_parent(req, p.psn);
    if (parent < 0) {
        return UCC_ERR_NO_MESSAGE;
    }

    comm->nacks_counter++;

    SEND_NB_NO_COMPL(comm, &p, sizeof(struct packet), parent, GET_RELIABLE_TAG(comm), comm->p2p_ctx);
   
    tl_trace(comm->lib, "[comm %d, rank %d] Sent NAK : parent %d, psn %d",
                         comm->comm_id, comm->rank, parent, p.psn);

    // Prepare to obtain the data.
    pp         = ucc_tl_mlx5_mcast_buf_get_free(comm);
    pp->psn    = p.psn;
    pp->length = PSN_TO_RECV_LEN(pp->psn, req, comm);

    comm->recv_drop_packet_in_progress = true;

    RECV_NB(comm, (void*) pp->buf, pp->length, parent, 2703, comm->p2p_ctx,
            ucc_tl_mlx5_mcast_recv_data_completion, pp, req);
    
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_mlx5_mcast_reliable_send(ucc_tl_mlx5_mcast_coll_comm_t* comm)
{
    int i;
    int parent;

    tl_trace(comm->lib, "comm %p, psn %d, last_acked %d, n_parent %d",
                         comm, comm->psn, comm->last_acked, comm->parent_n);

    assert(!comm->reliable_in_progress);

    for (i=0; i<comm->parent_n; i++) {

        parent                    = comm->parents[i];
        comm->p2p_spkt[i].type    = MCAST_P2P_ACK;
        comm->p2p_spkt[i].psn     = comm->last_acked + comm->wsize;
        comm->p2p_spkt[i].comm_id = comm->comm_id;
        
        tl_trace(comm->lib, "rank %d, Posting SEND to parent %d, n_parent %d,  psn %d",
                             comm->rank, parent, comm->parent_n, comm->psn);

        SEND_NB(comm, &comm->p2p_spkt[i], sizeof(struct packet),
                parent, GET_RELIABLE_TAG(comm), comm->p2p_ctx, ucc_tl_mlx5_mcast_send_completion, i, NULL);

    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_mcast_recv_completion(ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj)
{
    ucc_tl_mlx5_mcast_coll_comm_t *comm   = (ucc_tl_mlx5_mcast_coll_comm_t*)obj->data[0];
    int                            pkt_id = (int)obj->data[1];
    uint32_t                       psn;
    struct pp_packet              *pp;

    assert(comm->comm_id == comm->p2p_pkt[pkt_id].comm_id);

    if (comm->p2p_pkt[pkt_id].type != MCAST_P2P_ACK) {

        assert(comm->p2p_pkt[pkt_id].type == MCAST_P2P_NACK);
        
        psn = comm->p2p_pkt[pkt_id].psn;
        pp  = comm->r_window[psn % comm->wsize];
        
        tl_trace(comm->lib, "[comm %d, rank %d] Got NACK: from %d, psn %d, avail %d",
                             comm->comm_id, comm->rank,
                             comm->p2p_pkt[pkt_id].from, psn, pp->psn == psn);

        if (pp->psn == psn) {
            ucc_tl_mlx5_mcast_resend_packet_reliable(comm, pkt_id);
        } else {
            comm->p2p_pkt[pkt_id].type = MCAST_P2P_NEED_NACK_SEND;
            comm->nack_requests++;
        }

    } else {
        comm->racks_n++;

        psn = comm->p2p_pkt[pkt_id].psn;
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

static inline int add_uniq(int *arr, int *len, int value)
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
                                                ucc_tl_mlx5_mcast_coll_req_t *req, int root)
{
    int mask  = 1;
    int vrank = TO_VIRTUAL(comm->rank, comm->commsize, root);
    int child;

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

                    RECV_NB(comm,&comm->p2p_pkt[comm->child_n - 1], sizeof(struct packet),
                            child, GET_RELIABLE_TAG(comm), comm->p2p_ctx,
                            ucc_tl_mlx5_mcast_recv_completion, comm->child_n - 1, req);
                }
            }
        }

        mask <<= 1;
    }

    return UCC_OK;
}

static inline uint64_t ucc_tl_mlx5_mcast_get_timer(void)
{
    struct timeval t;
    gettimeofday(&t, 0);
    return (uint64_t)(t.tv_sec*1000000 + t.tv_usec);
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

void ucc_tl_mlx5_mcast_process_packet(ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                      ucc_tl_mlx5_mcast_coll_req_t *req,
                                      struct pp_packet *pp)
{
    void *dest;
    assert(pp->psn >= req->start_psn &&
           pp->psn < req->start_psn + req->num_packets);

    assert(pp->length == PSN_TO_RECV_LEN(pp->psn, req, comm));
    assert(pp->context == 0);

    if (pp->length >0 ) {
        dest = req->ptr + PSN_TO_RECV_OFFSET(pp->psn, req, comm);
        memcpy(dest, (void*) pp->buf, pp->length);
    }

    comm->r_window[pp->psn & (comm->wsize-1)] = pp;
    ucc_tl_mlx5_mcast_check_nack_requests(comm, pp->psn);
    req->to_recv--;
    comm->psn++;
    assert(comm->recv_drop_packet_in_progress == false);
}

