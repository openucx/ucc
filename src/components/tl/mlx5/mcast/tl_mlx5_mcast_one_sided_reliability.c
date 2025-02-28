/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast_one_sided_reliability.h"

static ucc_status_t ucc_tl_mlx5_mcast_one_sided_setup_reliability_buffers(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_status_t                   status  = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm    = tl_team->mcast->mcast_comm;
    int                            one_sided_total_slots_size, i;

    /* this array keeps track of the number of recv packets from each process
     * used in all the protocols */
    comm->one_sided.recvd_pkts_tracker = ucc_calloc(1, comm->commsize * sizeof(uint32_t),
                                                    "one_sided.recvd_pkts_tracker");
    if (!comm->one_sided.recvd_pkts_tracker) {
        tl_error(comm->lib, "unable to malloc for one_sided.recvd_pkts_tracker");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }

    comm->one_sided.sendbuf_memkey_list = ucc_calloc
            (1, comm->commsize * sizeof(ucc_tl_mlx5_mcast_slot_mem_info_t),
             "one_sided.sendbuf_memkey_list");
    if (!comm->one_sided.sendbuf_memkey_list) {
        tl_error(comm->lib, "unable to malloc for one_sided.sendbuf_memkey_list");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }

    /* below data structures are used in async design only */
    comm->one_sided.slot_size    = comm->one_sided.reliability_scheme_msg_threshold
                                           + ONE_SIDED_SLOTS_INFO_SIZE;
    one_sided_total_slots_size   = comm->one_sided.slot_size *
                                            ONE_SIDED_SLOTS_COUNT * sizeof(char);
    comm->one_sided.slots_buffer = (char *)ucc_calloc(1, one_sided_total_slots_size,
                                                      "one_sided.slots_buffer");
    if (!comm->one_sided.slots_buffer) {
        tl_error(comm->lib, "unable to malloc for one_sided.slots_buffer");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }
    comm->one_sided.slots_mr = ibv_reg_mr(comm->ctx->pd, comm->one_sided.slots_buffer,
                                          one_sided_total_slots_size, IBV_ACCESS_LOCAL_WRITE |
                                          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    if (!comm->one_sided.slots_mr) {
        tl_error(comm->lib, "unable to register for one_sided.slots_mr");
        status = UCC_ERR_NO_RESOURCE;
        goto failed;
    }
    
    /* this array holds local information about the slot status that was read from remote ranks */
    comm->one_sided.remote_slot_info = ucc_calloc(1, comm->commsize * ONE_SIDED_SLOTS_INFO_SIZE,
                                                  "one_sided.remote_slot_info");
    if (!comm->one_sided.remote_slot_info) {
        tl_error(comm->lib, "unable to malloc for one_sided.remote_slot_info");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }
    comm->one_sided.remote_slot_info_mr = ibv_reg_mr(comm->ctx->pd, comm->one_sided.remote_slot_info,
                                                     comm->commsize * ONE_SIDED_SLOTS_INFO_SIZE,
                                                     IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE |
                                                     IBV_ACCESS_REMOTE_READ);
    if (!comm->one_sided.remote_slot_info_mr) {
        tl_error(comm->lib, "unable to register for one_sided.remote_slot_info_mr");
        status = UCC_ERR_NO_RESOURCE;
        goto failed;
    }

    comm->one_sided.info = ucc_calloc(1, sizeof(ucc_tl_mlx5_one_sided_reliable_team_info_t) *
                                      comm->commsize, "one_sided.info");
    if (!comm->one_sided.info) {
        tl_error(comm->lib, "unable to allocate mem for one_sided.info");
        status = UCC_ERR_NO_MEMORY;
        goto failed;
    }

    status = ucc_tl_mlx5_mcast_create_rc_qps(comm->ctx, comm);
    if (UCC_OK != status) {
        tl_error(comm->lib, "RC qp create failed");
        goto failed;
    }

    /* below holds the remote addr/rkey to local slot field of all the
     * processes used in async protocol */
    comm->one_sided.info[comm->rank].slot_mem.rkey        = comm->one_sided.slots_mr->rkey;
    comm->one_sided.info[comm->rank].slot_mem.remote_addr = (uint64_t)comm->one_sided.slots_buffer;
    comm->one_sided.info[comm->rank].port_lid             = comm->ctx->port_lid;
    for (i = 0; i < comm->commsize; i++) {
        comm->one_sided.info[comm->rank].rc_qp_num[i] = comm->mcast.rc_qp[i]->qp_num;
    }

    tl_debug(comm->lib, "created the allgather reliability structures");

    return UCC_OK;

failed:
    return status;
}

static inline ucc_status_t ucc_tl_mlx5_mcast_one_sided_cleanup(ucc_tl_mlx5_mcast_coll_comm_t *comm)
{
    int j;

    if (comm->mcast.rc_qp != NULL) {
        for (j=0; j<comm->commsize; j++) {
            if (comm->mcast.rc_qp[j] != NULL && ibv_destroy_qp(comm->mcast.rc_qp[j])) {
                tl_error(comm->lib, "ibv_destroy_qp failed");
                return UCC_ERR_NO_RESOURCE;
            }
            comm->mcast.rc_qp[j] = NULL;
        }

        ucc_free(comm->mcast.rc_qp);
        comm->mcast.rc_qp = NULL;
    }

    if (comm->mcast.srq != NULL && ibv_destroy_srq(comm->mcast.srq)) {
        tl_error(comm->lib, "ibv_destroy_srq failed");
        return UCC_ERR_NO_RESOURCE;
    }
    comm->mcast.srq = NULL;

    if (comm->one_sided.slots_mr) {
        ibv_dereg_mr(comm->one_sided.slots_mr);
        comm->one_sided.slots_mr = 0;
    }

    if (comm->one_sided.remote_slot_info_mr) {
        ibv_dereg_mr(comm->one_sided.remote_slot_info_mr);
        comm->one_sided.remote_slot_info_mr = 0;
    }

    if (comm->one_sided.slots_buffer) {
        ucc_free(comm->one_sided.slots_buffer);
        comm->one_sided.slots_buffer = NULL;
    }

    if (comm->one_sided.recvd_pkts_tracker) {
        ucc_free(comm->one_sided.recvd_pkts_tracker);
        comm->one_sided.recvd_pkts_tracker = NULL;
    }

    if (comm->one_sided.sendbuf_memkey_list) {
        ucc_free(comm->one_sided.sendbuf_memkey_list);
        comm->one_sided.sendbuf_memkey_list = NULL;
    }

    if (comm->one_sided.remote_slot_info) {
        ucc_free(comm->one_sided.remote_slot_info);
        comm->one_sided.remote_slot_info = NULL;
    }

    if (comm->one_sided.info) {
        ucc_free(comm->one_sided.info);
        comm->one_sided.info = NULL;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_mcast_one_sided_reliability_init(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team  = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_mcast_coll_comm_t *comm     = tl_team->mcast->mcast_comm;
    ucc_status_t                   status   = UCC_OK;

    if (comm->commsize > ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE) {
        tl_warn(comm->lib,
                "team size is %d but max supported team size of mcast one-sided reliability is %d",
                comm->commsize, ONE_SIDED_RELIABILITY_MAX_TEAM_SIZE);
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_mlx5_mcast_one_sided_setup_reliability_buffers(team);
    if (status != UCC_OK) {
        tl_error(comm->lib, "setup reliablity resources failed");
        goto failed;
    }

     /* TODO double check if ucc inplace allgather is working properly */
    status = comm->service_coll.allgather_post(comm->p2p_ctx, NULL /*inplace*/, comm->one_sided.info,
                                               sizeof(ucc_tl_mlx5_one_sided_reliable_team_info_t),
                                               &comm->one_sided.reliability_req);
    if (UCC_OK != status) {
        tl_error(comm->lib, "oob allgather failed during one-sided reliability init");
        goto failed;
    }

    return status;

failed:
    if (UCC_OK != ucc_tl_mlx5_mcast_one_sided_cleanup(comm)) {
        tl_error(comm->lib, "mcast one-sided reliablity resource cleanup failed");
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_mcast_one_sided_reliability_test(ucc_base_team_t *team)
{
    ucc_tl_mlx5_team_t            *tl_team  = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    ucc_status_t                   status   = UCC_OK;
    ucc_tl_mlx5_mcast_coll_comm_t *comm     = tl_team->mcast->mcast_comm;

    /* check if the one sided config info is exchanged */
    status = comm->service_coll.coll_test(comm->one_sided.reliability_req);
    if (UCC_OK != status) {
        /* allgather is not completed yet */
        if (status < 0) {
            tl_error(comm->lib, "one sided config info exchange failed");
            goto failed;
        }
        return status;
    }

    /* we have all the info to make the reliable connections */
    status = ucc_tl_mlx5_mcast_modify_rc_qps(comm->ctx, comm);
    if (UCC_OK != status) {
        tl_error(comm->lib, "RC qp modify failed");
        goto failed;
    }

failed:
    if (UCC_OK != ucc_tl_mlx5_mcast_one_sided_cleanup(comm)) {
        tl_error(comm->lib, "mcast one-sided reliablity resource cleanup failed");
    }
    
    return status;
}

