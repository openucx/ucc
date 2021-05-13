/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_dpu.h"
#include "tl_dpu_coll.h"
#include "coll_score/ucc_coll_score.h"

UCC_CLASS_INIT_FUNC(ucc_tl_dpu_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    

    ucc_status_t ucc_status = UCC_OK; 
    ucc_tl_dpu_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_dpu_context_t);

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super);

    tl_info(ctx->super.super.lib, "starting: %p team_create", self);

    ucp_request_param_t     send_req_param,
                            recv_req_param;
    int tc_poll = UCC_TL_DPU_TC_POLL, i;
    size_t total_rkey_size = 0;
    
    self->coll_id   = 1;
    self->size      = params->params.oob.participants;
    self->rank      = params->rank;
    self->status    = UCC_OPERATION_INITIALIZED;
    self->conn_buf  = ucc_malloc(sizeof(ucc_tl_dpu_conn_buf_t),
                        "Allocate connection buffer");

    self->conn_buf->mmap_params.field_mask =
                                UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    self->conn_buf->mmap_params.address = (void*)&self->ctrl_seg;
    self->conn_buf->mmap_params.length = sizeof(self->ctrl_seg);

    ucc_status = ucs_status_to_ucc_status(
            ucp_mem_map(ctx->ucp_context, &self->conn_buf->mmap_params,
                        &self->ctrl_seg_memh));
    if (UCC_OK != ucc_status) {
        goto err;
    }

    ucc_status = ucs_status_to_ucc_status(
        ucp_rkey_pack(ctx->ucp_context, self->ctrl_seg_memh,
                      &self->conn_buf->ctrl_seg_rkey_buf,
                      &self->conn_buf->ctrl_seg_rkey_buf_size));
    if (UCC_OK != ucc_status) {
        goto err;
    }

    send_req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_DATATYPE;
    send_req_param.datatype     = ucp_dt_make_contig(1);
    send_req_param.cb.send      = ucc_tl_dpu_send_handler_nbx;

    recv_req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_DATATYPE;
    recv_req_param.datatype     = ucp_dt_make_contig(1);
    recv_req_param.cb.recv      = ucc_tl_dpu_recv_handler_nbx;

    self->send_req[0] = ucp_tag_send_nbx(ctx->ucp_ep,
                                    &self->conn_buf->mmap_params.address,
                                    sizeof(uint64_t),
                                    UCC_TL_DPU_EXCHANGE_ADDR_TAG,
                                    &send_req_param);
    ucc_status = ucc_tl_dpu_req_check(self, self->send_req[0]);
    if (UCC_OK != ucc_status) {
        goto err;
    }

    self->send_req[1] = ucp_tag_send_nbx(ctx->ucp_ep,
                                    &self->conn_buf->ctrl_seg_rkey_buf_size,
                                    sizeof(size_t),
                                    UCC_TL_DPU_EXCHANGE_LENGTH_TAG,
                                    &send_req_param);
    ucc_status = ucc_tl_dpu_req_check(self, self->send_req[1]);
    if (UCC_OK != ucc_status) {
        goto err;
    }

    self->send_req[2] = ucp_tag_send_nbx(ctx->ucp_ep,
                                    self->conn_buf->ctrl_seg_rkey_buf,
                                    self->conn_buf->ctrl_seg_rkey_buf_size,
                                    UCC_TL_DPU_EXCHANGE_RKEY_TAG,
                                    &send_req_param);
    ucc_status = ucc_tl_dpu_req_check(self, self->send_req[2]);
    if (UCC_OK != ucc_status) {
        goto err;
    }

    self->recv_req[0] = ucp_tag_recv_nbx(ctx->ucp_worker,
                                self->conn_buf->rem_rkeys_lengths,
                                sizeof(self->conn_buf->rem_rkeys_lengths),
                                UCC_TL_DPU_EXCHANGE_LENGTH_TAG, (uint64_t)-1,
                                &recv_req_param);
    ucc_status = ucc_tl_dpu_req_check(self, self->recv_req[0]);
    if (UCC_OK != ucc_status) {
        goto err;
    }

    for (i = 0; i < tc_poll; i++) {
        ucp_worker_progress(ctx->ucp_worker);
        if ((ucc_tl_dpu_req_test(&(self->send_req[0]), ctx->ucp_worker) == UCC_OK) &&
            (ucc_tl_dpu_req_test(&(self->send_req[1]), ctx->ucp_worker) == UCC_OK) &&
            (ucc_tl_dpu_req_test(&(self->send_req[2]), ctx->ucp_worker) == UCC_OK) &&
            (ucc_tl_dpu_req_test(&(self->recv_req[0]), ctx->ucp_worker) == UCC_OK))
        {
            self->status = UCC_INPROGRESS; /* Advance connection establishment */
            break;
        }
    }

    if (UCC_INPROGRESS != self->status) {
        return UCC_OK;
    }

    ucp_rkey_buffer_release(self->conn_buf->ctrl_seg_rkey_buf);
    self->conn_buf->ctrl_seg_rkey_buf = NULL;

    total_rkey_size     = self->conn_buf->rem_rkeys_lengths[0] +
                          self->conn_buf->rem_rkeys_lengths[1] +
                          self->conn_buf->rem_rkeys_lengths[2];
    self->conn_buf->rem_rkeys = ucc_malloc(total_rkey_size, "rem_rkeys alloc");
    self->recv_req[0]   = ucp_tag_recv_nbx(ctx->ucp_worker,
                                    &self->conn_buf->rem_addresses,
                                    sizeof(self->conn_buf->rem_addresses),
                                    UCC_TL_DPU_EXCHANGE_ADDR_TAG, (uint64_t)-1,
                                    &recv_req_param);
    if (ucc_tl_dpu_req_check(self, self->recv_req[0]) != UCC_OK) {
        goto err;
    }

    self->recv_req[1] = ucp_tag_recv_nbx(ctx->ucp_worker, self->conn_buf->rem_rkeys,
                                total_rkey_size,
                                UCC_TL_DPU_EXCHANGE_RKEY_TAG, (uint64_t)-1,
                                &recv_req_param);
    if (ucc_tl_dpu_req_check(self, self->recv_req[1]) != UCC_OK) {
        goto err;
    }

    for (i = 0; i < tc_poll; i++) {
        ucp_worker_progress(ctx->ucp_worker);
        if ((ucc_tl_dpu_req_test(&(self->recv_req[0]), ctx->ucp_worker) == UCC_OK) &&
            (ucc_tl_dpu_req_test(&(self->recv_req[1]), ctx->ucp_worker) == UCC_OK))
        {
            self->status = UCC_OK;
            break;
        }
    }
    if (UCC_OK != self->status) {
        return UCC_OK;
    }

    self->rem_ctrl_seg = self->conn_buf->rem_addresses[0];
    ucc_status = ucs_status_to_ucc_status(
        ucp_ep_rkey_unpack(ctx->ucp_ep, self->conn_buf->rem_rkeys,
                            &self->rem_ctrl_seg_key));
    if (UCC_OK != ucc_status) {
        goto err;
    }
    self->rem_data_in = self->conn_buf->rem_addresses[1];

    ucc_status = ucs_status_to_ucc_status(
        ucp_ep_rkey_unpack(ctx->ucp_ep,
                    (void*)((ptrdiff_t)self->conn_buf->rem_rkeys +
                    self->conn_buf->rem_rkeys_lengths[0]),
                    &self->rem_data_in_key));
    if (UCC_OK != ucc_status) {
        goto err;
    }
    self->rem_data_out = self->conn_buf->rem_addresses[2];

    ucc_status = ucs_status_to_ucc_status(
        ucp_ep_rkey_unpack(ctx->ucp_ep,
                            (void*)((ptrdiff_t)self->conn_buf->rem_rkeys +
                            self->conn_buf->rem_rkeys_lengths[1] +
                            self->conn_buf->rem_rkeys_lengths[0]),
                       &self->rem_data_out_key));
    if (UCC_OK != ucc_status) {
        goto err;
    }

    ucc_free(self->conn_buf->rem_rkeys);
    self->conn_buf->rem_rkeys = NULL;
    ucc_free(self->conn_buf);

    return self->status;
err:
    if (self->conn_buf->rem_rkeys) {
        ucc_free(self->conn_buf->rem_rkeys);
    }
    if (self->conn_buf->ctrl_seg_rkey_buf) {
        ucp_rkey_buffer_release(self->conn_buf->ctrl_seg_rkey_buf);
        self->conn_buf->ctrl_seg_rkey_buf = NULL;
    }
    ucc_free(self->conn_buf);

    return ucc_status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_dpu_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_dpu_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_dpu_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_dpu_team_destroy(ucc_base_team_t *tl_team)
{
    ucc_tl_dpu_team_t           *team = ucc_derived_of(tl_team, ucc_tl_dpu_team_t);
    ucc_tl_dpu_context_t        *ctx = UCC_TL_DPU_TEAM_CTX(team);
    ucc_tl_dpu_sync_t           hangup;
    ucc_tl_dpu_request_t        *hangup_req;
    ucp_request_param_t         req_param;
 
    hangup.coll_id  = team->coll_id;
    hangup.dtype    = UCC_DT_USERDEFINED;
    hangup.op       = UCC_OP_USERDEFINED;
    hangup.count_in      = 0;
 
    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                             UCP_OP_ATTR_FIELD_DATATYPE;
    req_param.datatype     = ucp_dt_make_contig(1);
    req_param.cb.send      = ucc_tl_dpu_send_handler_nbx;
 
    hangup_req = ucp_put_nbx(ctx->ucp_ep, &hangup, sizeof(hangup),
                             team->rem_ctrl_seg, team->rem_ctrl_seg_key,
                             &req_param);
    if (ucc_tl_dpu_req_check(team, hangup_req) != UCC_OK) {
        return UCC_ERR_NO_MESSAGE;
    }
    do {
        ucp_worker_progress(ctx->ucp_worker);
    } while((ucc_tl_dpu_req_test(&(hangup_req), ctx->ucp_worker) != UCC_OK));
 
    ucp_rkey_destroy(team->rem_ctrl_seg_key);
    ucp_rkey_destroy(team->rem_data_in_key);
    ucp_rkey_destroy(team->rem_data_out_key);
    ucp_mem_unmap(ctx->ucp_context, team->ctrl_seg_memh);

    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_dpu_team_t)(tl_team);

    return UCC_OK;
}

ucc_status_t ucc_tl_dpu_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_dpu_team_t       *team = ucc_derived_of(tl_team, ucc_tl_dpu_team_t);
    ucc_tl_dpu_context_t    *ctx = UCC_TL_DPU_TEAM_CTX(team);
    ucc_status_t            ucc_status = UCC_OK;
    int                     tc_poll = UCC_TL_DPU_TC_POLL, i = 0;
    size_t                  total_rkey_size;
    ucp_request_param_t     recv_req_param;

    if (UCC_OK == team->status) {
        return UCC_OK;
    }

    recv_req_param.op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK |
                                  UCP_OP_ATTR_FIELD_DATATYPE;
    recv_req_param.datatype     = ucp_dt_make_contig(1);
    recv_req_param.cb.recv      = ucc_tl_dpu_recv_handler_nbx;

    if (UCC_OPERATION_INITIALIZED == team->status) {
        for (i = 0; i < tc_poll; i++) {
            ucp_worker_progress(ctx->ucp_worker);
            if ((ucc_tl_dpu_req_test(&(team->send_req[0]), ctx->ucp_worker) == UCC_OK) &&
                (ucc_tl_dpu_req_test(&(team->send_req[1]), ctx->ucp_worker) == UCC_OK) &&
                (ucc_tl_dpu_req_test(&(team->send_req[2]), ctx->ucp_worker) == UCC_OK) &&
                (ucc_tl_dpu_req_test(&(team->recv_req[0]), ctx->ucp_worker) == UCC_OK))
            {
                team->status = UCC_INPROGRESS; /* Advance connection establishment */
                break;
            }
        }

        if (UCC_INPROGRESS != team->status) {
            return UCC_INPROGRESS;
        }

        /* Continue connection establishment */
        ucp_rkey_buffer_release(team->conn_buf->ctrl_seg_rkey_buf);
        team->conn_buf->ctrl_seg_rkey_buf = NULL;

        total_rkey_size = team->conn_buf->rem_rkeys_lengths[0] +
                          team->conn_buf->rem_rkeys_lengths[1] +
                          team->conn_buf->rem_rkeys_lengths[2];
        team->conn_buf->rem_rkeys = ucc_malloc(total_rkey_size, "rem_rkeys alloc");

        team->recv_req[0] = ucp_tag_recv_nbx(ctx->ucp_worker,
                            &team->conn_buf->rem_addresses,
                            sizeof(team->conn_buf->rem_addresses),
                            UCC_TL_DPU_EXCHANGE_ADDR_TAG, (uint64_t)-1,
                            &recv_req_param);
        if (ucc_tl_dpu_req_check(team, team->recv_req[0]) != UCC_OK) {
            goto err;
        }

        team->recv_req[1] = ucp_tag_recv_nbx(ctx->ucp_worker,
                            team->conn_buf->rem_rkeys,
                            total_rkey_size,
                            UCC_TL_DPU_EXCHANGE_RKEY_TAG, (uint64_t)-1,
                            &recv_req_param);
        if (ucc_tl_dpu_req_check(team, team->recv_req[1]) != UCC_OK) {
            goto err;
        }

        for (i = 0; i < tc_poll; i++) {
            ucp_worker_progress(ctx->ucp_worker);
            if ((ucc_tl_dpu_req_test(&(team->recv_req[0]), ctx->ucp_worker) == UCC_OK) &&
                (ucc_tl_dpu_req_test(&(team->recv_req[1]), ctx->ucp_worker) == UCC_OK))
            {
                team->status = UCC_OK;
                break;
            }
        }
        if (UCC_OK != team->status) {
            return UCC_INPROGRESS;
        }
    }

    if (UCC_INPROGRESS == team->status) {
        for (i = 0; i < tc_poll; i++) {
            ucp_worker_progress(ctx->ucp_worker);
            if ((ucc_tl_dpu_req_test(&(team->recv_req[0]), ctx->ucp_worker) == UCC_OK) &&
                (ucc_tl_dpu_req_test(&(team->recv_req[1]), ctx->ucp_worker) == UCC_OK))
            {
                team->status = UCC_OK;
                break;
            }
        }
        if (UCC_OK != team->status) {
            return UCC_INPROGRESS;
        }
    }

    team->rem_ctrl_seg = team->conn_buf->rem_addresses[0];
    ucc_status = ucs_status_to_ucc_status(
        ucp_ep_rkey_unpack(ctx->ucp_ep, team->conn_buf->rem_rkeys,
                            &team->rem_ctrl_seg_key));
    if (UCC_OK != ucc_status) {
        goto err;
    }
    team->rem_data_in = team->conn_buf->rem_addresses[1];

    ucc_status = ucs_status_to_ucc_status(
        ucp_ep_rkey_unpack(ctx->ucp_ep,
                    (void*)((ptrdiff_t)team->conn_buf->rem_rkeys +
                    team->conn_buf->rem_rkeys_lengths[0]),
                    &team->rem_data_in_key));
    if (UCC_OK != ucc_status) {
        goto err;
    }
    team->rem_data_out = team->conn_buf->rem_addresses[2];

    ucc_status = ucs_status_to_ucc_status(
        ucp_ep_rkey_unpack(ctx->ucp_ep,
                            (void*)((ptrdiff_t)team->conn_buf->rem_rkeys +
                            team->conn_buf->rem_rkeys_lengths[1] +
                            team->conn_buf->rem_rkeys_lengths[0]),
                       &team->rem_data_out_key));
    if (UCC_OK != ucc_status) {
        goto err;
    }

    ucc_free(team->conn_buf->rem_rkeys);
    team->conn_buf->rem_rkeys = NULL;
    ucc_free(team->conn_buf);

    return team->status;
err:
    if (team->conn_buf->rem_rkeys) {
        ucc_free(team->conn_buf->rem_rkeys);
    }
    if (team->conn_buf->ctrl_seg_rkey_buf) {
        ucp_rkey_buffer_release(team->conn_buf->ctrl_seg_rkey_buf);
        team->conn_buf->ctrl_seg_rkey_buf = NULL;
    }
    ucc_free(team->conn_buf);

    return ucc_status;
}

ucc_status_t ucc_tl_dpu_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_dpu_team_t  *team = ucc_derived_of(tl_team, ucc_tl_dpu_team_t);
    ucc_tl_dpu_lib_t   *lib  = UCC_TL_DPU_TEAM_LIB(team);
    ucc_coll_score_t   *score;
    ucc_status_t        status;

    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status = ucc_coll_score_build_default(tl_team, UCC_TL_DPU_DEFAULT_SCORE,
                           ucc_tl_dpu_coll_init, UCC_TL_DPU_SUPPORTED_COLLS,
                           NULL, 0, &score);
    if (UCC_OK != status) {
        return status;
    }
    if (strlen(lib->super.super.score_str) > 0) {
        status = ucc_coll_score_update_from_str(lib->super.super.score_str,
                                                score, team->size,
                                                ucc_tl_dpu_coll_init, &team->super.super,
                                                UCC_TL_DPU_DEFAULT_SCORE,
                                                NULL);
        if (status == UCC_ERR_INVALID_PARAM) {
            /* User provided incorrect input - try to proceed */
            goto err;
        }
    }
    *score_p = score;
    return status;
err:
    ucc_coll_score_free(score);
    return status;
}