/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include "tl_ucp_sendrecv.h"

void ucc_tl_ucp_send_recv_counter_inc_st(uint32_t *counter)
{
    ++(*counter);
}

void ucc_tl_ucp_send_recv_counter_inc_mt(uint32_t *counter)
{
    ucc_atomic_add32(counter, 1);
}

void ucc_tl_ucp_send_completion_cb_st(void *request, ucs_status_t status,
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in send completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ++task->tagged.send_completed;
    ucp_request_free(request);
}

void ucc_tl_ucp_send_completion_cb_mt(void *request, ucs_status_t status,
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in send completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ucc_atomic_add32(&task->tagged.send_completed, 1);
    ucp_request_free(request);
}

void ucc_tl_ucp_put_completion_cb(void *request, ucs_status_t status,
                                  void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in put completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->onesided.put_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_get_completion_cb(void *request, ucs_status_t status,
                                  void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in get completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->onesided.get_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_flush_completion_cb(void *request, ucs_status_t status,
                                    void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in ep flush completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->flush_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_recv_completion_cb_mt(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info, /* NOLINT */
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ucc_atomic_add32(&task->tagged.recv_completed, 1);
    ucp_request_free(request);
}

void ucc_tl_ucp_recv_completion_cb_st(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info, /* NOLINT */
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ++task->tagged.recv_completed;
    ucp_request_free(request);
}

ucc_status_t ucc_tl_ucp_send_nbx(void *buffer, size_t msglen,
                                 ucc_rank_t dest_group_rank,
                                 const ucp_request_param_t *req_param,
                                 ucc_tl_ucp_task_t *task)
{
    const ucc_coll_args_t *args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucc_status_t           status;
    ucp_ep_h               ep;
    ucp_tag_t              ucp_tag;
    ucs_status_ptr_t       ucp_status;

    status = ucc_tl_ucp_get_ep(team, dest_group_rank, &ep);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    ucp_tag = UCC_TL_UCP_MAKE_SEND_TAG((args->mask & UCC_COLL_ARGS_FIELD_TAG),
                                       task->tagged.tag, UCC_TL_TEAM_RANK(team),
                                       team->super.super.params.id,
                                       team->super.super.params.scope_id,
                                       team->super.super.params.scope);
    ucp_status = ucp_tag_send_nbx(ep, buffer, msglen, ucp_tag, req_param);
    task->tagged.send_posted++;

    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        UCC_TL_UCP_TEAM_CTX(team)->sendrecv_cbs.p2p_counter_inc(
            &task->tagged.send_completed);
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_recv_nbx(void *buffer, size_t msglen,
                                 ucc_rank_t dest_group_rank,
                                 const ucp_request_param_t *req_param,
                                 ucc_tl_ucp_task_t *task)
{
    const ucc_coll_args_t *args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucp_tag_t              ucp_tag, ucp_tag_mask;
    ucs_status_ptr_t       ucp_status;


    UCC_TL_UCP_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask,
                             (args->mask & UCC_COLL_ARGS_FIELD_TAG),
                             task->tagged.tag, dest_group_rank,
                             team->super.super.params.id,
                             team->super.super.params.scope_id,
                             team->super.super.params.scope);

    ucp_status = ucp_tag_recv_nbx(team->worker->ucp_worker, buffer, msglen,
                                  ucp_tag, ucp_tag_mask, req_param);
    task->tagged.recv_posted++;

    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        UCC_TL_UCP_TEAM_CTX(team)->sendrecv_cbs.p2p_counter_inc(
            &task->tagged.recv_completed);
    }
    return UCC_OK;
}
