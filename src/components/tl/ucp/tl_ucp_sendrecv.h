/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_TL_UCP_SENDRECV_H_
#define UCC_TL_UCP_SENDRECV_H_

#include "tl_ucp_tag.h"
#include "tl_ucp_ep.h"
#include "utils/ucc_compiler_def.h"
#include "components/mc/base/ucc_mc_base.h"

void ucc_tl_ucp_send_completion_cb_st(void *request, ucs_status_t status,
                                      void *user_data);
void ucc_tl_ucp_send_completion_cb_mt(void *request, ucs_status_t status,
                                      void *user_data);

                                    
void ucc_tl_ucp_put_completion_cb(void *request, ucs_status_t status,
                                  void *user_data);

void ucc_tl_ucp_get_completion_cb(void *request, ucs_status_t status,
                                  void *user_data);

void ucc_tl_ucp_recv_completion_cb_st(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info,
                                      void *user_data);
void ucc_tl_ucp_recv_completion_cb_mt(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info,
                                      void *user_data);
  
#define UCC_TL_UCP_MAKE_TAG(_user_tag, _tag, _rank, _id, _scope_id, _scope)    \
    ((((uint64_t) (_user_tag)) << UCC_TL_UCP_USER_TAG_BITS_OFFSET) |           \
     (((uint64_t) (_tag))      << UCC_TL_UCP_TAG_BITS_OFFSET)      |           \
     (((uint64_t) (_rank))     << UCC_TL_UCP_SENDER_BITS_OFFSET)   |           \
     (((uint64_t) (_scope))    << UCC_TL_UCP_SCOPE_BITS_OFFSET)    |           \
     (((uint64_t) (_scope_id)) << UCC_TL_UCP_SCOPE_ID_BITS_OFFSET) |           \
     (((uint64_t) (_id))       << UCC_TL_UCP_ID_BITS_OFFSET))

#define UCC_TL_UCP_MAKE_SEND_TAG(_user_tag, _tag, _rank, _id, _scope_id, _scope)          \
    UCC_TL_UCP_MAKE_TAG(_user_tag, _tag, _rank, _id, _scope_id, _scope)

#define UCC_TL_UCP_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _user_tag, _tag,     \
                                 _src, _id, _scope_id, _scope)                 \
    do {                                                                       \
        ucc_assert((_tag) <= UCC_TL_UCP_MAX_TAG);                              \
        ucc_assert((_src) <= UCC_TL_UCP_MAX_SENDER);                           \
        ucc_assert((_id) <= UCC_TL_UCP_MAX_ID);                                \
        (_ucp_tag_mask) = (uint64_t)(-1);                                      \
        (_ucp_tag) =                                                           \
            UCC_TL_UCP_MAKE_TAG((_user_tag), (_tag), (_src), (_id),            \
                                (_scope_id), (_scope)); \
    } while (0)

#define UCC_TL_UCP_CHECK_REQ_STATUS()                                          \
    do {                                                                       \
        if (ucc_unlikely(UCS_PTR_IS_ERR(ucp_status))) {                        \
            tl_error(UCC_TL_TEAM_LIB(team),                                    \
                     "tag %u; dest %d; team_id %u; errmsg %s",                 \
                     task->tagged.tag, dest_group_rank,                        \
                     team->super.super.params.id,                              \
                     ucs_status_string(UCS_PTR_STATUS(ucp_status)));           \
            ucp_request_cancel(team->worker->ucp_worker, ucp_status);          \
            ucp_request_free(ucp_status);                                      \
            return ucs_status_to_ucc_status(UCS_PTR_STATUS(ucp_status));       \
        }                                                                      \
    } while (0)

static inline ucs_status_ptr_t
ucc_tl_ucp_send_common(void *buffer, size_t msglen, ucc_memory_type_t mtype,
                       ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                       ucc_tl_ucp_task_t *task, ucp_send_nbx_callback_t cb, void *user_data)
{
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucp_request_param_t req_param;
    ucc_status_t        status;
    ucp_ep_h            ep;
    ucp_tag_t           ucp_tag;

    status = ucc_tl_ucp_get_ep(team, dest_group_rank, &ep);
    if (ucc_unlikely(UCC_OK != status)) {
        return UCS_STATUS_PTR(UCS_ERR_NO_MESSAGE);
    }
    ucp_tag = UCC_TL_UCP_MAKE_SEND_TAG((args->mask & UCC_COLL_ARGS_FIELD_TAG),
        task->tagged.tag, UCC_TL_TEAM_RANK(team), team->super.super.params.id,
        team->super.super.params.scope_id, team->super.super.params.scope);
    req_param.op_attr_mask =
        UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
        UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    req_param.datatype    = ucp_dt_make_contig(msglen);
    req_param.cb.send     = cb;
    req_param.memory_type = ucc_memtype_to_ucs[mtype];
    req_param.user_data   = user_data;
    task->tagged.send_posted++;
    return ucp_tag_send_nbx(ep, buffer, 1, ucp_tag, &req_param);
}

static inline ucc_status_t ucc_tl_ucp_send_nb_st(void *buffer, size_t msglen,
                                                 ucc_memory_type_t mtype,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team,
                                                 ucc_tl_ucp_task_t *task)
{
    ucs_status_ptr_t ucp_status;

    ucp_status = ucc_tl_ucp_send_common(
        buffer, msglen, mtype, dest_group_rank, team, task,
        ucc_tl_ucp_send_completion_cb_st, (void *)task);
    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        ++task->tagged.send_completed;
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_send_nb_mt(void *buffer, size_t msglen,
                                                 ucc_memory_type_t mtype,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team,
                                                 ucc_tl_ucp_task_t *task)
{
    ucs_status_ptr_t ucp_status;

    ucp_status = ucc_tl_ucp_send_common(
        buffer, msglen, mtype, dest_group_rank, team, task,
        ucc_tl_ucp_send_completion_cb_mt, (void *)task);
    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        ucc_atomic_add32(&task->tagged.send_completed, 1);
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_send_nb(void *buffer, size_t msglen,
                                              ucc_memory_type_t mtype,
                                              ucc_rank_t        dest_group_rank,
                                              ucc_tl_ucp_team_t *team,
                                              ucc_tl_ucp_task_t *task)
{
    return UCC_TL_UCP_TEAM_CTX(team)->sendrecv_cbs.ucc_tl_ucp_send_nb(
        buffer, msglen, mtype, dest_group_rank, team, task);
}

static inline ucc_status_t
ucc_tl_ucp_send_cb(void *buffer, size_t msglen, ucc_memory_type_t mtype,
                   ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                   ucc_tl_ucp_task_t *task, ucp_send_nbx_callback_t cb, void *user_data)
{
    ucs_status_ptr_t ucp_status;

    ucp_status = ucc_tl_ucp_send_common(buffer, msglen, mtype, dest_group_rank,
                                        team, task, cb, user_data);
    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        cb(NULL, UCS_OK, user_data);
    }
    return UCC_OK;
}

static inline ucs_status_ptr_t
ucc_tl_ucp_recv_common(void *buffer, size_t msglen, ucc_memory_type_t mtype,
                       ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                       ucc_tl_ucp_task_t *task, ucp_tag_recv_nbx_callback_t cb, void *user_data)
{
    ucc_coll_args_t    *args = &TASK_ARGS(task);
    ucp_request_param_t req_param;
    ucp_tag_t           ucp_tag, ucp_tag_mask;

    // coverity[result_independent_of_operands:FALSE]
    UCC_TL_UCP_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask,
                             (args->mask & UCC_COLL_ARGS_FIELD_TAG),
                             task->tagged.tag, dest_group_rank,
                             team->super.super.params.id,
                             team->super.super.params.scope_id,
                             team->super.super.params.scope);
    req_param.op_attr_mask =
        UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_DATATYPE |
        UCP_OP_ATTR_FIELD_USER_DATA | UCP_OP_ATTR_FIELD_MEMORY_TYPE;
    req_param.datatype    = ucp_dt_make_contig(msglen);
    req_param.cb.recv     = cb;
    req_param.memory_type = ucc_memtype_to_ucs[mtype];
    req_param.user_data   = user_data;
    task->tagged.recv_posted++;
    return ucp_tag_recv_nbx(team->worker->ucp_worker, buffer, 1, ucp_tag,
                            ucp_tag_mask, &req_param);
}

static inline ucc_status_t ucc_tl_ucp_recv_nb_mt(void *buffer, size_t msglen,
                                                 ucc_memory_type_t mtype,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team,
                                                 ucc_tl_ucp_task_t *task)
{
    ucs_status_ptr_t ucp_status;

    ucp_status = ucc_tl_ucp_recv_common(
        buffer, msglen, mtype, dest_group_rank, team, task,
        ucc_tl_ucp_recv_completion_cb_mt, (void *)task);
    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        ucc_atomic_add32(&task->tagged.recv_completed, 1);
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_recv_nb_st(void *buffer, size_t msglen,
                                                 ucc_memory_type_t mtype,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team,
                                                 ucc_tl_ucp_task_t *task)
{
    ucs_status_ptr_t ucp_status;

    ucp_status = ucc_tl_ucp_recv_common(
        buffer, msglen, mtype, dest_group_rank, team, task,
        ucc_tl_ucp_recv_completion_cb_st, (void *)task);
    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        ++task->tagged.recv_completed;
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_recv_nb(void *buffer, size_t msglen,
                                              ucc_memory_type_t mtype,
                                              ucc_rank_t        dest_group_rank,
                                              ucc_tl_ucp_team_t *team,
                                              ucc_tl_ucp_task_t *task)
{
    return UCC_TL_UCP_TEAM_CTX(team)->sendrecv_cbs.ucc_tl_ucp_recv_nb(
        buffer, msglen, mtype, dest_group_rank, team, task);
}

static inline ucc_status_t
ucc_tl_ucp_recv_cb(void *buffer, size_t msglen, ucc_memory_type_t mtype,
                   ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                   ucc_tl_ucp_task_t *task, ucp_tag_recv_nbx_callback_t cb,
                   void *user_data)
{
    ucs_status_ptr_t ucp_status;

    ucp_status = ucc_tl_ucp_recv_common(buffer, msglen, mtype, dest_group_rank,
                                        team, task, cb, user_data);
    if (UCS_OK != ucp_status) {
        UCC_TL_UCP_CHECK_REQ_STATUS();
    } else {
        cb(NULL, UCS_OK, NULL, user_data);
    }
    return UCC_OK;
}

/* Non-Zero recv: if msglen == 0 then it is a no-op */
static inline ucc_status_t ucc_tl_ucp_recv_nz_mt(void *buffer, size_t msglen,
                                                 ucc_memory_type_t mtype,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team,
                                                 ucc_tl_ucp_task_t *task)
{
    if (msglen == 0) {
        task->tagged.recv_posted++;
        ucc_atomic_add32(&task->tagged.recv_completed, 1);
        return UCC_OK;
    }
    return ucc_tl_ucp_recv_nb_mt(buffer, msglen, mtype, dest_group_rank, team,
                                 task);
}

/* Non-Zero recv: if msglen == 0 then it is a no-op */
static inline ucc_status_t ucc_tl_ucp_recv_nz_st(void *buffer, size_t msglen,
                                                 ucc_memory_type_t mtype,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team,
                                                 ucc_tl_ucp_task_t *task)
{
    if (msglen == 0) {
        task->tagged.recv_posted++;
        task->tagged.recv_completed++;
        return UCC_OK;
    }
    return ucc_tl_ucp_recv_nb_st(buffer, msglen, mtype, dest_group_rank, team,
                                 task);
}

/* Non-Zero send: if msglen == 0 then it is a no-op */
static inline ucc_status_t ucc_tl_ucp_send_nz_mt(void *buffer, size_t msglen,
                                                 ucc_memory_type_t mtype,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team,
                                                 ucc_tl_ucp_task_t *task)
{
    if (msglen == 0) {
        task->tagged.send_posted++;
        ucc_atomic_add32(&task->tagged.send_completed, 1);
        return UCC_OK;
    }
    return ucc_tl_ucp_send_nb_mt(buffer, msglen, mtype, dest_group_rank, team,
                                 task);
}

/* Non-Zero send: if msglen == 0 then it is a no-op */
static inline ucc_status_t ucc_tl_ucp_send_nz_st(void *buffer, size_t msglen,
                                                 ucc_memory_type_t mtype,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team,
                                                 ucc_tl_ucp_task_t *task)
{
    if (msglen == 0) {
        task->tagged.send_posted++;
        task->tagged.send_completed++;
        return UCC_OK;
    }
    return ucc_tl_ucp_send_nb_st(buffer, msglen, mtype, dest_group_rank, team,
                                 task);
}

static inline ucc_status_t ucc_tl_ucp_send_nz(void *buffer, size_t msglen,
                                              ucc_memory_type_t mtype,
                                              ucc_rank_t        dest_group_rank,
                                              ucc_tl_ucp_team_t *team,
                                              ucc_tl_ucp_task_t *task)
{
    return UCC_TL_UCP_TEAM_CTX(team)->sendrecv_cbs.ucc_tl_ucp_send_nz(
        buffer, msglen, mtype, dest_group_rank, team, task);
}

static inline ucc_status_t ucc_tl_ucp_recv_nz(void *buffer, size_t msglen,
                                              ucc_memory_type_t mtype,
                                              ucc_rank_t        dest_group_rank,
                                              ucc_tl_ucp_team_t *team,
                                              ucc_tl_ucp_task_t *task)
{
    return UCC_TL_UCP_TEAM_CTX(team)->sendrecv_cbs.ucc_tl_ucp_recv_nz(
        buffer, msglen, mtype, dest_group_rank, team, task);
}

static inline ucc_status_t
ucc_tl_ucp_resolve_p2p_by_va(ucc_tl_ucp_team_t *team, void *va, ucp_ep_h *ep,
                             ucc_rank_t peer, uint64_t *rva, ucp_rkey_h *rkey,
                             int *segment)
{
    ucc_tl_ucp_context_t *ctx            = UCC_TL_UCP_TEAM_CTX(team);
    ptrdiff_t             key_offset     = 0;
    const size_t          section_offset = sizeof(uint64_t) * ctx->n_rinfo_segs;
    ucc_rank_t            core_rank;
    uint64_t             *rvas;
    uint64_t             *key_sizes;
    void                 *keys;
    void                 *offset;
    ptrdiff_t             base_offset;

    *segment  = -1;
    core_rank = ucc_ep_map_eval(UCC_TL_TEAM_MAP(team), peer);
    ucc_assert(UCC_TL_CORE_TEAM(team) != NULL);
    peer = ucc_get_ctx_rank(UCC_TL_CORE_TEAM(team), core_rank);

    offset = ucc_get_team_ep_addr(UCC_TL_CORE_CTX(team), UCC_TL_CORE_TEAM(team),
                                  core_rank, ucc_tl_ucp.super.super.id);

    base_offset = (ptrdiff_t)(TL_UCP_EP_ADDR_ONESIDED_INFO(offset, ctx));
    rvas        = (uint64_t *)base_offset;
    key_sizes   = PTR_OFFSET(base_offset, (section_offset * 2));
    keys        = PTR_OFFSET(base_offset, (section_offset * 3));

    for (int i = 0; i < ctx->n_rinfo_segs; i++) {
        uint64_t base = (uint64_t)team->va_base[i];
        uint64_t end = base + team->base_length[i];
        if ((uint64_t)va >= base &&
            (uint64_t)va < end) {
            *segment = i;
            break;
        }
        key_offset += key_sizes[i];
    }
    if (ucc_unlikely(0 > *segment)) {
        tl_error(UCC_TL_TEAM_LIB(team),
            "attempt to perform one-sided operation on non-registered memory %p", va);
        return UCC_ERR_NOT_FOUND;
    }
    if (ucc_unlikely(NULL == UCC_TL_UCP_REMOTE_RKEY(ctx, peer, *segment))) {
        ucs_status_t ucs_status =
            ucp_ep_rkey_unpack(*ep, PTR_OFFSET(keys, key_offset),
                               &UCC_TL_UCP_REMOTE_RKEY(ctx, peer, *segment));
        if (UCS_OK != ucs_status) {
            return ucs_status_to_ucc_status(ucs_status);
        }
    }
    *rkey = UCC_TL_UCP_REMOTE_RKEY(ctx, peer, *segment);
    *rva  = rvas[*segment] + ((uint64_t)va - (uint64_t)team->va_base[*segment]);
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_flush(ucc_tl_ucp_team_t *team)
{
    ucp_request_param_t req_param = {0};
    ucs_status_ptr_t    req;

    req = ucp_worker_flush_nbx(team->worker->ucp_worker, &req_param);
    if (UCS_OK != req) {
        if (UCS_PTR_IS_ERR(req)) {
            return ucs_status_to_ucc_status(UCS_PTR_STATUS(req));
        }
        ucp_request_free(req);
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_ep_flush(ucc_rank_t dest_group_rank,
                                               ucc_tl_ucp_team_t *team)
{
    ucp_request_param_t req_param = {0};
    ucc_status_t        status;
    ucs_status_ptr_t    req;
    ucp_ep_h            ep;

    status = ucc_tl_ucp_get_ep(team, dest_group_rank, &ep);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    req = ucp_ep_flush_nbx(ep, &req_param);
    if (UCS_OK != req) {
        if (UCS_PTR_IS_ERR(req)) {
            return ucs_status_to_ucc_status(UCS_PTR_STATUS(req));
        }
        ucp_request_free(req);
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_put_nb(void *buffer, void *target,
                                             size_t             msglen,
                                             ucc_rank_t         dest_group_rank,
                                             ucc_tl_ucp_team_t *team,
                                             ucc_tl_ucp_task_t *task)
{
    ucp_request_param_t req_param = {0};
    int                 segment   = 0;
    ucp_rkey_h          rkey      = NULL;
    uint64_t            rva       = 0;
    ucs_status_ptr_t    ucp_status;
    ucc_status_t        status;
    ucp_ep_h            ep;

    status = ucc_tl_ucp_get_ep(team, dest_group_rank, &ep);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    status = ucc_tl_ucp_resolve_p2p_by_va(team, target, &ep, dest_group_rank,
                                          &rva, &rkey, &segment);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    req_param.op_attr_mask =
        UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    req_param.cb.send   = ucc_tl_ucp_put_completion_cb;
    req_param.user_data = (void *)task;

    ucp_status = ucp_put_nbx(ep, buffer, msglen, rva, rkey, &req_param);

    task->onesided.put_posted++;
    if (UCS_OK != ucp_status) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            return ucs_status_to_ucc_status(UCS_PTR_STATUS(ucp_status));
        }
    } else {
        task->onesided.put_completed++;
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_get_nb(void *buffer, void *target,
                                             size_t             msglen,
                                             ucc_rank_t         dest_group_rank,
                                             ucc_tl_ucp_team_t *team,
                                             ucc_tl_ucp_task_t *task)
{
    ucp_request_param_t req_param = {0};
    int                 segment   = 0;
    ucp_rkey_h          rkey      = NULL;
    uint64_t            rva       = 0;
    ucs_status_ptr_t    ucp_status;
    ucc_status_t        status;
    ucp_ep_h            ep;

    status = ucc_tl_ucp_get_ep(team, dest_group_rank, &ep);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    status = ucc_tl_ucp_resolve_p2p_by_va(team, target, &ep, dest_group_rank,
                                          &rva, &rkey, &segment);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    req_param.op_attr_mask =
        UCP_OP_ATTR_FIELD_CALLBACK | UCP_OP_ATTR_FIELD_USER_DATA;
    req_param.cb.send   = ucc_tl_ucp_get_completion_cb;
    req_param.user_data = (void *)task;

    ucp_status = ucp_get_nbx(ep, buffer, msglen, rva, rkey, &req_param);

    task->onesided.get_posted++;
    if (UCS_OK != ucp_status) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            return ucs_status_to_ucc_status(UCS_PTR_STATUS(ucp_status));
        }
    } else {
        task->onesided.get_completed++;
    }

    return UCC_OK;
}

static inline ucc_status_t ucc_tl_ucp_atomic_inc(void *     target,
                                                 ucc_rank_t dest_group_rank,
                                                 ucc_tl_ucp_team_t *team)
{
    ucp_request_param_t req_param = {0};
    int                 segment   = 0;
    uint64_t            one       = 1;
    ucp_rkey_h          rkey      = NULL;
    uint64_t            rva       = 0;
    ucs_status_ptr_t    ucp_status;
    ucc_status_t        status;
    ucp_ep_h            ep;

    status = ucc_tl_ucp_get_ep(team, dest_group_rank, &ep);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    status = ucc_tl_ucp_resolve_p2p_by_va(team, target, &ep, dest_group_rank,
                                          &rva, &rkey, &segment);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    req_param.op_attr_mask = UCP_OP_ATTR_FIELD_DATATYPE;
    req_param.datatype     = ucp_dt_make_contig(sizeof(uint64_t));

    ucp_status = ucp_atomic_op_nbx(ep, UCP_ATOMIC_OP_ADD, &one, 1, rva, rkey,
                                   &req_param);

    if (UCS_OK != ucp_status) {
        if (UCS_PTR_IS_ERR(ucp_status)) {
            return ucs_status_to_ucc_status(UCS_PTR_STATUS(ucp_status));
        }
        ucp_request_free(ucp_status);
    }
    return UCC_OK;
}
/* Accept only UCC_OK */
#define UCPCHECK_GOTO(_cmd, _task, _label)                                     \
    do {                                                                       \
        ucc_status_t _status = (_cmd);                                         \
        if (UCC_OK != _status) {                                               \
            _task->super.status = _status;                                     \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#endif
