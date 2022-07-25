/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "ucc_service_coll.h"
#include "ucc_team.h"

uint64_t ucc_service_coll_map_cb(uint64_t ep, void *cb_ctx)
{
    ucc_service_coll_req_t *req  = cb_ctx;
    ucc_team_t             *team = req->team;
    ucc_rank_t              team_rank;

    team_rank = ucc_ep_map_eval(req->subset.map, (ucc_rank_t)ep);
    return ucc_ep_map_eval(team->ctx_map, team_rank);
}

static inline ucc_status_t
ucc_service_coll_req_init(ucc_team_t *team, ucc_subset_t *subset,
                          ucc_tl_team_t          **service_team,
                          ucc_service_coll_req_t **_req)
{
    ucc_context_t          *ctx = team->contexts[0];
    ucc_service_coll_req_t *req;

    *service_team = NULL;
    req = ucc_malloc(sizeof(*req), "service_req");
    if (!req) {
        ucc_error("failed to allocate %zd bytes for service coll req",
                  sizeof(*req));
        return UCC_ERR_NO_MEMORY;
    }
    req->team   = team;
    req->subset = *subset;

    if (ctx->service_team) {
        *service_team         = ctx->service_team;
        subset->map.type      = UCC_EP_MAP_CB;
        subset->map.cb.cb     = ucc_service_coll_map_cb;
        subset->map.cb.cb_ctx = req;
    } else {
        ucc_assert(team->service_team);
        *service_team = team->service_team;
    }

    *_req = req;
    return UCC_OK;
}

ucc_status_t ucc_service_allreduce(ucc_team_t *team, void *sbuf, void *rbuf,
                                   ucc_datatype_t dt, size_t count,
                                   ucc_reduction_op_t op, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req)
{
    ucc_tl_team_t  *steam;
    ucc_tl_iface_t *tl_iface;
    ucc_status_t    status;

    status = ucc_service_coll_req_init(team, &subset, &steam, req);
    if (UCC_OK != status) {
        return status;
    }

    tl_iface = UCC_TL_TEAM_IFACE(steam);
    status = tl_iface->scoll.allreduce(&steam->super, sbuf, rbuf, dt, count, op,
                                       subset, &(*req)->task);
    if (status < 0) {
        ucc_free(*req);
        ucc_error("failed to start service allreduce for team %p: %s", team,
                  ucc_status_string(status));
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_service_allgather(ucc_team_t *team, void *sbuf, void *rbuf,
                                   size_t msgsize, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req)
{
    ucc_tl_team_t  *steam;
    ucc_tl_iface_t *tl_iface;
    ucc_status_t    status;

    status = ucc_service_coll_req_init(team, &subset, &steam, req);
    if (UCC_OK != status) {
        return status;
    }

    tl_iface = UCC_TL_TEAM_IFACE(steam);
    status   = tl_iface->scoll.allgather(&steam->super, sbuf, rbuf, msgsize,
                                       subset, &(*req)->task);
    if (status < 0) {
        ucc_free(*req);
        ucc_error("failed to start service allreduce for team %p: %s", team,
                  ucc_status_string(status));
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_service_bcast(ucc_team_t *team, void *buf, size_t msgsize,
                               ucc_rank_t root, ucc_subset_t subset,
                               ucc_service_coll_req_t **req)
{
    ucc_tl_team_t  *steam;
    ucc_tl_iface_t *tl_iface;
    ucc_status_t    status;

    status = ucc_service_coll_req_init(team, &subset, &steam, req);
    if (UCC_OK != status) {
        return status;
    }

    tl_iface = UCC_TL_TEAM_IFACE(steam);
    status = tl_iface->scoll.bcast(&steam->super, buf, msgsize,
                                   root, subset, &(*req)->task);
    if (status < 0) {
        ucc_free(*req);
        ucc_error("failed to start service bcast for team %p: %s", team,
                  ucc_status_string(status));
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_service_coll_test(ucc_service_coll_req_t *req)
{
    ucc_status_t status;

    status = ucc_collective_test(&req->task->super);
    if (UCC_INPROGRESS == status) {
        ucc_context_progress(req->team->contexts[0]);
    }
    return status;
}

ucc_status_t ucc_service_coll_finalize(ucc_service_coll_req_t *req)
{
    ucc_status_t status;

    status = ucc_collective_finalize(&req->task->super);
    ucc_free(req);
    return status;
}

typedef struct ucc_internal_oob_coll_info {
    ucc_team_t  *team;
    ucc_subset_t subset;
} ucc_internal_oob_coll_info_t;

static ucc_status_t ucc_internal_oob_allgather(void *sbuf, void *rbuf,
                                               size_t size, void *coll_info,
                                               void **request)
{
    ucc_internal_oob_coll_info_t *ci  = coll_info;
    ucc_service_coll_req_t       *req = NULL;
    ucc_status_t                  status;

    status =
        ucc_service_allgather(ci->team, sbuf, rbuf, size, ci->subset, &req);
    *request = (void *)req;
    return status;
}

static ucc_status_t ucc_internal_oob_test(void *request)
{
    ucc_service_coll_req_t *req = request;
    return ucc_service_coll_test(req);
}

static ucc_status_t ucc_internal_oob_free(void *request)
{
    ucc_service_coll_req_t *req = request;
    return ucc_service_coll_finalize(req);
}

ucc_status_t ucc_internal_oob_init(ucc_team_t *team, ucc_subset_t subset,
                                   ucc_team_oob_coll_t *oob)
{
    ucc_internal_oob_coll_info_t *ci;

    ci = ucc_malloc(sizeof(*ci), "internal_coll_info");
    if (!ci) {
        ucc_error("failed to allocate %zd bytes for internal_coll_info",
                  sizeof(*ci));
        return UCC_ERR_NO_MEMORY;
    }

    ci->team       = team;
    ci->subset     = subset;
    oob->coll_info = ci;
    oob->allgather = ucc_internal_oob_allgather;
    oob->req_test  = ucc_internal_oob_test;
    oob->req_free  = ucc_internal_oob_free;
    oob->n_oob_eps = (uint32_t)subset.map.ep_num;
    oob->oob_ep    = (uint32_t)subset.myrank;

    return UCC_OK;
}

void ucc_internal_oob_finalize(ucc_team_oob_coll_t *oob)
{
    ucc_free(oob->coll_info);
}
