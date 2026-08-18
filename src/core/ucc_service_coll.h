/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_SERVICE_COLL_H_
#define UCC_SERVICE_COLL_H_

#include "ucc/api/ucc.h"
#include "components/tl/ucc_tl.h"

typedef struct ucc_service_coll_req {
    ucc_coll_task_t *task;
    ucc_team_t      *team;
    void *           data;
    ucc_subset_t     subset;
    uint8_t embedded; /* 1: req storage is caller-owned, not heap; test/
                         finalize must not ucc_free it (e.g. the vote req) */
} ucc_service_coll_req_t;

ucc_status_t ucc_service_allreduce(ucc_team_t *team, void *sbuf, void *rbuf,
                                   ucc_datatype_t dt, size_t count,
                                   ucc_reduction_op_t op, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req);

/**
 * Member-scoped service allreduce whose @subset map already resolves each
 * agreement-member index directly to a CONTEXT rank (built from params.ep_map),
 * so it does NOT depend on the team's ctx_map being materialized. Runs over the
 * persistent ctx->service_team (which must exist) and writes into the
 * caller-provided embedded @req storage (never heap-freed). Used by the
 * cache-action agreement vote before ADDR_EXCHANGE has built the team. A post
 * failure must be treated as fatal by the caller (a silent local fallback would
 * re-introduce cross-rank divergence). */
ucc_status_t ucc_service_allreduce_ctx(
    ucc_team_t *team, ucc_service_coll_req_t *req, void *sbuf, void *rbuf,
    ucc_datatype_t dt, size_t count, ucc_reduction_op_t op,
    ucc_subset_t subset);

ucc_status_t ucc_service_allgather(ucc_team_t *team, void *sbuf, void *rbuf,
                                   size_t msgsize, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req);

ucc_status_t ucc_service_bcast(ucc_team_t *team, void *buf, size_t msgsize,
                               ucc_rank_t root, ucc_subset_t subset,
                               ucc_service_coll_req_t **req);

ucc_status_t ucc_service_coll_test(ucc_service_coll_req_t *req);

ucc_status_t ucc_service_coll_finalize(ucc_service_coll_req_t *req);

ucc_status_t ucc_internal_oob_init(ucc_team_t *team, ucc_subset_t subset,
                                   ucc_team_oob_coll_t *oob);

void ucc_internal_oob_finalize(ucc_team_oob_coll_t *oob);

ucc_status_t ucc_collective_finalize_internal(ucc_coll_task_t *task);

/**
 * Create datatype validation schedule for rooted collectives
 *
 * This function checks if datatype validation is needed and creates a schedule
 * with validation logic if required. If validation is not needed, returns the
 * original task unchanged.
 *
 * @param team The UCC team
 * @param task The actual collective task (already created by TL/CL)
 * @param status_out If non-NULL, set to the error status when returning NULL
 * @return Schedule with validation (as ucc_coll_task_t*), or original task, or NULL on error
 */
ucc_coll_task_t* ucc_service_dt_check(ucc_team_t *team, ucc_coll_task_t *task,
                                      ucc_status_t *status_out);

#endif
