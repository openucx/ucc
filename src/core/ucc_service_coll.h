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
} ucc_service_coll_req_t;

ucc_status_t ucc_service_allreduce(ucc_team_t *team, void *sbuf, void *rbuf,
                                   ucc_datatype_t dt, size_t count,
                                   ucc_reduction_op_t op, ucc_subset_t subset,
                                   ucc_service_coll_req_t **req);

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
 * Create a non-blocking datatype-consistency wrapper for rooted collectives.
 *
 * Must be called for every gather/scatter collective on every rank when
 * check_asymmetric_dt is enabled, regardless of whether local ucc_coll_init
 * succeeded.  Ranks that failed init contribute a sentinel value to the
 * service allreduce so that all ranks obtain a uniform result at post/progress
 * time rather than hanging.
 *
 * @param team         Core UCC team.
 * @param coll_args    Original user collective arguments (always valid).
 * @param local_status Result of the local ucc_coll_init call.
 * @param task         Task returned by ucc_coll_init, or NULL if it failed.
 * @param status_out   Set to the error when NULL is returned.
 * @return Wrapper schedule task, or NULL on internal error.
 */
ucc_coll_task_t* ucc_service_dt_check(ucc_team_t            *team,
                                      const ucc_coll_args_t *coll_args,
                                      ucc_status_t           local_status,
                                      ucc_coll_task_t       *task,
                                      ucc_status_t          *status_out);

#endif
