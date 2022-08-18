/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_SERVICE_COLL_H_
#define UCC_SERVICE_COLL_H_

#include "ucc/api/ucc.h"
#include "components/tl/ucc_tl.h"

typedef struct ucc_service_coll_req {
    ucc_coll_task_t *task;
    ucc_team_t      *team;
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
#endif
