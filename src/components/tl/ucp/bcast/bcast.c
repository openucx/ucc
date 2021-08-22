/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "bcast.h"

ucc_base_coll_alg_info_t
    ucc_tl_ucp_bcast_algs[UCC_TL_UCP_BCAST_ALG_LAST + 1] = {
        [UCC_TL_UCP_BCAST_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_BCAST_ALG_KNOMIAL,
             .name = "knomial",
             .desc =
                 "recursive k-ing with arbitrary radix (latency oriented alg)"},
        [UCC_TL_UCP_BCAST_ALG_SAG_KNOMIAL] =
            {.id   = UCC_TL_UCP_BCAST_ALG_SAG_KNOMIAL,
             .name = "sag_knomial",
             .desc = "recursive k-nomial scatter followed by k-nomial "
                     "allgather (bw oriented alg)"},
        [UCC_TL_UCP_BCAST_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_bcast_knomial_start(ucc_coll_task_t *task);
ucc_status_t ucc_tl_ucp_bcast_knomial_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_bcast_init(ucc_tl_ucp_task_t *task)
{
    task->super.post     = ucc_tl_ucp_bcast_knomial_start;
    task->super.progress = ucc_tl_ucp_bcast_knomial_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_bcast_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    status               = ucc_tl_ucp_bcast_init(task);
    *task_h              = &task->super;
    return status;
}
