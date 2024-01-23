/**
 * Copyright(c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "../allgather/allgather.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_task_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *     team,
                                              ucc_tl_ucp_task_t *   task)
{
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(
    ucc_service_coll_req_t *scoll_req, ucc_tl_ucp_task_t *sw_task)
{
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_tl_ucp_task_t *   task)
{
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_free_gwbi(ucc_coll_task_t *coll_task)
{
    return UCC_OK;
}
