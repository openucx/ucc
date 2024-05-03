/**
 * Copyright(c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "allreduce_sliding_window.h"
#include "../allgather/allgather.h"
#include "utils/ucc_dt_reduce.h"
#include "tl_ucp_ep.h"

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_alloc_pipe(
    ucc_base_coll_args_t __attribute__((unused)) *coll_args,//NOLINT
    ucc_base_team_t __attribute__((unused)) *team,//NOLINT
    ucc_tl_ucp_task_t __attribute__((unused)) *task)//NOLINT
{
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_task_init(
    ucc_base_coll_args_t __attribute__((unused)) *coll_args,//NOLINT
    ucc_base_team_t __attribute__((unused)) *team,//NOLINT
    ucc_tl_ucp_task_t __attribute__((unused)) *task)//NOLINT
{
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allreduce_sliding_window_allgather_info_finalize(//NOLINT
   ucc_service_coll_req_t __attribute__((unused)) *scoll_req, //NOLINT
   ucc_tl_ucp_task_t __attribute__((unused)) *sw_task)//NOLINT
{
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sliding_window_free_gwbi(
    ucc_coll_task_t __attribute__((unused)) *coll_task)//NOLINT
{
    return UCC_OK;
}
