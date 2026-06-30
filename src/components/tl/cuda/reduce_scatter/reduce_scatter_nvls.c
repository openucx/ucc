/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatterv/reduce_scatterv.h"
#include "reduce_scatter/reduce_scatter.h"

ucc_status_t ucc_tl_cuda_reduce_scatter_nvls_init(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *tl_team,
    ucc_coll_task_t **task_p)
{
    return ucc_tl_cuda_reduce_scatterv_nvls_init_common(
        coll_args,
        tl_team,
        task_p,
        ucc_tl_cuda_reduce_scatter_get_offset,
        ucc_tl_cuda_reduce_scatter_get_count);
}
