/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce_scatter.h"
#include "components/mc/ucc_mc.h"

ucc_base_coll_alg_info_t
ucc_tl_cuda_reduce_scatter_algs[UCC_TL_CUDA_REDUCE_SCATTER_ALG_LAST + 1] = {
        [UCC_TL_CUDA_REDUCE_SCATTER_ALG_AUTO] =
            {.id   = UCC_TL_CUDA_REDUCE_SCATTER_ALG_AUTO,
             .name = "auto",
             .desc = "choose reduce scatter algorithm based on CUDA topology"},
        [UCC_TL_CUDA_REDUCE_SCATTER_ALG_RING] =
            {.id   = UCC_TL_CUDA_REDUCE_SCATTER_ALG_RING,
             .name = "ring",
             .desc = "multiring reduce scatter algorithm"},
        [UCC_TL_CUDA_REDUCE_SCATTER_ALG_LINEAR] =
            {.id   = UCC_TL_CUDA_REDUCE_SCATTER_ALG_LINEAR,
             .name = "linear",
             .desc = "linear reduce scatter algorithm"},
        [UCC_TL_CUDA_REDUCE_SCATTER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

size_t ucc_tl_cuda_reduce_scatter_get_count(const ucc_tl_cuda_task_t *task,
                                            ucc_rank_t block) //NOLINT
{
    const ucc_coll_args_t *args  = &TASK_ARGS(task);
    size_t                 count = args->dst.info.count;

    if (UCC_IS_INPLACE(*args)) {
        count = args->dst.info.count / UCC_TL_TEAM_SIZE(TASK_TEAM(task));
    }
    return count;
}

size_t ucc_tl_cuda_reduce_scatter_get_offset(const ucc_tl_cuda_task_t *task,
                                             ucc_rank_t block)
{
    return ucc_tl_cuda_reduce_scatter_get_count(task, block) * block;
}

ucc_status_t ucc_tl_cuda_reduce_scatter_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *tl_team,
                                             ucc_coll_task_t **task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);

    if (ucc_tl_cuda_team_topo_is_fully_connected(team->topo)) {
        return ucc_tl_cuda_reduce_scatter_linear_init(coll_args, tl_team,
                                                      task_p);
    } else {
        return ucc_tl_cuda_reduce_scatter_ring_init(coll_args, tl_team, task_p);
    }
}
