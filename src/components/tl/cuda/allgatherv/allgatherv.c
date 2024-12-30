/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv.h"
#include "components/mc/ucc_mc.h"

ucc_base_coll_alg_info_t
    ucc_tl_cuda_allgatherv_algs[UCC_TL_CUDA_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_TL_CUDA_ALLGATHERV_ALG_AUTO] =
            {.id   = UCC_TL_CUDA_ALLGATHERV_ALG_AUTO,
             .name = "auto",
             .desc = "choose allgatherv algorithm based on CUDA topology"},
        [UCC_TL_CUDA_ALLGATHERV_ALG_RING] =
            {.id   = UCC_TL_CUDA_ALLGATHERV_ALG_RING,
             .name = "ring",
             .desc = "multiring allgatherv algorithm"},
        [UCC_TL_CUDA_ALLGATHERV_ALG_LINEAR] =
            {.id   = UCC_TL_CUDA_ALLGATHERV_ALG_LINEAR,
             .name = "linear",
             .desc = "linear allgatherv algorithm"},
        [UCC_TL_CUDA_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

size_t ucc_tl_cuda_allgatherv_get_count(const ucc_tl_cuda_task_t *task,
                                        ucc_rank_t                block)
{
    const ucc_coll_args_t *args = &TASK_ARGS(task);

    return ucc_coll_args_get_count(args, args->dst.info_v.counts, block);
}

size_t ucc_tl_cuda_allgatherv_get_offset(const ucc_tl_cuda_task_t *task,
                                         ucc_rank_t                block)
{
    const ucc_coll_args_t *args = &TASK_ARGS(task);

    return ucc_coll_args_get_displacement(args, args->dst.info_v.displacements,
                                          block);
}

ucc_status_t ucc_tl_cuda_allgatherv_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *     tl_team,
                                         ucc_coll_task_t **    task_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);

    if (ucc_tl_cuda_team_topo_is_fully_connected(team->topo)) {
        return ucc_tl_cuda_allgatherv_linear_init(coll_args, tl_team, task_p);
    } else {
        return ucc_tl_cuda_allgatherv_ring_init(coll_args, tl_team, task_p);
    }
}
