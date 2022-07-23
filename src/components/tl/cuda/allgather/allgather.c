/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgather.h"
#include "components/mc/ucc_mc.h"

ucc_base_coll_alg_info_t
    ucc_tl_cuda_allgather_algs[UCC_TL_CUDA_ALLGATHER_ALG_LAST + 1] = {
        [UCC_TL_CUDA_ALLGATHER_ALG_AUTO] =
            {.id   = UCC_TL_CUDA_ALLGATHER_ALG_AUTO,
             .name = "auto",
             .desc = "choose allgather algorithm based on CUDA topology"},
        [UCC_TL_CUDA_ALLGATHER_ALG_RING] =
            {.id   = UCC_TL_CUDA_ALLGATHER_ALG_RING,
             .name = "ring",
             .desc = "multiring allgather algorithm"},
        [UCC_TL_CUDA_ALLGATHER_ALG_LINEAR] =
            {.id   = UCC_TL_CUDA_ALLGATHER_ALG_LINEAR,
             .name = "linear",
             .desc = "linear allgather algorithm"},
        [UCC_TL_CUDA_ALLGATHER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

size_t ucc_tl_cuda_allgather_get_count(const ucc_tl_cuda_task_t *task,
                                       ucc_rank_t                block)
{
    return TASK_ARGS(task).dst.info.count / UCC_TL_TEAM_SIZE(TASK_TEAM(task));
}

size_t ucc_tl_cuda_allgather_get_offset(const ucc_tl_cuda_task_t *task,
                                        ucc_rank_t                block)
{
    return (TASK_ARGS(task).dst.info.count /
            UCC_TL_TEAM_SIZE(TASK_TEAM(task))) *
           block;
}

ucc_status_t ucc_tl_cuda_allgather_init(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *tl_team,
                                        ucc_coll_task_t **task_p)
{
    return ucc_tl_cuda_allgather_ring_init(coll_args, tl_team, task_p);
}
