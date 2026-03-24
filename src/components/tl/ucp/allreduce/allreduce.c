/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_string.h"

#define ALLREDUCE_MAX_PATTERN_SIZE 256

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allreduce_algs[UCC_TL_UCP_ALLREDUCE_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL,
             .name = "knomial",
             .desc =
                 "recursive knomial with arbitrary radix (optimized for latency)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL,
             .name = "sra_knomial",
             .desc = "recursive knomial scatter-reduce followed by knomial "
                     "allgather (optimized for BW)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_DBT] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_DBT,
             .name = "dbt",
             .desc = "allreduce over double binary tree where a leaf in one tree "
                     "will be intermediate in other (optimized for BW)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW,
             .name = "sliding_window",
             .desc = "sliding window allreduce (optimized for running on DPU)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_RING] =
            {.id   = UCC_TL_UCP_ALLREDUCE_ALG_RING,
             .name = "ring",
             .desc = "reduce-scatter ring followed by allgather ring "
                     "(topology-aware, optimized for BW)"},
        [UCC_TL_UCP_ALLREDUCE_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

char *ucc_tl_ucp_allreduce_score_str_get(ucc_tl_ucp_team_t *team)
{
    int                   max_size = ALLREDUCE_MAX_PATTERN_SIZE;
    char                 *str      = ucc_malloc(max_size * sizeof(char));
    ucc_tl_ucp_context_t *ctx      = UCC_TL_UCP_TEAM_CTX(team);
    uint64_t              cuda_types =
        ctx->ucp_memory_types &
        (UCC_BIT(UCC_MEMORY_TYPE_CUDA) |
         UCC_BIT(UCC_MEMORY_TYPE_CUDA_MANAGED));
    uint64_t  non_cuda_types = ctx->ucp_memory_types & (~cuda_types);
    char     *non_cuda_str;
    char     *cuda_str;

    if (team->cuda_ring && cuda_types) {
        cuda_str = ucc_malloc(max_size * sizeof(char));
        ucc_mtype_map_to_str(cuda_types, ",", cuda_str, max_size);
        if (non_cuda_types) {
            non_cuda_str = ucc_malloc(max_size * sizeof(char));
            ucc_mtype_map_to_str(non_cuda_types, ",", non_cuda_str, max_size);
            ucc_snprintf_safe(str, max_size,
                "allreduce:0-4k:@%d#allreduce:4k-inf:%s:@%d"
                "#allreduce:4k-inf:%s:@%d",
                UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL,
                cuda_str, UCC_TL_UCP_ALLREDUCE_ALG_RING,
                non_cuda_str, UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL);
            ucc_free(cuda_str);
            ucc_free(non_cuda_str);
            return str;
        }
        ucc_snprintf_safe(str, max_size,
            "allreduce:0-4k:@%d#allreduce:4k-inf:%s:@%d"
            "#allreduce:4k-inf:@%d",
            UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL,
            cuda_str, UCC_TL_UCP_ALLREDUCE_ALG_RING,
            UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL);
        ucc_free(cuda_str);
        return str;
    }

    ucc_snprintf_safe(str, max_size,
                      UCC_TL_UCP_ALLREDUCE_DEFAULT_ALG_SELECT_STR,
                      UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL);
    return str;
}

ucc_status_t ucc_tl_ucp_allreduce_init(ucc_tl_ucp_task_t *task)
{
    ucc_status_t status;
    ALLREDUCE_TASK_CHECK(TASK_ARGS(task), TASK_TEAM(task));
    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
out:
    return status;
}

ucc_status_t ucc_tl_ucp_allreduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);
    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    status = ucc_tl_ucp_allreduce_knomial_init_common(task);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_ucp_put_task(task);
    }
out:
    return status;
}
