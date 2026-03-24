/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "reduce_scatter.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_string.h"

#define REDUCE_SCATTER_MAX_PATTERN_SIZE 256

ucc_base_coll_alg_info_t
    ucc_tl_ucp_reduce_scatter_algs[UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST + 1] = {
        [UCC_TL_UCP_REDUCE_SCATTER_ALG_RING] =
            {.id   = UCC_TL_UCP_REDUCE_SCATTER_ALG_RING,
             .name = "ring",
             .desc = "O(N) ring"},
        [UCC_TL_UCP_REDUCE_SCATTER_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_REDUCE_SCATTER_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "recursive k-ing with arbitrary radix"},
        [UCC_TL_UCP_REDUCE_SCATTER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

char *ucc_tl_ucp_reduce_scatter_score_str_get(ucc_tl_ucp_team_t *team)
{
    int                   max_size = REDUCE_SCATTER_MAX_PATTERN_SIZE;
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
                "reduce_scatter:%s:@%d"
                "#reduce_scatter:%s:@%d",
                cuda_str, UCC_TL_UCP_REDUCE_SCATTER_ALG_RING,
                non_cuda_str, UCC_TL_UCP_REDUCE_SCATTER_ALG_KNOMIAL);
            ucc_free(cuda_str);
            ucc_free(non_cuda_str);
            return str;
        }
        ucc_snprintf_safe(str, max_size,
            "reduce_scatter:%s:@%d"
            "#reduce_scatter:@%d",
            cuda_str, UCC_TL_UCP_REDUCE_SCATTER_ALG_RING,
            UCC_TL_UCP_REDUCE_SCATTER_ALG_KNOMIAL);
        ucc_free(cuda_str);
        return str;
    }

    ucc_snprintf_safe(str, max_size,
                      UCC_TL_UCP_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR,
                      UCC_TL_UCP_REDUCE_SCATTER_ALG_KNOMIAL);
    return str;
}
