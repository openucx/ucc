/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "../../../utils/ucc_string.h"
#include "../../../utils/ucc_log.h"
#include "../../../utils/ucc_coll_utils.h"

#define ALLGATHER_MAX_PATTERN_SIZE                                             \
    (sizeof(UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR_1PPN) * 2)

ucc_base_coll_alg_info_t
    ucc_tl_ucp_allgather_algs[UCC_TL_UCP_ALLGATHER_ALG_LAST + 1] = {
        [UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL,
             .name = "knomial",
             .desc = "recursive k-ing with arbitrary radix"},
        [UCC_TL_UCP_ALLGATHER_ALG_RING] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_RING,
             .name = "ring",
             .desc = "O(N) Ring"},
        [UCC_TL_UCP_ALLGATHER_ALG_NEIGHBOR] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_NEIGHBOR,
             .name = "neighbor",
             .desc = "O(N) Neighbor Exchange N/2 steps"},
        [UCC_TL_UCP_ALLGATHER_ALG_BRUCK] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_BRUCK,
             .name = "bruck",
             .desc = "O(log(N)) Variation of Bruck algorithm"},
        [UCC_TL_UCP_ALLGATHER_ALG_SPARBIT] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_SPARBIT,
             .name = "sparbit",
             .desc = "O(log(N)) SPARBIT algorithm"},
        [UCC_TL_UCP_ALLGATHER_ALG_LINEAR] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_LINEAR,
             .name = "linear",
             .desc = "O(N - 1) Linear algorithm, one-shot"},
        [UCC_TL_UCP_ALLGATHER_ALG_LINEAR_BATCHED] =
            {.id   = UCC_TL_UCP_ALLGATHER_ALG_LINEAR_BATCHED,
             .name = "batched",
             .desc = "O(N - 1) Linear algorithm, K-send/receive in flight"},
        [UCC_TL_UCP_ALLGATHER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_allgather_init(ucc_tl_ucp_task_t *task)
{
    return ucc_tl_ucp_allgather_ring_init_common(task);
}

char *ucc_tl_ucp_allgather_score_str_get(ucc_tl_ucp_team_t *team)
{
    int                   max_size = ALLGATHER_MAX_PATTERN_SIZE;
    int                   algo_num = UCC_TL_TEAM_SIZE(team) % 2
                                         ? UCC_TL_UCP_ALLGATHER_ALG_RING
                                         : UCC_TL_UCP_ALLGATHER_ALG_NEIGHBOR;
    char *                str      = ucc_malloc(max_size * sizeof(char));
    ucc_tl_ucp_context_t *ctx      = UCC_TL_UCP_TEAM_CTX(team);
    uint64_t              cuda_types =
        ctx->ucp_memory_types &
        (UCC_BIT(UCC_MEMORY_TYPE_CUDA) | UCC_BIT(UCC_MEMORY_TYPE_CUDA_MANAGED));
    uint64_t    non_cuda_types = ctx->ucp_memory_types & (~cuda_types);
    ucc_sbgp_t *sbgp;
    char *      non_cuda_str;
    char *      cuda_str;

    if (team->cfg.use_reordering) {
        sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        if (!ucc_ep_map_is_identity(&sbgp->map)) {
            algo_num = UCC_TL_UCP_ALLGATHER_ALG_RING;
        }
    }

    if (team->topo && ucc_topo_is_single_ppn(team->topo)) {
        if (cuda_types) {
            cuda_str = ucc_malloc(max_size * sizeof(char));
            ucc_mtype_map_to_str(cuda_types, ",", cuda_str, max_size);
            if (non_cuda_types) {
                non_cuda_str = ucc_malloc(max_size * sizeof(char));
                ucc_mtype_map_to_str(non_cuda_types, ",", non_cuda_str,
                                     max_size);
                ucc_snprintf_safe(
                    str, max_size,
                    UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR_1PPN, cuda_str,
                    non_cuda_str, algo_num);
                ucc_free(cuda_str);
                ucc_free(non_cuda_str);
                return str;
            }
            ucc_snprintf_safe(
                str, max_size,
                UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR_1PPN_CUDA, cuda_str,
                algo_num);
            ucc_free(cuda_str);
            return str;
        }
    }
    ucc_snprintf_safe(str, max_size,
                      UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR, algo_num);
    return str;
}
