/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"

#define ALLGATHER_MAX_PATTERN_SIZE (sizeof(UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR))

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
        [UCC_TL_UCP_ALLGATHER_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

ucc_status_t ucc_tl_ucp_allgather_init(ucc_tl_ucp_task_t *task)
{
    return ucc_tl_ucp_allgather_ring_init_common(task);
}

char *ucc_tl_ucp_allgather_score_str_get(ucc_tl_ucp_team_t *team)
{
    int   max_size = ALLGATHER_MAX_PATTERN_SIZE;
    int   algo_num = UCC_TL_TEAM_SIZE(team) % 2
                         ? UCC_TL_UCP_ALLGATHER_ALG_RING
                         : UCC_TL_UCP_ALLGATHER_ALG_NEIGHBOR;
    char *str      = ucc_malloc(max_size * sizeof(char));
    ucc_sbgp_t *sbgp;

    if (team->cfg.use_reordering) {
        sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        if (!ucc_ep_map_is_identity(&sbgp->map)) {
            algo_num = UCC_TL_UCP_ALLGATHER_ALG_RING;
        }
    }
    ucc_snprintf_safe(str, max_size,
                      UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR, algo_num);
    return str;
}
