/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"

#define ALLGATHER_MAX_PATTERN_SIZE (sizeof(UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR))

/*--------------YAELIS FUNCTION---------------------*/



/*

ucc_tl_ucp_recv_nb(void *buffer, size_t msglen, ucc_memory_type_t mtype,
                   ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                   ucc_tl_ucp_task_t *task)

ucc_mc_memcpy(PTR_OFFSET(args->dst.info.buffer, offset), args->src.info.buffer,
         args->src.info.count * ucc_dt_size(args->src.info.datatype),
         args->dst.info.mem_type, args->src.info.mem_type);

ucc_status_t ucc_mc_memcpy(void *dst, const void *src, size_t len,
                           ucc_memory_type_t dst_mem,
                           ucc_memory_type_t src_mem);


ucc_tl_ucp_send_nb(void *buffer, size_t msglen, ucc_memory_type_t mtype,
                   ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                   ucc_tl_ucp_task_t *task)
*/

ucc_status_t new_ucp_tl_self_copy_nb(void *dst, void *src, size_t len, ucc_memory_type_t dst_mem,ucc_memory_type_t src_mem, ucc_rank_t rank, ucc_tl_ucp_team_t *team, ucc_tl_ucp_task_t *task){
    ucc_status_t status;
    status = ucc_tl_ucp_send_nb(src, len, src_mem, rank, team, task);
    // check here all occurances of returns (if this is ok)
    if (ucc_unlikely(UCC_OK != status)) {
                printf("\n allgather.c line 41 \n");
                task->super.status = status;
                return status;
            }
    status = ucc_tl_ucp_recv_nb(dst, len, dst_mem, rank, team, task);
    if (ucc_unlikely(UCC_OK != status)) {
                printf("\n allgather.c line 47 \n");
                task->super.status = status;
                return status;
            }
    return UCC_OK;
}

/*--------------YAELIS FUNCTION---------------------*/
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
