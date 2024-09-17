/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef ALLGATHER_H_
#define ALLGATHER_H_
#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"
/*--------------YAELIS FUNCTION---------------------*/
 #include "tl_ucp_sendrecv.h"



/*

ucc_status_t ucc_mc_memcpy(void *dst, const void *src, size_t len,
                           ucc_memory_type_t dst_mem,
                           ucc_memory_type_t src_mem);


ucc_tl_ucp_send_nb(void *buffer, size_t msglen, ucc_memory_type_t mtype,
                   ucc_rank_t dest_group_rank, ucc_tl_ucp_team_t *team,
                   ucc_tl_ucp_task_t *task)


#define NEW_MEMCPY(use_cuda, dst, src, len, dst_mem_type, src_mem_type, rank, team, task) do { \
    if (use_cuda) { \
        ucc_mc_memcpy(dst, src, len, dst_mem_type, src_mem_type); \
    } else { \
        new_ucp_tl_self_copy_nb(dst, src, len, dst_mem_type, src_mem_type, rank, team, task); \
    } \
} while (0)
*/


/*--------------YAELIS FUNCTION---------------------*/

enum {
    UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL,
    UCC_TL_UCP_ALLGATHER_ALG_RING,
    UCC_TL_UCP_ALLGATHER_ALG_NEIGHBOR,
    UCC_TL_UCP_ALLGATHER_ALG_BRUCK,
    UCC_TL_UCP_ALLGATHER_ALG_SPARBIT,
    UCC_TL_UCP_ALLGATHER_ALG_LAST
};

extern ucc_base_coll_alg_info_t
    ucc_tl_ucp_allgather_algs[UCC_TL_UCP_ALLGATHER_ALG_LAST + 1];

#define UCC_TL_UCP_ALLGATHER_DEFAULT_ALG_SELECT_STR                            \
    "allgather:0-4k:@0#allgather:4k-inf:@%d"

char *ucc_tl_ucp_allgather_score_str_get(ucc_tl_ucp_team_t *team);

static inline int ucc_tl_ucp_allgather_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_ALLGATHER_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_allgather_algs[i].name)) {
            break;
        }
    }
    return i;
}

/*--------------YAELIS FUNCTION---------------------*/
ucc_status_t new_ucp_tl_self_copy_nb(void *dst, void *src, size_t len, ucc_memory_type_t dst_mem,ucc_memory_type_t src_mem, ucc_rank_t rank, ucc_tl_ucp_team_t *team, ucc_tl_ucp_task_t *task);
/*--------------YAELIS FUNCTION---------------------*/

ucc_status_t ucc_tl_ucp_allgather_init(ucc_tl_ucp_task_t *task);

/* Ring */
ucc_status_t ucc_tl_ucp_allgather_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t      *team,
                                            ucc_coll_task_t     **task_h);

ucc_status_t ucc_tl_ucp_allgather_ring_init_common(ucc_tl_ucp_task_t *task);

void ucc_tl_ucp_allgather_ring_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_ring_start(ucc_coll_task_t *task);

/* Neighbor Exchange */
ucc_status_t ucc_tl_ucp_allgather_neighbor_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h);

void ucc_tl_ucp_allgather_neighbor_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_neighbor_start(ucc_coll_task_t *task);

/* Bruck */
ucc_status_t ucc_tl_ucp_allgather_bruck_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h);

void ucc_tl_ucp_allgather_bruck_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_bruck_start(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgather_bruck_finalize(ucc_coll_task_t *coll_task);

/* Sparbit */
ucc_status_t ucc_tl_ucp_allgather_sparbit_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h);

/* Uses allgather_kn_radix from config */
ucc_status_t ucc_tl_ucp_allgather_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t      *team,
                                               ucc_coll_task_t     **task_h);

/* Internal interface with custom radix */
ucc_status_t ucc_tl_ucp_allgather_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix);
#endif
