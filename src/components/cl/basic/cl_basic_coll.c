/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "cl_basic.h"
#include "utils/ucc_coll_utils.h"

ucc_status_t ucc_cl_basic_select_tl(ucc_base_coll_args_t *coll_args,
                                    ucc_cl_basic_team_t *cl_basic_team,
                                    ucc_tl_team_t **tl_team)
{
    ucc_coll_args_t *args = &coll_args->args;
    /* If NCCL TL is available and both src and dst buffers are CUDA then
       choose NCCL TL for alltoall and alltoallv otherwise use UCP TL*/
    if (cl_basic_team->tl_nccl_team != NULL) {
        switch (args->coll_type) {
        case UCC_COLL_TYPE_ALLGATHER:
        case UCC_COLL_TYPE_ALLREDUCE:
        case UCC_COLL_TYPE_ALLTOALL:
            if ((args->dst.info.mem_type == UCC_MEMORY_TYPE_CUDA) &&
                (UCC_IS_INPLACE(*args) ||
                 (args->src.info.mem_type == UCC_MEMORY_TYPE_CUDA))) {
                *tl_team = cl_basic_team->tl_nccl_team;
                return UCC_OK;
            }
            break;
        case UCC_COLL_TYPE_ALLGATHERV:
            if ((args->dst.info_v.mem_type == UCC_MEMORY_TYPE_CUDA) &&
                (UCC_IS_INPLACE(*args) ||
                 (args->src.info.mem_type == UCC_MEMORY_TYPE_CUDA))) {
                *tl_team = cl_basic_team->tl_nccl_team;
                return UCC_OK;
            }
            break;
        case UCC_COLL_TYPE_ALLTOALLV:
            if ((args->dst.info_v.mem_type == UCC_MEMORY_TYPE_CUDA) &&
                (UCC_IS_INPLACE(*args) ||
                 (args->src.info_v.mem_type == UCC_MEMORY_TYPE_CUDA))) {
                *tl_team = cl_basic_team->tl_nccl_team;
                return UCC_OK;
            }
            break;
        case UCC_COLL_TYPE_BCAST:
            if (args->src.info.mem_type == UCC_MEMORY_TYPE_CUDA) {
                *tl_team = cl_basic_team->tl_nccl_team;
                return UCC_OK;
            }
            break;
        default:
            *tl_team = cl_basic_team->tl_ucp_team;
            return UCC_OK;
        }
    }
    *tl_team = cl_basic_team->tl_ucp_team;
    return UCC_OK;
}

ucc_status_t ucc_cl_basic_coll_init(ucc_base_coll_args_t *coll_args,
                                    ucc_base_team_t *team,
                                    ucc_coll_task_t **task)
{
    ucc_cl_basic_team_t *cl_team = ucc_derived_of(team, ucc_cl_basic_team_t);
    ucc_tl_team_t *tl_team;

    ucc_cl_basic_select_tl(coll_args, cl_team, &tl_team);
    return UCC_TL_TEAM_IFACE(tl_team)->coll.init(coll_args, &tl_team->super,
                                                 task);
}
