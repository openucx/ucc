/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "cl_hier.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_team.h"
#include "utils/ucc_coll_utils.h"
#include "cl_hier_coll.h"

const char *
    ucc_cl_hier_default_alg_select_str[UCC_CL_HIER_N_DEFAULT_ALG_SELECT_STR] = {
        UCC_CL_HIER_ALLREDUCE_DEFAULT_ALG_SELECT_STR};

ucc_status_t ucc_cl_hier_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t      *team,
                                   ucc_coll_task_t     **task)
{
    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_cl_hier_allreduce_rab_init(coll_args, team, task);
    case UCC_COLL_TYPE_BARRIER:
        return ucc_cl_hier_barrier_init(coll_args, team, task);
    case UCC_COLL_TYPE_ALLTOALL:
        return ucc_cl_hier_alltoall_init(coll_args, team, task);
    case UCC_COLL_TYPE_ALLTOALLV:
        return ucc_cl_hier_alltoallv_init(coll_args, team, task);
    default:
        cl_error(team->context->lib, "coll_type %s is not supported",
                 ucc_coll_type_str(coll_args->args.coll_type));
        break;
    }
    return UCC_ERR_NOT_SUPPORTED;
}

static inline int alg_id_from_str(ucc_coll_type_t coll_type, const char *str)
{
    switch (coll_type) {
    case UCC_COLL_TYPE_ALLTOALLV:
        return ucc_cl_hier_alltoallv_alg_from_str(str);
    case UCC_COLL_TYPE_ALLTOALL:
        return ucc_cl_hier_alltoall_alg_from_str(str);
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_cl_hier_allreduce_alg_from_str(str);
    default:
        break;
    }
    return -1;
}

ucc_status_t ucc_cl_hier_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t   coll_type,
                                        ucc_memory_type_t mem_type, //NOLINT
                                        ucc_base_coll_init_fn_t *init)
{
    ucc_status_t status = UCC_OK;
    if (alg_id_str) {
        alg_id = alg_id_from_str(coll_type, alg_id_str);
    }

    switch (coll_type) {
    case UCC_COLL_TYPE_ALLTOALLV:
        switch (alg_id) {
        case UCC_CL_HIER_ALLTOALLV_ALG_NODE_SPLIT:
            *init = ucc_cl_hier_alltoallv_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        switch (alg_id) {
        case UCC_CL_HIER_ALLTOALL_ALG_NODE_SPLIT:
            *init = ucc_cl_hier_alltoall_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        switch (alg_id) {
        case UCC_CL_HIER_ALLREDUCE_ALG_RAB:
            *init = ucc_cl_hier_allreduce_rab_init;
            break;
        case UCC_CL_HIER_ALLREDUCE_ALG_SPLIT_RAIL:
            *init = ucc_cl_hier_allreduce_split_rail_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
        break;
    }
    return status;
}
