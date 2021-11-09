/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALL_H_
#define ALLTOALL_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

ucc_status_t ucc_tl_ucp_alltoall_init(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t      *team,
                                               ucc_coll_task_t     **task_h);

ucc_status_t ucc_tl_ucp_alltoall_pairwise_init_common(ucc_tl_ucp_task_t *task);

#define ALLTOALL_CHECK_INPLACE(_args, _team)                \
    do {                                                    \
        if (UCC_IS_INPLACE(_args)) {                        \
            tl_error(UCC_TL_TEAM_LIB(_team),                \
                     "inplace alltoall is not supported");  \
            status = UCC_ERR_NOT_SUPPORTED;                 \
            goto out;                                       \
        }                                                   \
    } while (0)

#define ALLTOALL_CHECK_USERDEFINED_DT(_args, _team  )             \
    do {                                                          \
        if (!UCC_DT_IS_PREDEFINED((_args).src.info.datatype) ||   \
            !UCC_DT_IS_PREDEFINED((_args).dst.info.datatype)) {   \
            tl_error(UCC_TL_TEAM_LIB(_team),                      \
                     "user defined datatype is not supported");   \
            status = UCC_ERR_NOT_SUPPORTED;                       \
            goto out;                                             \
        }                                                         \
    } while (0)

#define ALLTOALL_TASK_CHECK(_args, _team)              \
    ALLTOALL_CHECK_INPLACE((_args), (_team));          \
    ALLTOALL_CHECK_USERDEFINED_DT((_args), (_team));

#endif
