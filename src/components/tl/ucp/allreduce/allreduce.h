/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLREDUCE_H_
#define ALLREDUCE_H_
#include "../tl_ucp_reduce.h"

enum {
    UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL,
    UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL,
    UCC_TL_UCP_ALLREDUCE_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_allreduce_algs[UCC_TL_UCP_ALLREDUCE_ALG_LAST + 1];
ucc_status_t ucc_tl_ucp_allreduce_init(ucc_tl_ucp_task_t *task);

#define UCC_TL_UCP_ALLREDUCE_DEFAULT_ALG_SELECT_STR                            \
    "allreduce:0-4k:@0#allreduce:4k-inf:@1"

#define CHECK_SAME_MEMTYPE(_args, _team)                                       \
    do {                                                                       \
        if (!UCC_IS_INPLACE(_args) &&                                          \
            (_args.src.info.mem_type != _args.dst.info.mem_type)) {            \
            tl_error(UCC_TL_TEAM_LIB(_team),                                   \
                     "assymetric src/dst memory types are not supported yet"); \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while (0)

#define ALLREDUCE_TASK_CHECK(_args, _team)                                     \
    CHECK_SAME_MEMTYPE((_args), (_team));

ucc_status_t ucc_tl_ucp_allreduce_knomial_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_allreduce_knomial_init_common(ucc_tl_ucp_task_t *task);

ucc_status_t ucc_tl_ucp_allreduce_knomial_start(ucc_coll_task_t *task);

void ucc_tl_ucp_allreduce_knomial_progress(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allreduce_knomial_finalize(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_init(ucc_base_coll_args_t *coll_args,
                                                   ucc_base_team_t *     team,
                                                   ucc_coll_task_t **    task_h);

ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_start(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_progress(ucc_coll_task_t *task);

static inline int ucc_tl_ucp_allreduce_alg_from_str(const char *str)
{
    int i;
    for (i = 0; i < UCC_TL_UCP_ALLREDUCE_ALG_LAST; i++) {
        if (0 == strcasecmp(str, ucc_tl_ucp_allreduce_algs[i].name)) {
            break;
        }
    }
    return i;
}
#endif
