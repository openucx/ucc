/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_REDUCE_H_
#define UCC_TL_UCP_REDUCE_H_
#include "tl_ucp_coll.h"
#include "components/mc/ucc_mc.h"

static inline ucc_status_t
ucc_tl_ucp_reduce_multi(void *src1, void *src2, void *dst, size_t n_vectors,
                        size_t count, size_t stride, ucc_datatype_t dt,
                        ucc_memory_type_t mem_type, ucc_tl_ucp_task_t *task,
                        int is_avg)
{
    if (is_avg) {
        return ucc_dt_reduce_multi_alpha(
            src1, src2, dst, n_vectors, count, stride, dt, UCC_OP_PROD,
            (double)1 / (double)UCC_TL_TEAM_SIZE(TASK_TEAM(task)), mem_type,
            &TASK_ARGS(task));
    }
    return ucc_dt_reduce_multi(src1, src2, dst, n_vectors, count, stride,
                               dt, mem_type, &TASK_ARGS(task));
}

#endif
