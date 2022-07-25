/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_REDUCE_H_
#define UCC_TL_UCP_REDUCE_H_
#include "tl_ucp_coll.h"
#include "utils/ucc_dt_reduce.h"

static inline ucc_status_t
ucc_tl_ucp_reduce_multi(void *src1, void *src2, void *dst, size_t n_vectors,
                        size_t count, size_t stride, ucc_datatype_t dt,
                        ucc_memory_type_t mem_type, ucc_tl_ucp_task_t *task,
                        int is_avg)
{
    if (count == 0) {
        return UCC_OK;
    }
    if (is_avg) {
        return ucc_dt_reduce_multi_alpha(
            src1, src2, dst, n_vectors, count, stride, dt, UCC_OP_PROD,
            (double)1 / (double)UCC_TL_TEAM_SIZE(TASK_TEAM(task)), mem_type,
            &TASK_ARGS(task));
    }
    return ucc_dt_reduce_multi(src1, src2, dst, n_vectors, count, stride,
                               dt, mem_type, &TASK_ARGS(task));
}

static inline ucc_status_t
ucc_tl_ucp_reduce_multi_nb(void *src1, void *src2, void *dst, size_t n_vectors,
                           size_t count, size_t stride, ucc_datatype_t dt,
                           ucc_tl_ucp_task_t *task, int is_avg,
                           ucc_ee_executor_t *exec,
                           ucc_ee_executor_task_t **etask)
{
    if (count == 0) {
        *etask = NULL;
        return UCC_OK;
    }

    if (is_avg) {
        return ucc_dt_reduce_multi_alpha_nb(
            src1, src2, dst, n_vectors, count, stride, dt,
            (double)1 / (double)UCC_TL_TEAM_SIZE(TASK_TEAM(task)),
            &TASK_ARGS(task), exec, etask);
    }

    return ucc_dt_reduce_multi_nb(src1, src2, dst, n_vectors, count, stride,
                                  dt, &TASK_ARGS(task), exec, etask);
}

#endif
