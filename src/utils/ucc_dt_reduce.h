/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_DT_REDUCE_H_
#define UCC_DT_REDUCE_H_

#include "components/ec/base/ucc_ec_base.h"
#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"

static inline ucc_status_t
ucc_dt_reduce_userdefined(void *src1, void *src2, void *dst, size_t n_vectors,
                          size_t count, size_t stride, ucc_dt_generic_t *dt)
{
    ucc_reduce_cb_params_t params = {.mask      = 0,
                                     .src1      = src1,
                                     .src2      = src2,
                                     .dst       = dst,
                                     .n_vectors = n_vectors,
                                     .count     = count,
                                     .stride    = stride,
                                     .dt        = dt};

    return dt->ops.reduce.cb(&params);
}

static inline ucc_status_t
ucc_dt_reduce_strided(void *src1, void *src2, void *dst, size_t n_vectors,
                      size_t count, size_t stride, ucc_datatype_t dt,
                      ucc_coll_args_t *args, uint16_t flags, double alpha,
                      ucc_ee_executor_t *exec, ucc_ee_executor_task_t **task)
{
    ucc_ee_executor_task_args_t eargs;

    if (count == 0 || n_vectors == 0) {
        *task = NULL;
        return UCC_OK;
    }
    if (!UCC_DT_IS_PREDEFINED(dt)) {
        ucc_assert(UCC_DT_HAS_REDUCE(dt));
        *task = NULL;
        return ucc_dt_reduce_userdefined(src1, src2, dst, n_vectors, count,
                                         stride, ucc_dt_to_generic(dt));
    } else {
        eargs.flags                 = flags;
        eargs.task_type             = UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED;
        eargs.reduce_strided.count  = count;
        eargs.reduce_strided.dt     = dt;
        eargs.reduce_strided.op     = args->op;
        eargs.reduce_strided.n_src2 = n_vectors;
        eargs.reduce_strided.dst    = dst;
        eargs.reduce_strided.src1   = src1;
        eargs.reduce_strided.src2   = src2;
        eargs.reduce_strided.stride = stride;
        eargs.reduce_strided.alpha  = alpha;

        return ucc_ee_executor_task_post(exec, &eargs, task);
    }
}

static inline ucc_status_t ucc_dt_reduce(void *src1, void *src2, void *dst,
                                         size_t count, ucc_datatype_t dt,
                                         ucc_coll_args_t *args, uint16_t flags,
                                         double alpha, ucc_ee_executor_t *exec,
                                         ucc_ee_executor_task_t **task)
{
    return ucc_dt_reduce_strided(src1, src2, dst, 1, count, 0, dt, args, flags,
                                 alpha, exec, task);
}

static inline ucc_status_t ucc_dt_reduce_multi(void **srcs, void *dst,
                                               size_t n_srcs,
                                               size_t count, ucc_datatype_t dt,
                                               ucc_coll_args_t *args,
                                               uint16_t flags, double alpha,
                                               ucc_ee_executor_t *exec,
                                               ucc_ee_executor_task_t **task)
{
    ucc_ee_executor_task_args_t eargs;

    ucc_assert(n_srcs <= UCC_EE_EXECUTOR_NUM_BUFS);
    if (count == 0 || n_srcs == 0) {
        *task = NULL;
        return UCC_OK;
    }

    if (!UCC_DT_IS_PREDEFINED(dt)) {
        ucc_assert(UCC_DT_HAS_REDUCE(dt));
        *task = NULL;
        return ucc_dt_reduce_userdefined(srcs, NULL, dst, n_srcs, count, 0,
                                         ucc_dt_to_generic(dt));
    } else {
        eargs.task_type = UCC_EE_EXECUTOR_TASK_REDUCE;
        eargs.flags = flags;
        eargs.reduce.alpha = alpha;
        eargs.reduce.count = count;
        eargs.reduce.dt = dt;
        eargs.reduce.op = args->op;
        eargs.reduce.dst = dst;
        eargs.reduce.n_srcs = n_srcs;
        for (size_t i = 0; i < n_srcs; i++) {
            eargs.reduce.srcs[i] = srcs[i];
        }

        return ucc_ee_executor_task_post(exec, &eargs, task);
    }
}

#endif
