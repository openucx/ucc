/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_DT_REDUCE_H_
#define UCC_DT_REDUCE_H_

#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"

static inline ucc_status_t
ucc_dt_reduce(void *src1, void *src2, void *dst, size_t count,
              ucc_datatype_t dt, ucc_memory_type_t mem_type,
              ucc_coll_args_t *args)
{
    if (UCC_DT_IS_PREDEFINED(dt)) {
        return ucc_mc_reduce(src1, src2, dst, count,
                             dt, args->op, mem_type);
    } else {
        ucc_assert(UCC_DT_HAS_REDUCE(dt));
        return ucc_mc_reduce_userdefined(src1, src2, dst, 1, count,
                                         0, ucc_dt_to_generic(dt));
    }
}

static inline ucc_status_t
ucc_dt_reduce_nb(void *src1, void *src2, void *dst, size_t count,
                 ucc_datatype_t dt, ucc_coll_args_t *args,
                 ucc_ee_executor_t *exec, ucc_ee_executor_task_t **task)
{
    ucc_ee_executor_task_args_t eargs;

    if (!UCC_DT_IS_PREDEFINED(dt)) {
        ucc_assert(UCC_DT_HAS_REDUCE(dt));
        *task = NULL;
        return ucc_mc_reduce_userdefined(src1, src2, dst, 1, count,
                                         0, ucc_dt_to_generic(dt));
    } else {
        eargs.task_type = UCC_EE_EXECUTOR_TASK_TYPE_REDUCE;
        eargs.bufs[0]   = dst;
        eargs.bufs[1]   = src1;
        eargs.bufs[2]   = src2;
        eargs.count     = count;
        eargs.dt        = dt;
        eargs.op        = args->op;

        return ucc_ee_executor_task_post(exec, &eargs, task);
    }
}

static inline ucc_status_t
ucc_dt_reduce_multi(void *src1, void *src2, void *dst, size_t n_vectors,
                    size_t count, size_t stride, ucc_datatype_t dt,
                    ucc_memory_type_t mem_type, ucc_coll_args_t *args)
{
    if (!UCC_DT_IS_PREDEFINED(dt)) {
        ucc_assert(UCC_DT_HAS_REDUCE(dt));
        return ucc_mc_reduce_userdefined(src1, src2, dst, n_vectors, count,
                                         stride, ucc_dt_to_generic(dt));
    } else {
        return ucc_mc_reduce_multi(src1, src2, dst, n_vectors, count, stride,
                                   dt, args->op, mem_type);
    }
}

static inline ucc_status_t
ucc_dt_reduce_multi_nb(void *src1, void *src2, void *dst,  size_t n_vectors,
                       size_t count, size_t stride, ucc_datatype_t dt,
                       ucc_coll_args_t *args, ucc_ee_executor_t *exec,
                       ucc_ee_executor_task_t **task)
{
    ucc_ee_executor_task_args_t eargs;

    if (!UCC_DT_IS_PREDEFINED(dt)) {
        ucc_assert(UCC_DT_HAS_REDUCE(dt));
        *task = NULL;
        return ucc_mc_reduce_userdefined(src1, src2, dst, n_vectors, count,
                                         stride, ucc_dt_to_generic(dt));
    } else {
        eargs.task_type = UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI;
        eargs.bufs[0]   = dst;
        eargs.bufs[1]   = src1;
        eargs.bufs[2]   = src2;
        eargs.count     = count;
        eargs.size      = n_vectors;
        eargs.stride    = stride;
        eargs.dt        = dt;
        eargs.op        = args->op;

        return ucc_ee_executor_task_post(exec, &eargs, task);
    }
}

static inline ucc_status_t
ucc_dt_reduce_multi_alpha(void *src1, void *src2, void *dst, size_t n_vectors,
                          size_t count, size_t stride, ucc_datatype_t dt,
                          ucc_reduction_op_t vector_op, double alpha,
                          ucc_memory_type_t mem_type, ucc_coll_args_t *args)
{
    /* reduce_multi is used for OP_AVG implementation that can only be
       used with predefined dtypes */
    ucc_assert(UCC_DT_IS_PREDEFINED(dt));
    return ucc_mc_reduce_multi_alpha(src1, src2, dst, n_vectors, count,
                                     stride, dt, args->op,
                                     vector_op, alpha, mem_type);
}

static inline ucc_status_t
ucc_dt_reduce_multi_alpha_nb(void *src1, void *src2, void *dst,
                             size_t n_vectors, size_t count, size_t stride,
                             ucc_datatype_t dt, double alpha,
                             ucc_coll_args_t *args, ucc_ee_executor_t *exec,
                             ucc_ee_executor_task_t **task)
{
    ucc_ee_executor_task_args_t eargs;

    /* reduce_multi is used for OP_AVG implementation that can only be
       used with predefined dtypes */
    ucc_assert(UCC_DT_IS_PREDEFINED(dt));
    eargs.task_type = UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI_ALPHA;
    eargs.bufs[0]   = dst;
    eargs.bufs[1]   = src1;
    eargs.bufs[2]   = src2;
    eargs.alpha     = alpha;
    eargs.count     = count;
    eargs.size      = n_vectors;
    eargs.stride    = stride;
    eargs.dt        = dt;
    eargs.op        = args->op;

    return ucc_ee_executor_task_post(exec, &eargs, task);
}

#endif
