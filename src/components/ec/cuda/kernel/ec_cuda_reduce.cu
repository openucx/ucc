/**
 * Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UINT32_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

extern "C" {
#include "../ec_cuda.h"
}

extern "C" {

ucc_status_t ucc_ec_cuda_reduce_int(
    ucc_ee_executor_task_args_t *task, ucc_datatype_t dt, ucc_reduction_op_t op,
    unsigned long bk, int th, cudaStream_t stream);

ucc_status_t ucc_ec_cuda_reduce_float(
    ucc_ee_executor_task_args_t *task, ucc_datatype_t dt, ucc_reduction_op_t op,
    unsigned long bk, int th, cudaStream_t stream);

ucc_status_t ucc_ec_cuda_reduce_complex(
    ucc_ee_executor_task_args_t *task, ucc_datatype_t dt, ucc_reduction_op_t op,
    unsigned long bk, int th, cudaStream_t stream);

ucc_status_t ucc_ec_cuda_reduce(
    ucc_ee_executor_task_args_t *task, unsigned num_threads,
    unsigned num_blocks, cudaStream_t stream)
{
    int                th = num_threads;
    unsigned long      bk = num_blocks;
    ucc_reduction_op_t op;
    ucc_datatype_t     dt;
    size_t             count;
    int                i;
    ucc_status_t       status;

    if (task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {
        count = task->reduce.count;
        dt    = task->reduce.dt;
        op    = task->reduce.op;
        bk    = ucc_min((count + th - 1) / th, bk);
    } else if (task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED) {
        count = task->reduce_strided.count;
        dt    = task->reduce_strided.dt;
        op    = task->reduce_strided.op;
        bk    = ucc_min((count + th - 1) / th, bk);
    } else {
        if (task->reduce_multi_dst.n_bufs == 0) {
            return UCC_OK;
        }
        count = 0;
        for (i = 0; i < task->reduce_multi_dst.n_bufs; i++) {
            count += task->reduce_multi_dst.counts[i];
        }
        dt = task->reduce_multi_dst.dt;
        op = task->reduce_multi_dst.op;
        bk = 4 * task->reduce_multi_dst.n_bufs;
    }

    if (count == 0) {
        return UCC_OK;
    }

    status = ucc_ec_cuda_reduce_int(task, dt, op, bk, th, stream);
    if (status != UCC_ERR_NOT_SUPPORTED) {
        return status;
    }
    status = ucc_ec_cuda_reduce_float(task, dt, op, bk, th, stream);
    if (status != UCC_ERR_NOT_SUPPORTED) {
        return status;
    }
    status = ucc_ec_cuda_reduce_complex(task, dt, op, bk, th, stream);
    if (status != UCC_ERR_NOT_SUPPORTED) {
        return status;
    }

    ec_error(
        &ucc_ec_cuda.super,
        "unsupported reduction type (%s)",
        ucc_datatype_str(dt));
    return UCC_ERR_NOT_SUPPORTED;
}
}
