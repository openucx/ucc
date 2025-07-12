/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_coll.h"
#include "ucc_perftest.h"
#include <ucc/api/ucc.h>
#include <utils/ucc_math.h>
#include <utils/ucc_coll_utils.h>

ucc_pt_op_reduce_strided::ucc_pt_op_reduce_strided(ucc_datatype_t dt,
                                                   ucc_memory_type mt,
                                                   ucc_reduction_op_t op,
                                                   int nbufs,
                                                   ucc_pt_comm *communicator,
                                                   ucc_pt_generator_base *generator) :
                                                   ucc_pt_coll(communicator, generator)
{
    has_inplace_   = false;
    has_reduction_ = true;
    has_range_     = true;
    has_bw_        = true;

    if (nbufs == UCC_PT_DEFAULT_N_BUFS) {
        nbufs = 2;
    }

    if (nbufs < 2) {
        throw std::runtime_error("dt reduce op requires at least 2 bufs");
    }

    data_type = dt;
    mem_type  = mt;
    reduce_op = op;
    num_bufs  = nbufs;
}

ucc_status_t ucc_pt_op_reduce_strided::init_args(ucc_pt_test_args_t &test_args)
{
    ucc_ee_executor_task_args_t &args    = test_args.executor_args;
    size_t                       dt_size = ucc_dt_size(data_type);
    size_t                       size    = generator->get_src_count() * dt_size;
    size_t                       stride  = generator->get_src_count() * dt_size;
    ucc_status_t st;

    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, size, mem_type), exit, st);
    UCCCHECK_GOTO(ucc_pt_alloc(&src_header, size * num_bufs, mem_type),
                  free_dst, st);

    args.task_type             = UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED;
    args.reduce_strided.dst    = dst_header->addr;
    args.reduce_strided.src1   = src_header->addr;
    args.reduce_strided.src2   = PTR_OFFSET(src_header->addr, size);
    args.reduce_strided.n_src2 = num_bufs - 1;
    args.reduce_strided.stride = stride;
    args.reduce_strided.count  = generator->get_src_count();
    args.reduce_strided.dt     = data_type;
    args.reduce_strided.op     = reduce_op;
    args.flags         = 0;

    return UCC_OK;
free_dst:
    ucc_pt_free(dst_header);
exit:
    return st;
}

float ucc_pt_op_reduce_strided::get_bw(float time_ms, int grsize,
                                       ucc_pt_test_args_t test_args)
{
    ucc_ee_executor_task_args_t &args = test_args.executor_args;
    float                        S    = args.reduce_strided.count *
                                        ucc_dt_size(data_type);

    return (num_bufs + 1) * (S / time_ms) / 1000.0;
}

void ucc_pt_op_reduce_strided::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_pt_free(src_header);
    ucc_pt_free(dst_header);
}
