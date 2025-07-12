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

ucc_pt_op_reduce::ucc_pt_op_reduce(ucc_datatype_t dt, ucc_memory_type mt,
                                   ucc_reduction_op_t op, int nbufs,
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

    if (nbufs > UCC_EE_EXECUTOR_NUM_BUFS) {
        throw std::runtime_error("dt reduce op supports up to " +
                                 std::to_string(UCC_EE_EXECUTOR_NUM_BUFS) +
                                 " bufs");
    }

    data_type = dt;
    mem_type  = mt;
    reduce_op = op;
    num_bufs  = nbufs;
}

ucc_status_t ucc_pt_op_reduce::init_args(ucc_pt_test_args_t &test_args)
{
    ucc_ee_executor_task_args_t &args   = test_args.executor_args;
    size_t                       dt_size = ucc_dt_size(data_type);
    size_t                       size    = generator->get_src_count() * dt_size;
    ucc_status_t st;
    int i;

    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, size, mem_type), exit, st);
    UCCCHECK_GOTO(ucc_pt_alloc(&src_header, size * num_bufs, mem_type),
                  free_dst, st);

    args.task_type     = UCC_EE_EXECUTOR_TASK_REDUCE;
    args.reduce.dst    = dst_header->addr;
    args.reduce.n_srcs = num_bufs;
    args.reduce.count  = generator->get_src_count();
    args.reduce.dt     = data_type;
    args.reduce.op     = reduce_op;
    args.flags         = 0;
    for (i = 0; i < num_bufs; i++) {
        args.reduce.srcs[i] = PTR_OFFSET(src_header->addr, i * size);
    }

    return UCC_OK;
free_dst:
    ucc_pt_free(dst_header);
exit:
    return st;
}

float ucc_pt_op_reduce::get_bw(float time_ms, int grsize,
                                ucc_pt_test_args_t test_args)
{
    ucc_ee_executor_task_args_t &args = test_args.executor_args;
    float                        S    = args.reduce.count * ucc_dt_size(data_type);

    return (num_bufs + 1) * (S / time_ms) / 1000.0;
}

void ucc_pt_op_reduce::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_pt_free(src_header);
    ucc_pt_free(dst_header);
}
