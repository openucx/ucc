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

ucc_pt_op_memcpy::ucc_pt_op_memcpy(ucc_datatype_t dt, ucc_memory_type mt,
                                   ucc_pt_comm *communicator) :
                                   ucc_pt_coll(communicator)
{
    has_inplace_   = false;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;

    data_type = dt;
    mem_type  = mt;
}

ucc_status_t ucc_pt_op_memcpy::init_args(size_t count,
                                         ucc_pt_test_args_t &test_args)
{
    ucc_ee_executor_task_args_t &args = test_args.executor_args;
    size_t                       size = count * ucc_dt_size(data_type);
    ucc_status_t st;

    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header, size, mem_type), exit, st);
    UCCCHECK_GOTO(ucc_pt_alloc(&src_header, size, mem_type), free_dst, st);

    args.task_type = UCC_EE_EXECUTOR_TASK_COPY;
    args.copy.dst  = dst_header->addr;
    args.copy.src  = src_header->addr;
    args.copy.len  = size;

    return UCC_OK;
free_dst:
    ucc_pt_free(dst_header);
exit:
    return st;
}

float ucc_pt_op_memcpy::get_bw(float time_ms, int grsize,
                                ucc_pt_test_args_t test_args)
{
    ucc_ee_executor_task_args_t &args = test_args.executor_args;
    float                        S    = args.copy.len;

    return 2 * (S / time_ms) / 1000.0;
}

void ucc_pt_op_memcpy::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_pt_free(src_header);
    ucc_pt_free(dst_header);
}
