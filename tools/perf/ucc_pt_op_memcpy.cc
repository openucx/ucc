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
                                   int nbufs,
                                   ucc_pt_comm *communicator,
                                   ucc_pt_generator_base *generator) :
                                   ucc_pt_coll(communicator, generator)
{
    has_inplace_   = false;
    has_reduction_ = false;
    has_range_     = true;
    has_bw_        = true;

    if (nbufs == UCC_PT_DEFAULT_N_BUFS) {
        nbufs = 1;
    }

    if (nbufs > UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS) {
        throw std::runtime_error("max supported number of copy buffer is "
                                 STR(UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS));
    }

    data_type = dt;
    mem_type  = mt;
    num_bufs  = nbufs;
}

ucc_status_t ucc_pt_op_memcpy::init_args(ucc_pt_test_args_t &test_args)
{
    ucc_ee_executor_task_args_t &args    = test_args.executor_args;
    size_t                       dt_size = ucc_dt_size(data_type);
    size_t                       size    = generator->get_src_count() * dt_size;
    ucc_status_t st;
    int i;

    UCCCHECK_GOTO(ucc_pt_alloc(&dst_header,
                               num_bufs * size,
                               mem_type),
                  exit, st);
    UCCCHECK_GOTO(ucc_pt_alloc(&src_header,
                               num_bufs * size,
                               mem_type),
                  free_dst, st);

    if (num_bufs == 1) {
        args.task_type = UCC_EE_EXECUTOR_TASK_COPY;
        args.copy.dst  = dst_header->addr;
        args.copy.src  = src_header->addr;
        args.copy.len  = size;
    } else {
        args.task_type              = UCC_EE_EXECUTOR_TASK_COPY_MULTI;
        args.copy_multi.num_vectors = num_bufs;
        for (i = 0; i < num_bufs; i++) {
            args.copy_multi.src[i]    = PTR_OFFSET(src_header->addr,
                                                   size * i);
            args.copy_multi.dst[i]    = PTR_OFFSET(dst_header->addr,
                                                   size * i);
            args.copy_multi.counts[i] = generator->get_src_count();
        }
    }

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
    float S;
    int i;

    if (args.task_type == UCC_EE_EXECUTOR_TASK_COPY) {
        S = args.copy.len;
    } else {
        S = 0;
        for (i = 0; i < args.copy_multi.num_vectors; i++) {
            S += args.copy_multi.counts[i];
        }
    }

    return 2 * (S / time_ms) / 1000.0;
}

void ucc_pt_op_memcpy::free_args(ucc_pt_test_args_t &test_args)
{
    ucc_pt_free(src_header);
    ucc_pt_free(dst_header);
}
