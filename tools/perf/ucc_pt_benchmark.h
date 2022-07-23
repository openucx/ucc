/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_BENCH_H
#define UCC_PT_BENCH_H

#include "ucc_pt_config.h"
#include "ucc_pt_coll.h"
#include "ucc_pt_comm.h"
#include <ucc/api/ucc.h>

class ucc_pt_benchmark {
    ucc_pt_benchmark_config config;
    ucc_pt_comm *comm;
    ucc_pt_coll *coll;

    ucc_status_t barrier();
    void print_header();
    void print_time(size_t count, ucc_coll_args_t args,
                    double time);
public:
    ucc_pt_benchmark(ucc_pt_benchmark_config cfg, ucc_pt_comm *communicator);
    ucc_status_t run_bench() noexcept;
    ucc_status_t run_single_test(ucc_coll_args_t args,
                                 int nwarmup, int niter,
                                 double &time) noexcept;
    ~ucc_pt_benchmark();
};

#endif
