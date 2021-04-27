/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_BENCH_H
#define UCC_PT_BENCH_H

#include "ucc_pt_config.h"
#include "ucc_pt_coll.h"
#include "ucc_pt_comm.h"
#include <ucc/api/ucc.h>
#include <chrono>

class ucc_pt_benchmark {
    ucc_pt_benchmark_config config;
    ucc_pt_comm *comm;
    ucc_pt_coll *coll;

    ucc_status_t barrier();
    void print_header();
    void print_time(size_t count, std::chrono::nanoseconds time);
public:
    ucc_pt_benchmark(ucc_pt_benchmark_config cfg, ucc_pt_comm *communcator);
    ucc_status_t run_bench();
    ucc_status_t run_single_test(ucc_coll_args_t args,
                                 int nwarmup, int niter,
                                 std::chrono::nanoseconds &time);
    ~ucc_pt_benchmark();
};

#endif
