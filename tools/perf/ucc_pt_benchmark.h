/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_BENCH_H
#define UCC_PT_BENCH_H

#include "ucc_pt_config.h"
#include "ucc_pt_coll.h"
#include "generator/ucc_pt_generator.h"
#include "ucc_pt_comm.h"
#include "utils/ucc_coll_utils.h"
#include <ucc/api/ucc.h>

class ucc_pt_benchmark {
    ucc_pt_benchmark_config config;
    ucc_pt_comm *comm;
    ucc_pt_coll *coll;
    ucc_pt_generator_base *generator;

    void print_header();
    void print_time(size_t count, ucc_pt_test_args_t args, double time_avg,
                    double time_min, double time_max);
public:
    ucc_pt_benchmark(ucc_pt_benchmark_config cfg, ucc_pt_comm *communicator);
    ucc_status_t run_bench() noexcept;
    ucc_status_t run_single_coll_test(ucc_coll_args_t args,
                                      int nwarmup, int niter,
                                      double &time) noexcept;
    ucc_status_t run_single_executor_test(ucc_ee_executor_task_args_t args,
                                          int nwarmup, int niter,
                                          double &time) noexcept;
    ~ucc_pt_benchmark();
};

#endif
