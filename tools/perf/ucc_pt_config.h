/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_CONFIG_H
#define UCC_PT_CONFIG_H

#include <iostream>
#include <sstream>
#include <string>
#include <map>
#include <getopt.h>
#include <ucc/api/ucc.h>

enum ucc_pt_bootstrap_type_t {
    UCC_PT_BOOTSTRAP_MPI,
    UCC_PT_BOOTSTRAP_UCX
};

struct ucc_pt_bootstrap_config {
    ucc_pt_bootstrap_type_t bootstrap;
};

struct ucc_pt_comm_config {
    ucc_memory_type_t mt;
};

struct ucc_pt_benchmark_config {
    ucc_coll_type_t    coll_type;
    size_t             min_count;
    size_t             max_count;
    ucc_datatype_t     dt;
    ucc_memory_type_t  mt;
    ucc_reduction_op_t op;
    bool               inplace;
    bool               triggered;
    size_t             large_thresh;
    int                n_iter_small;
    int                n_warmup_small;
    int                n_iter_large;
    int                n_warmup_large;
    bool               full_print;
};

struct ucc_pt_config {
    ucc_pt_bootstrap_config bootstrap;
    ucc_pt_comm_config      comm;
    ucc_pt_benchmark_config bench;

    ucc_pt_config();
    ucc_status_t process_args(int argc, char *argv[]);
    void print_help();
};

#endif
