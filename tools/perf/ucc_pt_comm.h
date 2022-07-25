/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_COMM_H
#define UCC_PT_COMM_H

#include <ucc/api/ucc.h>
#include "ucc_pt_config.h"
#include "ucc_pt_bootstrap.h"
#include "ucc_pt_bootstrap_mpi.h"

class ucc_pt_comm {
    ucc_pt_comm_config cfg;
    ucc_lib_h lib;
    ucc_context_h context;
    ucc_team_h team;
    void *stream;
    ucc_ee_h ee;
    ucc_pt_bootstrap *bootstrap;
    void set_gpu_device();
public:
    ucc_pt_comm(ucc_pt_comm_config config);
    int get_rank();
    int get_size();
    ucc_ee_h get_ee();
    ucc_team_h get_team();
    ucc_context_h get_context();
    ~ucc_pt_comm();
    ucc_status_t init();
    ucc_status_t barrier();
    ucc_status_t allreduce(double* in, double *out, size_t size,
                           ucc_reduction_op_t op);
    ucc_status_t finalize();
};

#endif
