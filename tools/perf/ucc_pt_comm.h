/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_COMM_H
#define UCC_PT_COMM_H

#include <ucc/api/ucc.h>
#include "ucc_pt_bootstrap.h"
#include "ucc_pt_bootstrap_mpi.h"

class ucc_pt_comm {
    ucc_lib_h lib;
    ucc_context_h context;
    ucc_team_h team;
    ucc_pt_bootstrap *bootstrap;
public:
    ucc_pt_comm();
    int get_rank();
    int get_size();
    ucc_team_h get_team();
    ucc_context_h get_context();
    ~ucc_pt_comm();
    ucc_status_t init();
    ucc_status_t barrier();
    ucc_status_t allreduce(float* in, float *out, size_t size,
                           ucc_reduction_op_t op);
    ucc_status_t finalize();
};

#endif
