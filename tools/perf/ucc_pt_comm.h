/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_COMM_H
#define UCC_PT_COMM_H

#include <ucc/api/ucc.h>
#include "ucc_pt_config.h"
#include "ucc_pt_bootstrap.h"
#include "ucc_pt_bootstrap_mpi.h"
#include <vector>

class ucc_pt_comm {
    ucc_pt_comm_config cfg;
    ucc_lib_h lib;
    ucc_context_h context;
    std::vector<ucc_team_h> teams;
    int                     n_teams;
    ucc_pt_bootstrap *bootstrap;
    void set_gpu_device();
public:
    ucc_pt_comm(ucc_pt_comm_config config);
    int get_rank();
    int get_size();
    ucc_team_h    get_team(int team_id = 0);
    ucc_context_h get_context();
    ~ucc_pt_comm();
    ucc_status_t init(int n_threads);
    ucc_status_t barrier(int team_id = 0);
    ucc_status_t allreduce(float* in, float* out, size_t size,
                           ucc_reduction_op_t op, int team_id = 0);
    ucc_status_t finalize();
};

#endif
