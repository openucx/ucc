/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_BOOTSTRAP_MPI_H
#define UCC_PT_BOOTSTRAP_MPI_H

#include <mpi.h>
#include "ucc_pt_bootstrap.h"

class ucc_pt_bootstrap_mpi: public ucc_pt_bootstrap {
public:
    ucc_pt_bootstrap_mpi();
    ~ucc_pt_bootstrap_mpi();
    int get_rank() override;
    int get_size() override;
protected:
    int rank;
    int size;
    int ppn;
};

#endif
