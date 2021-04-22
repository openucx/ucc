/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
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
};

#endif
