/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_MPI_H
#define TEST_MPI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <mpi.h>
#include <ucc/api/ucc.h>

#define STR(x) #x
#define UCC_CHECK(_call)                                                       \
    if (UCC_OK != (_call)) {                                                   \
        fprintf(stderr, "*** UCC TEST FAIL: %s\n", STR(_call));                \
        MPI_Abort(MPI_COMM_WORLD, -1);                                         \
    }

extern ucc_team_h    ucc_world_team;
extern ucc_context_h team_ctx;

int ucc_mpi_test_init(int argc, char **argv, uint64_t coll_types,
                      unsigned thread_mode);

void ucc_mpi_test_finalize(void);

int ucc_mpi_create_comm_nb(MPI_Comm comm, ucc_team_h *team);

int ucc_mpi_create_comm(MPI_Comm comm, ucc_team_h *team);

void ucc_mpi_test_progress(void);

#endif
