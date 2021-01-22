/**
* Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#define _BSD_SOURCE
#include "test_mpi.h"
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

static inline void
do_barrier(ucc_team_h team) {
    ucc_coll_req_h request;
    ucc_coll_op_args_t coll = {
        .coll_type = UCC_COLL_TYPE_BARRIER,
    };
    UCC_CHECK(ucc_collective_init(&coll, &request, team));
    UCC_CHECK(ucc_collective_post(request));
    UCC_CHECK(ucc_collective_finalize(request));
}

int main (int argc, char **argv) {
    int rank, size;

    UCC_CHECK(ucc_mpi_test_init(argc, argv, UCC_COLL_TYPE_BARRIER
                                , UCC_THREAD_SINGLE));
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand(time(NULL));
    /* for (i=0; i<size; i++) { */
    /*     sleep_us = rand() % 1000; */
    /*     usleep(sleep_us); */
    /*     if (i == rank) { */
    /*         printf("Rank %d checks in\n", rank); */
    /*         fflush(stdout); */
    /*         usleep(100); */
    /*     } */
    /*     do_barrier(ucc_world_team); */
    /* } */
    do_barrier(ucc_world_team);
    ucc_mpi_test_finalize();
    return 0;
}
