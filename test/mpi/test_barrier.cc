/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"

TestBarrier::TestBarrier(ucc_test_team_t &team) : TestCase(team)
{
    status = UCC_OK;
    args.coll_type = UCC_COLL_TYPE_BARRIER;
    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

ucc_status_t TestBarrier::check()
{
    return status;
}

std::string TestBarrier::str() {
    return std::string("tc=")+std::string(ucc_coll_type_str(args.coll_type)) +
        std::string(" team=") + std::string(team_str(team.type));
}

ucc_status_t TestBarrier::test()
{
    return UCC_OK;
}

void TestBarrier::run()
{
    int rank, size;
    int *recv = NULL;
    MPI_Request rreq;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);
    if (0 == rank) {
        recv = new int[size];
    }
    srand(rank+1);
    /* random sleep 0 - 500 ms */
    usleep((rand() % 500)*1000);
    for (int i = 0; i < size; i++) {
        if (0 == rank) {
            MPI_Irecv(&recv[i], 1, MPI_INT, MPI_ANY_SOURCE, 123, team.comm, &rreq);
        }
        if (rank == i) {
            MPI_Send(&rank, 1, MPI_INT, 0, 123, team.comm);
        }
        UCC_CHECK(ucc_collective_post(req));
        do {
            status = ucc_collective_test(req);
            if (status < 0) {
                std::cerr << "failure in collective test\n";
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            ucc_context_progress(team.ctx);
        } while(UCC_OK != status);

        if  (0 == rank) {
            MPI_Wait(&rreq, MPI_STATUS_IGNORE);
        }
    }
    if (0 == rank) {
        for (int i = 0; i < size; i++) {
            if (recv[i] != i) {
                status = UCC_ERR_NO_MESSAGE;
                break;
            }
        }
    }
    delete recv;
}
