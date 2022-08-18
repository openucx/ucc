/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"

TestBarrier::TestBarrier(ucc_test_team_t &team, TestCaseParams &params) :
    TestCase(team, UCC_COLL_TYPE_BARRIER, params)
{
    status = UCC_OK;
    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

ucc_status_t TestBarrier::set_input()
{
    return UCC_OK;
}

ucc_status_t TestBarrier::reset_sbuf()
{
    return UCC_OK;
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

void TestBarrier::run(bool triggered)
{
    int completed = 1;
    int *recv = NULL;
    int rank, size;
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
            recv[i] = -1;
            completed = 0;
            MPI_Irecv(&recv[i], 1, MPI_INT, MPI_ANY_SOURCE, 123, team.comm, &rreq);
        }
        if (rank == i) {
            MPI_Ssend(&rank, 1, MPI_INT, 0, 123, team.comm);
        }
        UCC_CHECK(ucc_collective_post(req));
        do {
            status = ucc_collective_test(req);
            if (status < 0) {
                std::cerr << "failure in collective test\n";
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
            ucc_context_progress(team.ctx);
            if (0 == rank && !completed) {
                MPI_Test(&rreq, &completed, MPI_STATUS_IGNORE);
            }
            mpi_progress();
        } while(UCC_OK != status);
        while (0 == rank && !completed) {
            MPI_Test(&rreq, &completed, MPI_STATUS_IGNORE);
            mpi_progress();
        }
        if (i < size - 1) {
            UCC_CHECK(ucc_collective_finalize(req));
            UCC_CHECK(ucc_collective_init(&args, &req, team.team));
        }
    }
    MPI_Barrier(team.comm);
    if (0 == rank) {
        for (int i = 0; i < size; i++) {
            if (recv[i] != i) {
                status = UCC_ERR_NO_MESSAGE;
                break;
            }
        }
    }
    if (0 == rank) {
        delete[] recv;
    }
}
