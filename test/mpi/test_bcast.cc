/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestBcast::TestBcast(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_BCAST, params)
{
    int rank, size;
    size_t dt_size, count;

    dt      = params.dt;
    dt_size = ucc_dt_size(dt);
    count   = msgsize / dt_size;
    root    = params.root;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (skip_reduce(test_max_size < msgsize, TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    check_buf = ucc_malloc(msgsize, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize, mem_type));
    sbuf = sbuf_mc_header->addr;

    args.src.info.buffer   = sbuf;
    args.src.info.count    = count;
    args.src.info.datatype = dt;
    args.src.info.mem_type = mem_type;
    args.root              = root;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestBcast::set_input(int iter_persistent)
{
    size_t dt_size = ucc_dt_size(dt);
    size_t count   = msgsize / dt_size;
    int    rank;

    MPI_Comm_rank(team.comm, &rank);
    if (rank == root) {
        init_buffer(sbuf, count, dt, mem_type, rank * (iter_persistent + 1));
        UCC_CHECK(ucc_mc_memcpy(check_buf, sbuf, count * dt_size,
                  UCC_MEMORY_TYPE_HOST, mem_type));
    }
    return UCC_OK;
}

ucc_status_t TestBcast::check()
{
    size_t       count = args.src.info.count;
    MPI_Datatype mpi_dt = ucc_dt_to_mpi(dt);
    int          rank, completed;
    MPI_Request  req;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Ibcast(check_buf, count, mpi_dt, root, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return (rank == root)
               ? UCC_OK
               : compare_buffers(sbuf, check_buf, count, dt, mem_type);
}
