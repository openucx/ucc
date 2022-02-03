/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

TestBcast::TestBcast(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                     ucc_memory_type_t _mt, int _root, ucc_test_team_t &_team,
                     size_t _max_size) :
    TestCase(_team, UCC_COLL_TYPE_BCAST, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count   = _msgsize / dt_size;
    int rank, size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);
    root = _root;

    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT,
                    team.comm)) {
        return;
    }

    check_buf = ucc_malloc(_msgsize, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
    sbuf = sbuf_mc_header->addr;

    args.src.info.buffer   = sbuf;
    args.src.info.count    = count;
    args.src.info.datatype = TEST_DT;
    args.src.info.mem_type = _mt;
    args.root              = root;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestBcast::set_input()
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count   = msgsize / dt_size;
    int    rank;

    MPI_Comm_rank(team.comm, &rank);
    if (rank == root) {
        init_buffer(sbuf, count, TEST_DT, mem_type, rank);
        UCC_CHECK(ucc_mc_memcpy(check_buf, sbuf, count * dt_size,
                  UCC_MEMORY_TYPE_HOST, mem_type));
    }
    return UCC_OK;
}

ucc_status_t TestBcast::reset_sbuf()
{
    return UCC_OK;
}

ucc_status_t TestBcast::check()
{
    size_t       count = args.src.info.count;
    MPI_Datatype dt    = ucc_dt_to_mpi(TEST_DT);
    int          rank, completed;
    MPI_Request  req;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Ibcast(check_buf, count, dt, root, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return (rank == root) ? UCC_OK :
        compare_buffers(sbuf, check_buf, count, TEST_DT, mem_type);
}
