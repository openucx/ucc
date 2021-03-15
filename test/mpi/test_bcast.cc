/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

TestBcast::TestBcast(size_t _msgsize, ucc_memory_type_t _mt,
                     int _root, ucc_test_team_t &_team) :
    TestCase(_team, _mt, _msgsize)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count = _msgsize/dt_size;
    int rank, size;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);
    root = _root;
    args.coll_type            = UCC_COLL_TYPE_BCAST;

    if (skip(test_max_size && (test_max_size < (_msgsize*size)),
             TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&check_buf, _msgsize*size, _mt));
    UCC_CHECK(ucc_mc_alloc(&sbuf, _msgsize, _mt));
    if (rank == root) {
        init_buffer(sbuf, count, TEST_DT, _mt, rank);
    }

    args.src.info.buffer      = sbuf;
    args.src.info.count       = count;
    args.src.info.datatype    = TEST_DT;
    args.src.info.mem_type    = _mt;
    args.root                 = root;
    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

ucc_status_t TestBcast::check()
{
    size_t       count = args.src.info.count;
    MPI_Datatype dt    = ucc_dt_to_mpi(TEST_DT);
    int          rank;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Bcast((rank == root) ? sbuf : check_buf, count, dt, root, team.comm);
    return (rank == root) ? UCC_OK :
        compare_buffers(sbuf, check_buf, count, TEST_DT, mem_type);
}
