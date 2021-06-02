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
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count = _msgsize/dt_size;
    int rank, size;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);
    root = _root;
    args.coll_type = UCC_COLL_TYPE_BCAST;

    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT,
                    team.comm)) {
        return;
    }

    check_rbuf = ucc_malloc(_msgsize * size, "check rbuf");
    UCC_MALLOC_CHECK(check_rbuf);
    UCC_CHECK(ucc_mc_alloc(&sbuf, _msgsize, _mt));
    check_sbuf = ucc_malloc(_msgsize, "check sbuf");
    UCC_MALLOC_CHECK(check_sbuf);
    if (rank == root) {
        init_buffer(sbuf, count, TEST_DT, _mt, rank);
        UCC_CHECK(ucc_mc_memcpy(check_sbuf, sbuf, _msgsize,                        \
                  UCC_MEMORY_TYPE_HOST, _mt));
    }

    args.src.info.buffer      = sbuf;
    args.src.info.count       = count;
    args.src.info.datatype    = TEST_DT;
    args.src.info.mem_type    = _mt;
    args.root                 = root;
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestBcast::check()
{
    size_t       count = args.src.info.count;
    MPI_Datatype dt    = ucc_dt_to_mpi(TEST_DT);
    int          rank;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Bcast((rank == root) ? check_sbuf : check_rbuf, count, dt, root, team.comm);
    return (rank == root) ? UCC_OK :
        compare_buffers(sbuf, check_rbuf, count, TEST_DT, mem_type);
}
