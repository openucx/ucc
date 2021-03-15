/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

TestAllgather::TestAllgather(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                             ucc_memory_type_t _mt, ucc_test_team_t &_team) :
    TestCase(_team, _mt, _msgsize, _inplace)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count = _msgsize/dt_size;
    int rank, size;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    args.coll_type         = UCC_COLL_TYPE_ALLGATHER;

    if (skip(test_max_size &&
              (test_max_size < (_msgsize*size)),
        TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf, _msgsize*size, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_rbuf, _msgsize*size, UCC_MEMORY_TYPE_HOST));
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf, _msgsize, _mt));
        init_buffer(sbuf, count, TEST_DT, _mt, rank);
        UCC_CHECK(ucc_mc_alloc(&check_sbuf, _msgsize, UCC_MEMORY_TYPE_HOST));
        init_buffer(check_sbuf, count, TEST_DT, UCC_MEMORY_TYPE_HOST, rank);
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer((void*)((ptrdiff_t)rbuf + rank*count*dt_size),
                    count, TEST_DT, _mt, rank);
        init_buffer((void*)((ptrdiff_t)check_rbuf + rank*count*dt_size),
                    count, TEST_DT, UCC_MEMORY_TYPE_HOST, rank);
    }

    args.src.info.buffer   = sbuf;
    args.src.info.count    = count;
    args.src.info.datatype = TEST_DT;
    args.src.info.mem_type = _mt;
    args.dst.info.buffer   = rbuf;
    args.dst.info.count    = count;
    args.dst.info.datatype = TEST_DT;
    args.dst.info.mem_type = _mt;
    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

ucc_status_t TestAllgather::check()
{
    size_t       count = args.dst.info.count;
    MPI_Datatype dt    = ucc_dt_to_mpi(TEST_DT);
    int          size;

    MPI_Comm_size(team.comm, &size);
    MPI_Allgather(inplace ? MPI_IN_PLACE : check_sbuf, count, dt,
                  check_rbuf, count, dt, team.comm);
    return compare_buffers(rbuf, check_rbuf, count*size, TEST_DT, mem_type);
}
