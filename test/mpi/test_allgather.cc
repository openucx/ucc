/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

TestAllgather::TestAllgather(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                             ucc_memory_type_t _mt, ucc_test_team_t &_team,
                             size_t _max_size) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t single_rank_count = _msgsize / dt_size;
    int rank, size;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    args.coll_type = UCC_COLL_TYPE_ALLGATHER;

    if (test_max_size < (_msgsize*size)) {
        test_skip = TEST_SKIP_MEM_LIMIT;
    }
    if (TEST_SKIP_NONE != skip_reduce(test_skip, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf, _msgsize*size, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_rbuf, _msgsize*size, UCC_MEMORY_TYPE_HOST));
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf, _msgsize, _mt));
        init_buffer(sbuf, single_rank_count, TEST_DT, _mt, rank);
        UCC_ALLOC_COPY_BUF(check_sbuf, UCC_MEMORY_TYPE_HOST, sbuf, _mt, _msgsize);
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer(
            (void *)((ptrdiff_t)rbuf + rank * single_rank_count * dt_size),
            single_rank_count, TEST_DT, _mt, rank);
        init_buffer((void *)((ptrdiff_t)check_rbuf +
                             rank * single_rank_count * dt_size),
                    single_rank_count, TEST_DT, UCC_MEMORY_TYPE_HOST, rank);
    }

    args.src.info.buffer   = sbuf;
    args.src.info.count    = single_rank_count;
    args.src.info.datatype = TEST_DT;
    args.src.info.mem_type = _mt;
    args.dst.info.buffer   = rbuf;
    args.dst.info.count    = single_rank_count * size;
    args.dst.info.datatype = TEST_DT;
    args.dst.info.mem_type = _mt;
    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

ucc_status_t TestAllgather::check()
{
    int size;
    MPI_Comm_size(team.comm, &size);
    size_t       single_rank_count = args.dst.info.count / size;
    MPI_Datatype dt    = ucc_dt_to_mpi(TEST_DT);

    MPI_Allgather(inplace ? MPI_IN_PLACE : check_sbuf, single_rank_count, dt,
                  check_rbuf, single_rank_count, dt, team.comm);
    return compare_buffers(rbuf, check_rbuf, single_rank_count * size, TEST_DT, mem_type);
}
