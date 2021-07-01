/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

TestAlltoall::TestAlltoall(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                           ucc_memory_type_t _mt, ucc_test_team_t &_team,
                           size_t _max_size) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count = _msgsize/dt_size;
    int rank;
    int nprocs;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    args.coll_type = UCC_COLL_TYPE_ALLTOALL;

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (_msgsize * nprocs),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize * nprocs, _mt));
    rbuf       = rbuf_mc_header->addr;
    check_rbuf = ucc_malloc(_msgsize * nprocs, "check rbuf");
    UCC_MALLOC_CHECK(check_rbuf);
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize * nprocs, _mt));
        sbuf = sbuf_mc_header->addr;
        init_buffer(sbuf, count * nprocs, TEST_DT, _mt, rank);
        UCC_ALLOC_COPY_BUF(check_sbuf_mc_header, UCC_MEMORY_TYPE_HOST, sbuf,
                           _mt, _msgsize * nprocs);
        check_sbuf = check_sbuf_mc_header->addr;
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer(rbuf, count * nprocs, TEST_DT, _mt, rank);
        init_buffer(check_rbuf, count * nprocs, TEST_DT, UCC_MEMORY_TYPE_HOST, rank);
    }

    args.src.info.buffer      = sbuf;
    args.src.info.count       = count;
    args.src.info.datatype    = TEST_DT;
    args.src.info.mem_type    = _mt;

    args.dst.info.buffer      = rbuf;
    args.dst.info.count       = count;
    args.dst.info.datatype    = TEST_DT;
    args.dst.info.mem_type    = _mt;
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAlltoall::check()
{
    size_t      count = args.src.info.count;
    MPI_Request req;
    int         completed;

    MPI_Ialltoall(inplace ? MPI_IN_PLACE : check_sbuf, count, ucc_dt_to_mpi(TEST_DT),
                  check_rbuf, count, ucc_dt_to_mpi(TEST_DT), team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check_rbuf, count, TEST_DT, mem_type);
}
