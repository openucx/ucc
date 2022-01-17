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
    TestCase(_team, UCC_COLL_TYPE_ALLGATHER, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size           = ucc_dt_size(TEST_DT);
    size_t single_rank_count = _msgsize / dt_size;
    int    rank, size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (_msgsize*size),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize * size, _mt));
    rbuf      = rbuf_mc_header->addr;
    check_buf = ucc_malloc(_msgsize*size, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        sbuf = sbuf_mc_header->addr;
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if (TEST_NO_INPLACE == inplace) {
        args.src.info.buffer   = sbuf;
        args.src.info.count    = single_rank_count;
        args.src.info.datatype = TEST_DT;
        args.src.info.mem_type = _mt;
    }
    args.dst.info.buffer   = rbuf;
    args.dst.info.count    = single_rank_count * size;
    args.dst.info.datatype = TEST_DT;
    args.dst.info.mem_type = _mt;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAllgather::set_input()
{
    size_t dt_size           = ucc_dt_size(TEST_DT);
    size_t single_rank_count = msgsize / dt_size;
    size_t single_rank_size  = single_rank_count * dt_size;
    int    rank;
    void  *buf, *check;

    MPI_Comm_rank(team.comm, &rank);
    if (inplace == TEST_NO_INPLACE) {
        buf   = sbuf;
    } else {
        buf   = PTR_OFFSET(rbuf, rank * single_rank_size);
    }
    check = PTR_OFFSET(check_buf, rank * single_rank_size);

    init_buffer(buf, single_rank_count, TEST_DT, mem_type, rank);
    UCC_CHECK(ucc_mc_memcpy(check, buf, single_rank_size,
                            UCC_MEMORY_TYPE_HOST, mem_type));
    return UCC_OK;
}

ucc_status_t TestAllgather::reset_sbuf()
{
    return UCC_OK;
}

ucc_status_t TestAllgather::check()
{
    int size, completed;
    MPI_Comm_size(team.comm, &size);
    size_t       single_rank_count = args.dst.info.count / size;
    MPI_Datatype dt    = ucc_dt_to_mpi(TEST_DT);
    MPI_Request  req;

    MPI_Iallgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   check_buf, single_rank_count, dt, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check_buf, single_rank_count * size, TEST_DT,
                           mem_type);
}
