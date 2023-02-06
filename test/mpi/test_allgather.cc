/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestAllgather::TestAllgather(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_ALLGATHER, params)
{
    int    rank, size;
    size_t dt_size, single_rank_count;

    dt                = params.dt;
    dt_size           = ucc_dt_size(dt);
    single_rank_count = msgsize / dt_size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize*size),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize * size, mem_type));
    rbuf      = rbuf_mc_header->addr;
    check_buf = ucc_malloc(msgsize * size, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    if (!inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize, mem_type));
        sbuf                   = sbuf_mc_header->addr;
        args.src.info.buffer   = sbuf;
        args.src.info.count    = single_rank_count;
        args.src.info.datatype = dt;
        args.src.info.mem_type = mem_type;
    }
    args.dst.info.buffer   = rbuf;
    args.dst.info.count    = single_rank_count * size;
    args.dst.info.datatype = dt;
    args.dst.info.mem_type = mem_type;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAllgather::set_input(int iter_persistent)
{
    size_t dt_size           = ucc_dt_size(dt);
    size_t single_rank_count = msgsize / dt_size;
    size_t single_rank_size  = single_rank_count * dt_size;
    int    rank;
    void  *buf, *check;

    MPI_Comm_rank(team.comm, &rank);
    if (inplace) {
        buf   = PTR_OFFSET(rbuf, rank * single_rank_size);
    } else {
        buf   = sbuf;
    }
    check = PTR_OFFSET(check_buf, rank * single_rank_size);

    init_buffer(buf, single_rank_count, dt, mem_type,
                rank * (iter_persistent + 1));
    UCC_CHECK(ucc_mc_memcpy(check, buf, single_rank_size,
                            UCC_MEMORY_TYPE_HOST, mem_type));
    return UCC_OK;
}

ucc_status_t TestAllgather::check()
{
    int size, completed;
    MPI_Comm_size(team.comm, &size);
    size_t       single_rank_count = args.dst.info.count / size;
    MPI_Datatype mpi_dt            = ucc_dt_to_mpi(dt);
    MPI_Request  req;

    MPI_Iallgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, check_buf,
                   single_rank_count, mpi_dt, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check_buf, single_rank_count * size, dt,
                           mem_type);
}
