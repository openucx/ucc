/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

TestScatter::TestScatter(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_SCATTER, params)
{
    size_t dt_size           = ucc_dt_size(TEST_DT);
    size_t single_rank_count = msgsize / dt_size;
    int    rank, size;

    root = params.root;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize * size),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    if (rank == root) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize * size, mem_type));
        sbuf = sbuf_mc_header->addr;
        if (TEST_NO_INPLACE == inplace) {
            UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize, mem_type));
            rbuf = rbuf_mc_header->addr;
        } else {
            rbuf_mc_header = NULL;
            rbuf = NULL;
        }
    } else {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize, mem_type));
        rbuf = rbuf_mc_header->addr;
        sbuf_mc_header = NULL;
        sbuf = NULL;
    }

    check_buf = ucc_malloc(msgsize * size, "check buf");
    UCC_MALLOC_CHECK(check_buf);

    if (TEST_INPLACE == inplace) {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    args.root = root;
    if (rank == root) {
        args.src.info.buffer   = sbuf;
        args.src.info.count    = single_rank_count * size;
        args.src.info.datatype = TEST_DT;
        args.src.info.mem_type = mem_type;
        if (TEST_NO_INPLACE == inplace) {
            args.dst.info.buffer   = rbuf;
            args.dst.info.count    = single_rank_count;
            args.dst.info.datatype = TEST_DT;
            args.dst.info.mem_type = mem_type;
        }
    } else {
        args.dst.info.buffer   = rbuf;
        args.dst.info.count    = single_rank_count;
        args.dst.info.datatype = TEST_DT;
        args.dst.info.mem_type = mem_type;
    }

    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestScatter::set_input()
{
    size_t dt_size           = ucc_dt_size(TEST_DT);
    size_t single_rank_count = msgsize / dt_size;
    size_t single_rank_size  = single_rank_count * dt_size;
    int    rank, size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (rank == root) {
        init_buffer(sbuf, single_rank_count * size, TEST_DT, mem_type, rank);
        UCC_CHECK(ucc_mc_memcpy(check_buf, sbuf, single_rank_size * size,
                                UCC_MEMORY_TYPE_HOST, mem_type));
    }
    return UCC_OK;
}

ucc_status_t TestScatter::reset_sbuf()
{
    return UCC_OK;
}

ucc_status_t TestScatter::check()
{
    size_t        single_rank_count = msgsize / ucc_dt_size(TEST_DT);
    size_t        single_rank_size  = single_rank_count * ucc_dt_size(TEST_DT);
    MPI_Datatype  dt                = ucc_dt_to_mpi(TEST_DT);
    MPI_Request   req;
    int           size, rank, completed;
    MPI_Comm_size(team.comm, &size);
    MPI_Comm_rank(team.comm, &rank);

    MPI_Iscatter(check_buf, single_rank_count, dt,
                (rank == root) ? MPI_IN_PLACE : check_buf, single_rank_count,
                 dt, root, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    if (rank == root) {
        if (TEST_INPLACE == inplace) {
            return compare_buffers(sbuf, check_buf, single_rank_count * size,
                                   TEST_DT, mem_type);
        } else {
            return compare_buffers(rbuf,
                                   PTR_OFFSET(check_buf, single_rank_size * rank),
                                   single_rank_count, TEST_DT,
                                   mem_type);
        }
    } else {
        return compare_buffers(rbuf, check_buf, single_rank_count, TEST_DT,
                               mem_type);
    }
}
