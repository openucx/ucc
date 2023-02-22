/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestScatter::TestScatter(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_SCATTER, params)
{
    int    rank, size;
    size_t dt_size, single_rank_count;

    dt                = params.dt;
    dt_size           = ucc_dt_size(dt);
    single_rank_count = msgsize / dt_size;
    root              = params.root;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize * size),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    if (rank == root) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize * size, mem_type));
        sbuf = sbuf_mc_header->addr;
        if (inplace) {
            rbuf_mc_header = NULL;
            rbuf = NULL;
        } else {
            UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize, mem_type));
            rbuf = rbuf_mc_header->addr;
        }
    } else {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize, mem_type));
        rbuf = rbuf_mc_header->addr;
        sbuf_mc_header = NULL;
        sbuf = NULL;
    }

    check_buf = ucc_malloc(msgsize * size, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    args.root = root;
    if (rank == root) {
        args.src.info.buffer   = sbuf;
        args.src.info.count    = single_rank_count * size;
        args.src.info.datatype = dt;
        args.src.info.mem_type = mem_type;
        if (!inplace) {
            args.dst.info.buffer   = rbuf;
            args.dst.info.count    = single_rank_count;
            args.dst.info.datatype = dt;
            args.dst.info.mem_type = mem_type;
        }
    } else {
        args.dst.info.buffer   = rbuf;
        args.dst.info.count    = single_rank_count;
        args.dst.info.datatype = dt;
        args.dst.info.mem_type = mem_type;
    }

    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestScatter::set_input(int iter_persistent)
{
    size_t dt_size           = ucc_dt_size(dt);
    size_t single_rank_count = msgsize / dt_size;
    size_t single_rank_size  = single_rank_count * dt_size;
    int    rank, size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (rank == root) {
        init_buffer(sbuf, single_rank_count * size, dt, mem_type,
                    rank * (iter_persistent + 1));
        UCC_CHECK(ucc_mc_memcpy(check_buf, sbuf, single_rank_size * size,
                                UCC_MEMORY_TYPE_HOST, mem_type));
    }
    return UCC_OK;
}

ucc_status_t TestScatter::check()
{
    size_t        single_rank_count = msgsize / ucc_dt_size(dt);
    size_t        single_rank_size  = single_rank_count * ucc_dt_size(dt);
    MPI_Datatype  mpi_dt            = ucc_dt_to_mpi(dt);
    MPI_Request   req;
    int           size, rank, completed;
    MPI_Comm_size(team.comm, &size);
    MPI_Comm_rank(team.comm, &rank);

    MPI_Iscatter(check_buf, single_rank_count, mpi_dt,
                 (rank == root) ? MPI_IN_PLACE : check_buf, single_rank_count,
                 mpi_dt, root, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    if (rank == root) {
        if (inplace) {
            return compare_buffers(sbuf, check_buf, single_rank_count * size,
                                   dt, mem_type);
        } else {
            return compare_buffers(
                rbuf, PTR_OFFSET(check_buf, single_rank_size * rank),
                single_rank_count, dt, mem_type);
        }
    } else {
        return compare_buffers(rbuf, check_buf, single_rank_count, dt,
                               mem_type);
    }
}
