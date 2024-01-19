/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestGather::TestGather(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_GATHER, params)
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
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize * size, mem_type));
        rbuf = rbuf_mc_header->addr;
        if (inplace) {
            sbuf_mc_header = NULL;
            sbuf = NULL;
        } else {
            UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize, mem_type));
            sbuf = sbuf_mc_header->addr;
        }
    } else {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize, mem_type));
        sbuf = sbuf_mc_header->addr;
        rbuf_mc_header = NULL;
        rbuf = NULL;
    }

    check_buf = ucc_malloc(msgsize*size, "check buf");
    UCC_MALLOC_CHECK(check_buf);

    args.root = root;
    if (rank == root) {
        args.dst.info.buffer   = rbuf;
        args.dst.info.count    = single_rank_count * size;
        args.dst.info.datatype = dt;
        args.dst.info.mem_type = mem_type;
        if (!inplace) {
            args.src.info.buffer   = sbuf;
            args.src.info.count    = single_rank_count;
            args.src.info.datatype = dt;
            args.src.info.mem_type = mem_type;
        }
    } else {
        args.src.info.buffer   = sbuf;
        args.src.info.count    = single_rank_count;
        args.src.info.datatype = dt;
        args.src.info.mem_type = mem_type;
    }

    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestGather::set_input(int iter_persistent)
{
    size_t dt_size           = ucc_dt_size(dt);
    size_t single_rank_count = msgsize / dt_size;
    size_t single_rank_size  = single_rank_count * dt_size;
    int    rank;
    void  *buf;

    this->iter_persistent = iter_persistent;
    MPI_Comm_rank(team.comm, &rank);
    if (rank == root) {
        if (inplace) {
            buf = PTR_OFFSET(rbuf, rank * single_rank_size);
        } else {
            buf = sbuf;
        }
    } else {
        buf = sbuf;
    }
    init_buffer(buf, single_rank_count, dt, mem_type,
                rank * (iter_persistent + 1));
    return UCC_OK;
}

ucc_status_t TestGather::check()
{
    int size, rank, i;
    size_t dt_size, single_rank_count;

    MPI_Comm_size(team.comm, &size);
    MPI_Comm_rank(team.comm, &rank);
    if (rank != root) {
        return UCC_OK;
    }

    dt_size = ucc_dt_size(dt);
    single_rank_count = msgsize / dt_size;
    for (i = 0; i < size; i++) {
        init_buffer(PTR_OFFSET(check_buf, i * single_rank_count * dt_size),
                    single_rank_count, dt, UCC_MEMORY_TYPE_HOST,
                    i * (iter_persistent + 1));
    }

    return compare_buffers(rbuf, check_buf, single_rank_count * size, dt,
                           mem_type);
}
