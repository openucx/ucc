/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestAlltoall::TestAlltoall(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_ALLTOALL, params)
{
    void*  work_buf = nullptr;
    int    rank, nprocs;
    size_t dt_size, single_rank_count;

    dt                = params.dt;
    dt_size           = ucc_dt_size(dt);
    single_rank_count = msgsize / dt_size;
    is_onesided       = (params.buffers != nullptr);

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize * nprocs),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    if (!is_onesided) {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize * nprocs, mem_type));
        rbuf      = rbuf_mc_header->addr;
    } else {
        sbuf     = params.buffers[MEM_SEND_SEGMENT];
        rbuf     = params.buffers[MEM_RECV_SEGMENT];
        work_buf = params.buffers[MEM_WORK_SEGMENT];
    }

    check_buf = ucc_malloc(msgsize * nprocs, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    if (!inplace) {
        if (!is_onesided) {
            UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize * nprocs, mem_type));
            sbuf = sbuf_mc_header->addr;
        }
    }
    if (is_onesided) {
        args.mask  |=
            UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;
        args.flags |= UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS;
        args.global_work_buffer = work_buf;
    }

    if (!inplace) {
        args.src.info.buffer      = sbuf;
        args.src.info.count       = single_rank_count * nprocs;
        args.src.info.datatype    = dt;
        args.src.info.mem_type    = mem_type;
    }

    args.dst.info.buffer      = rbuf;
    args.dst.info.count       = single_rank_count * nprocs;
    args.dst.info.datatype    = dt;
    args.dst.info.mem_type    = mem_type;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAlltoall::set_input(int iter_persistent)
{
    size_t      dt_size           = ucc_dt_size(dt);
    size_t      single_rank_count = msgsize / dt_size;
    MPI_Request req;
    void *      buf;
    int         rank, nprocs, completed;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);
    if (inplace) {
        buf = rbuf;
    } else {
        buf = sbuf;
    }
    init_buffer(buf, single_rank_count * nprocs, dt, mem_type,
                rank * (iter_persistent + 1));
    UCC_CHECK(ucc_mc_memcpy(check_buf, buf,
                            single_rank_count * nprocs * dt_size,
                            UCC_MEMORY_TYPE_HOST, mem_type));

    if (is_onesided && persistent) {
        MPI_Ibarrier(team.comm, &req);
        do {
            MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
            ucc_context_progress(team.ctx);
        } while(!completed);
    }
    return UCC_OK;
}

ucc_status_t TestAlltoall::check()
{
    int         size, completed;
    size_t      single_rank_count;
    MPI_Request req;

    MPI_Comm_size(team.comm, &size);
    single_rank_count = args.src.info.count / size;

    MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, check_buf,
                  single_rank_count, ucc_dt_to_mpi(dt), team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check_buf, single_rank_count * size, dt,
                           mem_type);
}
