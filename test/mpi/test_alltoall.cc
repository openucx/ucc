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
                           size_t _max_size, void ** buffers) :
    TestCase(_team, UCC_COLL_TYPE_ALLTOALL, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size           = ucc_dt_size(TEST_DT);
    size_t single_rank_count = _msgsize / dt_size;
    bool   is_onesided       = (buffers != nullptr);
    void  *work_buf          = nullptr;
    int    rank;
    int    nprocs;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (_msgsize * nprocs),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    if (!is_onesided) {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize * nprocs, _mt));
        rbuf      = rbuf_mc_header->addr;
    } else {
        sbuf     = buffers[MEM_SEND_SEGMENT];
        rbuf     = buffers[MEM_RECV_SEGMENT];
        work_buf = buffers[MEM_WORK_SEGMENT];
    }

    check_buf = ucc_malloc(_msgsize * nprocs, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    if (TEST_NO_INPLACE == inplace) {
        if (!is_onesided) {
            UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize * nprocs, _mt));
            sbuf = sbuf_mc_header->addr;
        }
    } else {
        args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
    if (is_onesided) {
        args.mask =
            UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;
        args.flags |= UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS;
        args.global_work_buffer = work_buf;
    }

    if (TEST_NO_INPLACE == inplace) {
        args.src.info.buffer      = sbuf;
        args.src.info.count       = single_rank_count * nprocs;
        args.src.info.datatype    = TEST_DT;
        args.src.info.mem_type    = _mt;
    }

    args.dst.info.buffer      = rbuf;
    args.dst.info.count       = single_rank_count * nprocs;
    args.dst.info.datatype    = TEST_DT;
    args.dst.info.mem_type    = _mt;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAlltoall::set_input()
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t single_rank_count = msgsize / dt_size;
    void  *buf;
    int    rank, nprocs;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);
    if (TEST_NO_INPLACE == inplace) {
        buf = sbuf;
    } else {
        buf = rbuf;
    }
    init_buffer(buf, single_rank_count * nprocs, TEST_DT, mem_type, rank);
    UCC_CHECK(ucc_mc_memcpy(check_buf, buf, single_rank_count * nprocs * dt_size,
                            UCC_MEMORY_TYPE_HOST, mem_type));
    return UCC_OK;
}

ucc_status_t TestAlltoall::reset_sbuf()
{
    size_t dt_size           = ucc_dt_size(TEST_DT);
    size_t single_rank_count = msgsize / dt_size;
    int    rank, nprocs;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);
    if (TEST_NO_INPLACE == inplace) {
        init_buffer(sbuf, single_rank_count * nprocs, TEST_DT, mem_type, rank);
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

    MPI_Ialltoall(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                  check_buf, single_rank_count, ucc_dt_to_mpi(TEST_DT),
                  team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check_buf, single_rank_count * size, TEST_DT,
                           mem_type);
}
