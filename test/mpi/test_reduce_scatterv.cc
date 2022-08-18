/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestReduceScatterv::TestReduceScatterv(ucc_test_team_t &_team, TestCaseParams &params)
    : TestCase(_team, UCC_COLL_TYPE_REDUCE_SCATTERV, params)
{
    size_t dt_size = ucc_dt_size(params.dt);
    size_t count   = msgsize / dt_size;
    int    rank, comm_size;
    counts = NULL;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &comm_size);
    op             = params.op;
    dt             = params.dt;

    if (skip_reduce(test_max_size < msgsize, TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }
    counts = (int *)ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK(counts);

    size_t left  = count;
    size_t total = 0;
    for (int i = 0; i < comm_size; i++) {
        size_t c = 2 + i * 2;
        if (left < c) {
            c = left;
        }
        if (i == comm_size - 1) {
            counts[i] = left;
        } else {
            counts[i] = c;
        }

        if (left > 0) {
            left -= c;
        }
        total += counts[i];
    }
    ucc_assert(total == count);
    check_buf = ucc_malloc(msgsize, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, counts[rank] * dt_size, mem_type));
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize, mem_type));
        rbuf = rbuf_mc_header->addr;
        sbuf = sbuf_mc_header->addr;
    } else {
        args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize, mem_type));
        rbuf = rbuf_mc_header->addr;
    }

    if (inplace == TEST_NO_INPLACE) {
        args.src.info.buffer   = sbuf;
        args.src.info.count    = count;
        args.src.info.datatype = dt;
        args.src.info.mem_type = mem_type;
    }
    args.op                  = op;
    args.dst.info_v.counts   = (ucc_count_t *)counts;
    args.dst.info_v.buffer   = rbuf;
    args.dst.info_v.datatype = dt;
    args.dst.info_v.mem_type = mem_type;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestReduceScatterv::reset_sbuf()
{
    return UCC_OK;
}

ucc_status_t TestReduceScatterv::set_input()
{
    size_t dt_size = ucc_dt_size(dt);
    size_t count   = msgsize / dt_size;
    void * buf;
    int    rank;

    MPI_Comm_rank(team.comm, &rank);
    if (TEST_NO_INPLACE == inplace) {
        buf = sbuf;
    } else {
        buf = rbuf;
    }
    init_buffer(buf, count, dt, mem_type, rank);
    UCC_CHECK(ucc_mc_memcpy(check_buf, buf, count * dt_size,
                            UCC_MEMORY_TYPE_HOST, mem_type));
    return UCC_OK;
}

ucc_status_t TestReduceScatterv::check()
{
    ucc_status_t status;
    int          comm_rank, comm_size, completed;
    MPI_Request  req;
    size_t       offset;

    MPI_Comm_rank(team.comm, &comm_rank);
    MPI_Comm_size(team.comm, &comm_size);
    MPI_Ireduce_scatter(MPI_IN_PLACE, check_buf, counts, ucc_dt_to_mpi(dt),
                        op == UCC_OP_AVG ? MPI_SUM : ucc_op_to_mpi(op),
                        team.comm, &req);

    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while (!completed);

    if (op == UCC_OP_AVG) {
        status =
            divide_buffer(check_buf, team.team->size, counts[comm_rank], dt);
        if (status != UCC_OK) {
            return status;
        }
    }
    if (inplace) {
        offset = 0;
        for (int i = 0; i < comm_rank; i++) {
            offset += counts[i];
        }
        return compare_buffers(PTR_OFFSET(rbuf, offset * ucc_dt_size(dt)),
                               check_buf, counts[comm_rank], dt, mem_type);
    }
    return compare_buffers(rbuf, check_buf, counts[comm_rank], dt, mem_type);
}

TestReduceScatterv::~TestReduceScatterv()
{
    if (counts) {
        ucc_free(counts);
    }
}

std::string TestReduceScatterv::str()
{
    return std::string("tc=") + ucc_coll_type_str(args.coll_type) +
           " team=" + team_str(team.type) +
           " msgsize=" + std::to_string(msgsize) +
           " inplace=" + (inplace == TEST_INPLACE ? "1" : "0") +
           " dt=" + ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
