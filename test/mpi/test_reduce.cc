/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestReduce::TestReduce(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_REDUCE, params)
{
    size_t dt_size = ucc_dt_size(params.dt);
    size_t count   = msgsize/dt_size;
    int rank;

    MPI_Comm_rank(team.comm, &rank);
    dt   = params.dt;
    op   = params.op;
    root = params.root;

    if (skip_reduce(test_max_size < msgsize, TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }
    check_buf = ucc_malloc(msgsize, "check buf");
    UCC_MALLOC_CHECK(check_buf);

    if (rank == root) {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize, mem_type));
        rbuf = rbuf_mc_header->addr;
        args.dst.info.buffer   = rbuf;
        args.dst.info.count    = count;
        args.dst.info.datatype = dt;
        args.dst.info.mem_type = mem_type;
    }
    if ((rank != root) || (inplace == TEST_NO_INPLACE)) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, msgsize, mem_type));
        sbuf = sbuf_mc_header->addr;
    }
    if (inplace == TEST_INPLACE) {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    args.op                   = op;
    args.src.info.buffer      = sbuf;
    args.src.info.count       = count;
    args.src.info.datatype    = dt;
    args.src.info.mem_type    = mem_type;
    args.root                 = root;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestReduce::set_input()
{
    size_t dt_size = ucc_dt_size(dt);
    size_t count   = msgsize / dt_size;
    int    rank;
    void  *buf;

    MPI_Comm_rank(team.comm, &rank);
    if (inplace && rank == root) {
        buf = rbuf;
    } else {
        buf = sbuf;
    }

    init_buffer(buf, count, dt, mem_type, rank);
    UCC_CHECK(ucc_mc_memcpy(check_buf, buf, count * dt_size,
                            UCC_MEMORY_TYPE_HOST, mem_type));
    return UCC_OK;
}

ucc_status_t TestReduce::reset_sbuf()
{
    return UCC_OK;
}

ucc_status_t TestReduce::check()
{
    ucc_status_t status;
    size_t       count = args.src.info.count;
    int          rank, completed;
    MPI_Request  req;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Ireduce((root == rank) ? MPI_IN_PLACE : check_buf, check_buf,
                count, ucc_dt_to_mpi(dt),
                op == UCC_OP_AVG ? MPI_SUM : ucc_op_to_mpi(op), root, team.comm,
                &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    if (rank == root && op == UCC_OP_AVG) {
        status = divide_buffer(check_buf, team.team->size, count, dt);
        if (status != UCC_OK) {
            return status;
        }
    }
    return (rank != root) ? UCC_OK :
        compare_buffers(rbuf, check_buf, count, dt, mem_type);
}

std::string TestReduce::str() {
    return std::string("tc=")+ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op) +
        " root=" + std::to_string(root);
}
