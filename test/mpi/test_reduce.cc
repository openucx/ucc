/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestReduce::TestReduce(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                     ucc_datatype_t _dt, ucc_reduction_op_t _op,
                     ucc_memory_type_t _mt, int _root, ucc_test_team_t &_team,
                     size_t _max_size) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count = _msgsize/dt_size;
    int rank;
    MPI_Comm_rank(team.comm, &rank);
    dt = _dt;
    op = _op;
    root = _root;
    args.coll_type = UCC_COLL_TYPE_REDUCE;

    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT,
                    team.comm)) {
        return;
    }

    if (rank == root) {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize, _mt));
        rbuf = rbuf_mc_header->addr;
        check_rbuf = ucc_malloc(_msgsize, "check rbuf");
        UCC_MALLOC_CHECK(check_rbuf);
        args.dst.info.buffer   = rbuf;
        args.dst.info.count    = count;
        args.dst.info.datatype = _dt;
        args.dst.info.mem_type = _mt;
    }
    if ((rank != root) || (inplace == TEST_NO_INPLACE)) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        sbuf = sbuf_mc_header->addr;
    }
    if (inplace == TEST_INPLACE) {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
    UCC_CHECK(ucc_mc_alloc(&check_sbuf_mc_header, _msgsize,
                           UCC_MEMORY_TYPE_HOST));
    check_sbuf = check_sbuf_mc_header->addr;

    args.op                   = _op;
    args.src.info.buffer      = sbuf;
    args.src.info.count       = count;
    args.src.info.datatype    = _dt;
    args.src.info.mem_type    = _mt;
    args.root                 = root;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestReduce::set_input()
{
    size_t dt_size = ucc_dt_size(dt);
    size_t count   = msgsize / dt_size;
    int rank;
    void *buf;

    MPI_Comm_rank(team.comm, &rank);
    if (inplace && rank == root) {
        buf = rbuf;
    } else {
        buf = sbuf;
    }

    init_buffer(buf, count, dt, mem_type, rank);
    UCC_CHECK(ucc_mc_memcpy(check_sbuf, buf, count * dt_size,
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
    MPI_Ireduce(check_sbuf, check_rbuf, count, ucc_dt_to_mpi(dt),
                op == UCC_OP_AVG ? MPI_SUM : ucc_op_to_mpi(op), root, team.comm,
                &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);
    if (rank == root && op == UCC_OP_AVG) {
        status = divide_buffer(check_rbuf, team.team->size, count, dt);
        if (status != UCC_OK) {
            return status;
        }
    }
    return (rank != root) ? UCC_OK :
        compare_buffers(rbuf, check_rbuf, count, dt, mem_type);
}
