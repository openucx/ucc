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
        args.dst.info.mem_type = _mt;
    }
    if (rank != root || !inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        sbuf = sbuf_mc_header->addr;
    }
    if (inplace) {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }
    init_buffer((inplace && rank == root) ? rbuf : sbuf, count, dt, _mt, rank);
    UCC_ALLOC_COPY_BUF(check_sbuf_mc_header, UCC_MEMORY_TYPE_HOST,
                       (inplace && rank == root) ? rbuf : sbuf, _mt, _msgsize);
    check_sbuf = check_sbuf_mc_header->addr;

    args.mask                |= UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS;
    args.reduce.predefined_op = _op;

    args.src.info.buffer      = sbuf;
    args.src.info.count       = count;
    args.src.info.datatype    = _dt;
    args.src.info.mem_type    = _mt;
    args.root                 = root;
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestReduce::check()
{
    size_t       count = args.src.info.count;
    int          rank, completed;
    MPI_Request  req;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Ireduce(check_sbuf, check_rbuf, count, ucc_dt_to_mpi(dt),
                        ucc_op_to_mpi(op), root, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);
    return (rank != root) ? UCC_OK :
        compare_buffers(rbuf, check_rbuf, count, dt, mem_type);
}
