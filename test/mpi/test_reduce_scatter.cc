/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestReduceScatter::TestReduceScatter(size_t _msgsize,
                                     ucc_test_mpi_inplace_t _inplace,
                                     ucc_datatype_t _dt, ucc_reduction_op_t _op,
                                     ucc_memory_type_t _mt,
                                     ucc_test_team_t &_team, size_t _max_size) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size),
    recvcounts(nullptr)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count = _msgsize/dt_size;
    int rank, comm_size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &comm_size);
    op = _op;
    dt = _dt;
    args.coll_type = UCC_COLL_TYPE_REDUCE_SCATTER;

    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT,
                    team.comm) ||
        skip_reduce((count % comm_size != 0), TEST_SKIP_NOT_SUPPORTED,
                    team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize, _mt));
    rbuf       = rbuf_mc_header->addr;
    check_rbuf = ucc_malloc(_msgsize, "check rbuf");
    UCC_MALLOC_CHECK(check_rbuf);
    recvcounts = (int*)ucc_malloc(comm_size * sizeof(*recvcounts));
    UCC_MALLOC_CHECK(recvcounts);

    for (int i = 0; i < comm_size; i++) {
        recvcounts[i] = count / comm_size;
    }

    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        sbuf = sbuf_mc_header->addr;
        init_buffer(sbuf, count, dt, _mt, rank);
        UCC_ALLOC_COPY_BUF(check_sbuf_mc_header, UCC_MEMORY_TYPE_HOST, sbuf,
                           _mt, _msgsize);
        check_sbuf = check_sbuf_mc_header->addr;
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer(rbuf, count, dt, _mt, rank);
        init_buffer(check_rbuf, count, dt, UCC_MEMORY_TYPE_HOST, rank);
    }

    args.mask                |= UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS;
    args.reduce.predefined_op = _op;

    args.src.info.buffer      = sbuf;
    args.src.info.count       = count;
    args.src.info.datatype    = _dt;
    args.src.info.mem_type    = _mt;

    args.dst.info.buffer      = rbuf;
    args.dst.info.count       = count;
    args.dst.info.datatype    = _dt;
    args.dst.info.mem_type    = _mt;
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestReduceScatter::check()
{
    int comm_rank, comm_size;

    MPI_Comm_rank(team.comm, &comm_rank);
    MPI_Comm_size(team.comm, &comm_size);
    MPI_Reduce_scatter(inplace ? MPI_IN_PLACE : check_sbuf, check_rbuf,
                       recvcounts, ucc_dt_to_mpi(dt), ucc_op_to_mpi(op),
                       team.comm);
    if (inplace) {
        return compare_buffers(PTR_OFFSET(rbuf, comm_rank * msgsize/comm_size),
                               check_rbuf, recvcounts[0], dt, mem_type);
    }
    return compare_buffers(rbuf, check_rbuf, recvcounts[0], dt, mem_type);
}

TestReduceScatter::~TestReduceScatter() {
    if (recvcounts) {
        ucc_free(recvcounts);
    }
}

std::string TestReduceScatter::str() {
    return std::string("tc=")+ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
