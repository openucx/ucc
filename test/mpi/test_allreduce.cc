/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestAllreduce::TestAllreduce(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                             ucc_datatype_t _dt, ucc_reduction_op_t _op,
                             ucc_memory_type_t _mt, ucc_test_team_t &_team,
                             size_t _max_size) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count = _msgsize/dt_size;
    int rank;

    MPI_Comm_rank(team.comm, &rank);
    op = _op;
    dt = _dt;
    args.coll_type = UCC_COLL_TYPE_ALLREDUCE;

    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT,
                    team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize, _mt));
    rbuf       = rbuf_mc_header->addr;
    check_rbuf = ucc_malloc(_msgsize, "check rbuf");
    UCC_MALLOC_CHECK(check_rbuf);
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        sbuf = sbuf_mc_header->addr;
        init_buffer(sbuf, count, dt, _mt, rank);
        UCC_ALLOC_COPY_BUF(check_sbuf_mc_header, UCC_MEMORY_TYPE_HOST, sbuf,
                           _mt, _msgsize);
        check_sbuf = check_sbuf_mc_header->addr;

        args.src.info.buffer      = sbuf;
        args.src.info.count       = count;
        args.src.info.datatype    = _dt;
        args.src.info.mem_type    = _mt;
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer(rbuf, count, dt, _mt, rank);
        init_buffer(check_rbuf, count, dt, UCC_MEMORY_TYPE_HOST, rank);

        args.src.info.buffer      = NULL;
        args.src.info.count       = SIZE_MAX;
        args.src.info.datatype    = (ucc_datatype_t)-1;
        args.src.info.mem_type    = UCC_MEMORY_TYPE_UNKNOWN;
    }

    args.op                   = _op;
    args.dst.info.buffer      = rbuf;
    args.dst.info.count       = count;
    args.dst.info.datatype    = _dt;
    args.dst.info.mem_type    = _mt;
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAllreduce::check()
{
    size_t      count = args.dst.info.count;
    MPI_Request req;
    int         completed;

    MPI_Iallreduce(inplace ? MPI_IN_PLACE : check_sbuf, check_rbuf, count,
                   ucc_dt_to_mpi(dt), ucc_op_to_mpi(op), team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check_rbuf, count, dt, mem_type);
}

std::string TestAllreduce::str() {
    return std::string("tc=")+ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
