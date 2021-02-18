/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"
TestAllreduce::TestAllreduce(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                             ucc_datatype_t _dt, ucc_reduction_op_t _op,
                             ucc_memory_type_t _mt, ucc_test_team_t &_team) :
    TestCase(_team, _mt, _msgsize, _inplace)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count = _msgsize/dt_size;
    int rank;
    MPI_Comm_rank(team.comm, &rank);
    op = _op;
    dt = _dt;

    UCC_CHECK(ucc_mc_alloc(&rbuf, _msgsize, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_buf, _msgsize, _mt));
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf, _msgsize, _mt));
        init_buffer(sbuf, count, dt, _mt, rank);
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer(rbuf, count, dt, _mt, rank);
        init_buffer(check_buf, count, dt, _mt, rank);
    }

    args.coll_type            = UCC_COLL_TYPE_ALLREDUCE;
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
    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

ucc_status_t TestAllreduce::check()
{
    size_t count = args.src.info.count;
    MPI_Allreduce(inplace ? MPI_IN_PLACE : sbuf, check_buf, count,
                  ucc_dt_to_mpi(dt), ucc_op_to_mpi(op), team.comm);
    return compare_buffers(rbuf, check_buf, count, dt, mem_type);
}

std::string TestAllreduce::str() {
    return std::string("tc=")+ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
