/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestAlltoall::TestAlltoall(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                           ucc_datatype_t _dt, ucc_memory_type_t _mt,
                           ucc_test_team_t &_team) :
    TestCase(_team, _mt, _msgsize, _inplace)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count = _msgsize/dt_size;
    int rank;
    int nprocs;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);
    dt = _dt;

    if (skip(test_max_size && (test_max_size < (_msgsize * nprocs)),
             TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf, _msgsize * nprocs, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_buf, _msgsize * nprocs, _mt));
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf, _msgsize * nprocs, _mt));
        init_buffer(sbuf, count * nprocs, dt, _mt, rank);
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer(rbuf, count * nprocs, dt, _mt, rank);
        init_buffer(check_buf, count * nprocs, dt, _mt, rank);
    }

    args.coll_type            = UCC_COLL_TYPE_ALLTOALL;

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

ucc_status_t TestAlltoall::check()
{
    size_t count = args.src.info.count;
    MPI_Alltoall(inplace ? MPI_IN_PLACE : sbuf, count, ucc_dt_to_mpi(dt),
                 check_buf, count, ucc_dt_to_mpi(dt), team.comm);
    return compare_buffers(rbuf, check_buf, count, dt, mem_type);
}

std::string TestAlltoall::str()
{
    return std::string("tc=") + ucc_coll_type_str(UCC_COLL_TYPE_ALLTOALL) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt);
}
