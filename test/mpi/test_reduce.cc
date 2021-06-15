/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestReduce::TestReduce(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                       int _root, ucc_datatype_t _dt, ucc_reduction_op_t _op,
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
    args.coll_type = UCC_COLL_TYPE_REDUCE;

    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT,
                    team.comm)) {
        return;
    }

    if (rank == _root) {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize, _mt));
        rbuf       = rbuf_mc_header->addr;
        check_rbuf = ucc_malloc(_msgsize, "check rbuf");
        UCC_MALLOC_CHECK(check_rbuf);
        if (inplace == TEST_INPLACE) {
            init_buffer(rbuf, count, dt, _mt, rank);
            init_buffer(check_rbuf, count, dt, UCC_MEMORY_TYPE_HOST, rank);
        }
    }

    if (inplace == TEST_INPLACE) {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if ((TEST_NO_INPLACE == inplace) || (rank != _root)) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        sbuf = sbuf_mc_header->addr;
        init_buffer(sbuf, count, dt, _mt, rank);
        UCC_ALLOC_COPY_BUF(check_sbuf_mc_header, UCC_MEMORY_TYPE_HOST, sbuf,
                           _mt, _msgsize);
        check_sbuf = check_sbuf_mc_header->addr;
    }

    args.mask                |= UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS;
    args.reduce.predefined_op = _op;

    args.root                 = _root;

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

ucc_status_t TestReduce::check()
{
    int comm_rank;

    MPI_Comm_rank(team.comm, &comm_rank);

    if (inplace == TEST_NO_INPLACE) {
        MPI_Reduce(check_sbuf, check_rbuf, args.src.info.count,
                   ucc_dt_to_mpi(dt), ucc_op_to_mpi(op), args.root, team.comm);
    } else {
        if (comm_rank == args.root) {
            MPI_Reduce(MPI_IN_PLACE, check_rbuf, args.src.info.count,
                       ucc_dt_to_mpi(dt), ucc_op_to_mpi(op), args.root,
                       team.comm);
        } else {
            MPI_Reduce(check_sbuf, check_rbuf, args.src.info.count,
                       ucc_dt_to_mpi(dt), ucc_op_to_mpi(op), args.root,
                       team.comm);
        }
    }
    if (comm_rank == args.root) {
        return compare_buffers(rbuf, check_rbuf, args.src.info.count, dt,
                               mem_type);
    }
    return UCC_OK;
}

std::string TestReduce::str() {
    return std::string("tc=")+ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
