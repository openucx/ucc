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
    TestCase(_team, UCC_COLL_TYPE_REDUCE_SCATTER, _mt, _msgsize, _inplace,
             _max_size)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count   = _msgsize / dt_size;
    int    rank, comm_size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &comm_size);
    op = _op;
    dt = _dt;

    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT, team.comm) ||
        skip_reduce((count < comm_size), TEST_SKIP_NOT_SUPPORTED, team.comm)) {
        return;
    }
    check_buf = ucc_malloc( _msgsize, "check buf");
    UCC_MALLOC_CHECK(check_buf);

    count   = count - (count % comm_size);
    msgsize = _msgsize = count * dt_size;

    if (TEST_NO_INPLACE == inplace) {
        args.dst.info.count = count / comm_size;
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize / comm_size, _mt));
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        rbuf = rbuf_mc_header->addr;
        sbuf = sbuf_mc_header->addr;
    } else {
        args.mask           = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags          = UCC_COLL_ARGS_FLAG_IN_PLACE;
        args.dst.info.count = count;
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize, _mt));
        rbuf       = rbuf_mc_header->addr;
    }

    if (inplace == TEST_NO_INPLACE) {
        args.src.info.buffer      = sbuf;
        args.src.info.count       = count;
        args.src.info.datatype    = _dt;
        args.src.info.mem_type    = _mt;
    }
    args.op                   = _op;
    args.dst.info.buffer      = rbuf;
    args.dst.info.datatype    = _dt;
    args.dst.info.mem_type    = _mt;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestReduceScatter::set_input()
{
    size_t dt_size = ucc_dt_size(dt);
    size_t count   = msgsize / dt_size;
    void  *buf;
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

ucc_status_t TestReduceScatter::reset_sbuf()
{
    return UCC_OK;
}

ucc_status_t TestReduceScatter::check()
{
    ucc_status_t status;
    int          comm_rank, comm_size, completed;
    size_t       block_size, block_count;
    MPI_Request  req;

    MPI_Comm_rank(team.comm, &comm_rank);
    MPI_Comm_size(team.comm, &comm_size);
    block_size  = msgsize / comm_size;
    block_count = block_size / ucc_dt_size(dt);

    MPI_Ireduce_scatter_block(MPI_IN_PLACE, check_buf,
                              block_count, ucc_dt_to_mpi(dt),
                              op == UCC_OP_AVG ? MPI_SUM : ucc_op_to_mpi(op),
                              team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    if (op == UCC_OP_AVG) {
        status = divide_buffer(check_buf, team.team->size, block_count, dt);
        if (status != UCC_OK) {
            return status;
        }
    }
    if (inplace) {
        return compare_buffers(PTR_OFFSET(rbuf, comm_rank * block_size),
                               check_buf, block_count, dt, mem_type);
    }
    return compare_buffers(rbuf, check_buf, block_count, dt, mem_type);
}

TestReduceScatter::~TestReduceScatter() {}

std::string TestReduceScatter::str() {
    return std::string("tc=")+ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
