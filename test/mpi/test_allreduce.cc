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
    TestCase(_team, UCC_COLL_TYPE_ALLREDUCE, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count   = _msgsize/dt_size;
    int    rank;

    MPI_Comm_rank(team.comm, &rank);
    op = _op;
    dt = _dt;

    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT,
                    team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize, _mt));
    rbuf      = rbuf_mc_header->addr;
    check_buf = ucc_malloc(_msgsize, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    if (TEST_NO_INPLACE == inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        sbuf = sbuf_mc_header->addr;
        args.src.info.buffer      = sbuf;
        args.src.info.count       = count;
        args.src.info.datatype    = _dt;
        args.src.info.mem_type    = _mt;
    } else {
        args.mask                 = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags                = UCC_COLL_ARGS_FLAG_IN_PLACE;
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
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAllreduce::set_input()
{
    size_t dt_size = ucc_dt_size(dt);
    size_t count   = msgsize / dt_size;
    int    rank;
    void  *buf;

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

ucc_status_t TestAllreduce::reset_sbuf()
{
    return UCC_OK;
}

ucc_status_t TestAllreduce::check()
{
    size_t       count = args.dst.info.count;
    MPI_Request  req;
    int          completed;
    ucc_status_t status;

    MPI_Iallreduce(MPI_IN_PLACE, check_buf, count, ucc_dt_to_mpi(dt),
                   op == UCC_OP_AVG ? MPI_SUM : ucc_op_to_mpi(op), team.comm,
                   &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    if (op == UCC_OP_AVG) {
        status = divide_buffer(check_buf, team.team->size, count, dt);
        if (status != UCC_OK) {
            return status;
        }
    }

    return compare_buffers(rbuf, check_buf, count, dt, mem_type);
}

std::string TestAllreduce::str() {
    return std::string("tc=")+ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
