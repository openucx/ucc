/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

TestReduceScatterv::TestReduceScatterv(size_t _msgsize,
                                     ucc_test_mpi_inplace_t _inplace,
                                     ucc_datatype_t _dt, ucc_reduction_op_t _op,
                                     ucc_memory_type_t _mt,
                                     ucc_test_team_t &_team, size_t _max_size) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count = _msgsize/dt_size;
    int rank, comm_size;
    counts = NULL;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &comm_size);
    op = _op;
    dt = _dt;
    args.coll_type = UCC_COLL_TYPE_REDUCE_SCATTERV;


    if (skip_reduce(test_max_size < _msgsize, TEST_SKIP_MEM_LIMIT,
                    team.comm)) {
        return;
    }
    counts = (int *) ucc_malloc(comm_size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK(counts);

    size_t left = count;
    size_t total = 0;
    for (int i = 0; i < comm_size; i++) {
        size_t c = 2 + i*2;
        if (left < c) {
            c = left;
        }
        if (i == comm_size - 1) {
            counts[i] = left;
        } else {
            counts[i] = c;
        }

        if (left > 0) {
            left -= c;
        }
        total += counts[i];
    }
    ucc_assert(total == count);

    if (TEST_NO_INPLACE == inplace) {


        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, counts[rank] * dt_size, _mt));
        rbuf       = rbuf_mc_header->addr;
        check_rbuf = ucc_malloc( counts[rank] * dt_size, "check rbuf");
        UCC_MALLOC_CHECK(check_rbuf);
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, _msgsize, _mt));
        sbuf = sbuf_mc_header->addr;
        init_buffer(sbuf, count, dt, _mt, rank);
        UCC_ALLOC_COPY_BUF(check_sbuf_mc_header, UCC_MEMORY_TYPE_HOST, sbuf,
                           _mt, _msgsize);
        check_sbuf = check_sbuf_mc_header->addr;
  } else {
        args.mask           = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags          = UCC_COLL_ARGS_FLAG_IN_PLACE;

        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize, _mt));
        rbuf       = rbuf_mc_header->addr;
        check_rbuf = ucc_malloc(_msgsize, "check rbuf");
        UCC_MALLOC_CHECK(check_rbuf);
        init_buffer(rbuf, count, dt, _mt, rank);
        init_buffer(check_rbuf, count, dt, UCC_MEMORY_TYPE_HOST, rank);
    }

    args.op = _op;

    if (inplace == TEST_NO_INPLACE) {
        args.src.info.buffer      = sbuf;
        args.src.info.count       = count;
        args.src.info.datatype    = _dt;
        args.src.info.mem_type    = _mt;
    }
    args.dst.info_v.counts = (ucc_count_t*)counts;
    args.dst.info_v.buffer = rbuf;
    args.dst.info_v.datatype = _dt;
    args.dst.info_v.mem_type = _mt;
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestReduceScatterv::check()
{
    ucc_status_t status;
    int comm_rank, comm_size;

    MPI_Comm_rank(team.comm, &comm_rank);
    MPI_Comm_size(team.comm, &comm_size);
    MPI_Reduce_scatter(inplace ? MPI_IN_PLACE : check_sbuf, check_rbuf,
                             counts, ucc_dt_to_mpi(dt),
                             op == UCC_OP_AVG ? MPI_SUM : ucc_op_to_mpi(op),
                             team.comm);
    if (op == UCC_OP_AVG) {
        status = divide_buffer(check_rbuf, team.team->size, counts[comm_rank], dt);
        if (status != UCC_OK) {
            return status;
        }
    }
    if (inplace) {
        ucc_assert(0);
        // return compare_buffers(PTR_OFFSET(rbuf, comm_rank * block_size),
        //                        check_rbuf, block_count, dt, mem_type);
    }
    return compare_buffers(rbuf, check_rbuf, counts[comm_rank], dt, mem_type);
}

TestReduceScatterv::~TestReduceScatterv() {
    if (counts) {
        ucc_free(counts);
    }
}

std::string TestReduceScatterv::str() {
    return std::string("tc=")+ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0") + " dt=" +
        ucc_datatype_str(dt) + " op=" + ucc_reduction_op_str(op);
}
