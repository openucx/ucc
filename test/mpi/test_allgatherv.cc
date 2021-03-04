/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

TestAllgatherv::TestAllgatherv(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                               ucc_memory_type_t _mt, ucc_test_team_t &_team) :
    TestCase(_team, _mt, _msgsize, _inplace)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count = _msgsize/dt_size;
    int rank, size;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    UCC_CHECK(ucc_mc_alloc((void**)&counts, size * sizeof(uint32_t),
                           UCC_MEMORY_TYPE_HOST));
    UCC_CHECK(ucc_mc_alloc((void**)&displacements, size * sizeof(uint32_t),
                           UCC_MEMORY_TYPE_HOST));
    UCC_CHECK(ucc_mc_alloc(&rbuf, _msgsize*size, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_buf, _msgsize*size, _mt));
    for (int i = 0; i < size; i++) {
        counts[i] = count;
        displacements[i] = i * count;
    }
    if (TEST_NO_INPLACE == inplace) {
        args.mask = 0;
        UCC_CHECK(ucc_mc_alloc(&sbuf, _msgsize, _mt));
        init_buffer(sbuf, count, TEST_DT, _mt, rank);
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer((void*)((ptrdiff_t)rbuf + rank*count*dt_size),
                    count, TEST_DT, _mt, rank);
        init_buffer((void*)((ptrdiff_t)check_buf + rank*count*dt_size),
                    count, TEST_DT, _mt, rank);
    }
    args.coll_type                = UCC_COLL_TYPE_ALLGATHERV;
    args.src.info.buffer          = sbuf;
    args.src.info.count           = count;
    args.src.info.datatype        = TEST_DT;
    args.src.info.mem_type        = _mt;
    args.dst.info_v.buffer        = rbuf;
    args.dst.info_v.counts        = (ucc_count_t*)counts;
    args.dst.info_v.displacements = (ucc_aint_t*)displacements;
    args.dst.info_v.datatype      = TEST_DT;
    args.dst.info_v.mem_type      = _mt;
    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

TestAllgatherv::~TestAllgatherv() {
    if (counts) {
        ucc_mc_free(counts, UCC_MEMORY_TYPE_HOST);
    }
    if (displacements) {
        ucc_mc_free(displacements, UCC_MEMORY_TYPE_HOST);
    }
}
ucc_status_t TestAllgatherv::check()
{
    size_t       count = args.src.info.count;
    MPI_Datatype dt    = ucc_dt_to_mpi(TEST_DT);
    int          size;
    MPI_Comm_size(team.comm, &size);
    MPI_Allgatherv((inplace == TEST_INPLACE) ? MPI_IN_PLACE : sbuf, count, dt,
                   check_buf, (int*)counts, (int*)displacements, dt, team.comm);
    return compare_buffers(rbuf, check_buf, count*size, TEST_DT, mem_type);
}

std::string TestAllgatherv::str() {
    return std::string("tc=")+std::string(ucc_coll_type_str(args.coll_type)) +
        " team=" + team_str(team.type) + " msgsize=" +
        std::to_string(msgsize) + " inplace=" +
        (inplace == TEST_INPLACE ? "1" : "0");
}
