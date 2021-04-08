/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

TestAllgatherv::TestAllgatherv(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                               ucc_memory_type_t _mt, ucc_test_team_t &_team,
                               size_t _max_size) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count = _msgsize/dt_size;
    int rank, size;
    counts = NULL;
    displacements = NULL;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);
    args.coll_type = UCC_COLL_TYPE_ALLGATHERV;

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (_msgsize*size),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc((void**)&counts, size * sizeof(uint32_t),
                           UCC_MEMORY_TYPE_HOST));
    UCC_CHECK(ucc_mc_alloc((void**)&displacements, size * sizeof(uint32_t),
                           UCC_MEMORY_TYPE_HOST));
    UCC_CHECK(ucc_mc_alloc(&rbuf, _msgsize*size, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_rbuf, _msgsize*size, UCC_MEMORY_TYPE_HOST));
    for (int i = 0; i < size; i++) {
        counts[i] = count;
        displacements[i] = i * count;
    }
    if (TEST_NO_INPLACE == inplace) {
        args.mask = 0;
        UCC_CHECK(ucc_mc_alloc(&sbuf, _msgsize, _mt));
        init_buffer(sbuf, count, TEST_DT, _mt, rank);
        UCC_ALLOC_COPY_BUF(check_sbuf, UCC_MEMORY_TYPE_HOST, sbuf, _mt, _msgsize);
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer((void*)((ptrdiff_t)rbuf + rank*count*dt_size),
                    count, TEST_DT, _mt, rank);
        init_buffer((void*)((ptrdiff_t)check_rbuf + rank*count*dt_size),
                    count, TEST_DT, UCC_MEMORY_TYPE_HOST, rank);
    }
    if (TEST_NO_INPLACE == inplace) {
        args.src.info.buffer          = sbuf;
        args.src.info.datatype        = TEST_DT;
        args.src.info.mem_type        = _mt;
        args.src.info.count           = count;
    }
    args.dst.info_v.buffer        = rbuf;
    args.dst.info_v.counts        = (ucc_count_t*)counts;
    args.dst.info_v.displacements = (ucc_aint_t*)displacements;
    args.dst.info_v.datatype      = TEST_DT;
    args.dst.info_v.mem_type      = _mt;
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
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
    size_t       count = counts[0];
    MPI_Datatype dt    = ucc_dt_to_mpi(TEST_DT);
    int          size;
    MPI_Comm_size(team.comm, &size);
    MPI_Allgatherv((inplace == TEST_INPLACE) ? MPI_IN_PLACE : check_sbuf, count, dt,
                   check_rbuf, (int*)counts, (int*)displacements, dt, team.comm);
    return compare_buffers(rbuf, check_rbuf, count*size, TEST_DT, mem_type);
}
