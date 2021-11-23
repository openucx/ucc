/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

static void fill_counts_and_displacements(int size, int count,
                                          int *counts, int *displs)
{
    int bias = count / 2;
    int i;

    counts[0] = count - bias;
    displs[0] = 0;
    for (i = 1; i < size - 1; i++) {
        if (i % 2 == 0) {
            counts[i] = count - bias;
        } else {
            counts[i] = count + bias;
        }
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    if (size % 2 == 0) {
        counts[size - 1] = count + bias;
    } else {
        counts[size - 1] = count;
    }
    displs[size - 1] = displs[size - 2] + counts[size - 2];
}

TestAllgatherv::TestAllgatherv(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                               ucc_memory_type_t _mt, ucc_test_team_t &_team,
                               size_t _max_size) :
    TestCase(_team, _mt, _msgsize, _inplace, _max_size)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count = _msgsize / dt_size;
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

    counts = (int *) ucc_malloc(size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK(counts);
    displacements = (int *) ucc_malloc(size * sizeof(uint32_t), "displacements buf");
    UCC_MALLOC_CHECK(displacements);
    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, _msgsize * size, _mt));
    rbuf       = rbuf_mc_header->addr;
    check_rbuf = ucc_malloc(_msgsize*size, "check rbuf");
    UCC_MALLOC_CHECK(check_rbuf);
    fill_counts_and_displacements(size, count, counts, displacements);

    if (TEST_NO_INPLACE == inplace) {
        args.mask = 0;
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, counts[rank] * dt_size, _mt));
        sbuf = sbuf_mc_header->addr;
        init_buffer(sbuf, counts[rank], TEST_DT, _mt, rank);
        UCC_ALLOC_COPY_BUF(check_sbuf_mc_header, UCC_MEMORY_TYPE_HOST, sbuf,
                           _mt, counts[rank] * dt_size);
        check_sbuf = check_sbuf_mc_header->addr;
    } else {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
        init_buffer((void*)((ptrdiff_t)rbuf + displacements[rank] * dt_size),
                    counts[rank], TEST_DT, _mt, rank);
        init_buffer((void*)((ptrdiff_t)check_rbuf + displacements[rank] * dt_size),
                    counts[rank], TEST_DT, UCC_MEMORY_TYPE_HOST, rank);
    }
    if (TEST_NO_INPLACE == inplace) {
        args.src.info.buffer          = sbuf;
        args.src.info.datatype        = TEST_DT;
        args.src.info.mem_type        = _mt;
        args.src.info.count           = counts[rank];
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
        ucc_free(counts);
    }
    if (displacements) {
        ucc_free(displacements);
    }
}

ucc_status_t TestAllgatherv::check()
{
    MPI_Datatype dt          = ucc_dt_to_mpi(TEST_DT);
    int          total_count = 0;
    int          size, rank, completed, count, i;
    MPI_Request  req;

    MPI_Comm_size(team.comm, &size);
    MPI_Comm_rank(team.comm, &rank);
    count  = counts[rank];
    for (i = 0 ; i < size; i++) {
        total_count += counts[i];
    }
    MPI_Iallgatherv((inplace == TEST_INPLACE) ? MPI_IN_PLACE : check_sbuf,
                    count, dt, check_rbuf, (int*)counts, (int*)displacements,
                    dt, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check_rbuf, total_count, TEST_DT, mem_type);
}
