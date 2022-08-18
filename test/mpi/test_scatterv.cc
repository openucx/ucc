/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

static void fill_counts_and_displacements(int size, int count,
                                          uint32_t *counts, uint32_t *displs)
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

TestScatterv::TestScatterv(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_SCATTERV, params)
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count   = msgsize / dt_size;
    int    rank, size;

    root          = params.root;
    counts        = NULL;
    displacements = NULL;
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize * size),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    counts = (uint32_t *) ucc_malloc(size * sizeof(uint32_t),
                                     "counts buf");
    UCC_MALLOC_CHECK(counts);
    displacements = (uint32_t *) ucc_malloc(size * sizeof(uint32_t),
                                            "displacements buf");
    UCC_MALLOC_CHECK(displacements);
    fill_counts_and_displacements(size, count, counts, displacements);

    if (rank == root) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, count * size * dt_size, mem_type));
        sbuf = sbuf_mc_header->addr;
        if (TEST_NO_INPLACE == inplace) {
            UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, counts[rank] * dt_size,
                                   mem_type));
            rbuf = rbuf_mc_header->addr;
        } else {
            rbuf_mc_header = NULL;
            rbuf = NULL;
        }
    } else {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, counts[rank] * dt_size, mem_type));
        rbuf = rbuf_mc_header->addr;
        sbuf_mc_header = NULL;
        sbuf = NULL;
    }

    check_buf = ucc_malloc(count * size * dt_size, "check buf");
    UCC_MALLOC_CHECK(check_buf);

    if (TEST_INPLACE == inplace) {
        args.mask = UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags = UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    args.root = root;
    if (rank == root) {
        args.src.info_v.buffer        = sbuf;
        args.src.info_v.counts        = (ucc_count_t*)counts;
        args.src.info_v.displacements = (ucc_aint_t*)displacements;
        args.src.info_v.datatype      = TEST_DT;
        args.src.info_v.mem_type      = mem_type;
        if (TEST_NO_INPLACE == inplace) {
            args.dst.info.buffer   = rbuf;
            args.dst.info.count    = counts[rank];
            args.dst.info.datatype = TEST_DT;
            args.dst.info.mem_type = mem_type;
        }
    } else {
        args.dst.info.buffer   = rbuf;
        args.dst.info.count    = counts[rank];
        args.dst.info.datatype = TEST_DT;
        args.dst.info.mem_type = mem_type;
    }

    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestScatterv::set_input()
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count   = msgsize / dt_size;
    int    rank, size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (rank == root) {
        init_buffer(sbuf, count * size, TEST_DT, mem_type, rank);
        UCC_CHECK(ucc_mc_memcpy(check_buf, sbuf, count * size * dt_size,
                                UCC_MEMORY_TYPE_HOST, mem_type));
    }
    return UCC_OK;
}

ucc_status_t TestScatterv::reset_sbuf()
{
    return UCC_OK;
}

TestScatterv::~TestScatterv()
{
    if (counts) {
        ucc_free(counts);
    }
    if (displacements) {
        ucc_free(displacements);
    }
}

ucc_status_t TestScatterv::check()
{
    size_t        dt_size = ucc_dt_size(TEST_DT);
    size_t        count   = msgsize / dt_size;
    MPI_Datatype  dt      = ucc_dt_to_mpi(TEST_DT);
    MPI_Request   req;
    int           size, rank, completed;

    MPI_Comm_size(team.comm, &size);
    MPI_Comm_rank(team.comm, &rank);

    MPI_Iscatterv(check_buf, (int*)counts, (int*)displacements, dt,
                  (rank == root) ? MPI_IN_PLACE : check_buf, counts[rank],
                  dt, root, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    if (rank == root) {
        if (TEST_INPLACE == inplace) {
            return compare_buffers(sbuf, check_buf, count * size,
                                   TEST_DT, mem_type);
        } else {
            return compare_buffers(rbuf,
                                   PTR_OFFSET(check_buf, displacements[rank] * dt_size),
                                   counts[rank], TEST_DT, mem_type);
        }
    } else {
        return compare_buffers(rbuf, check_buf, counts[rank], TEST_DT,
                               mem_type);
    }
}
