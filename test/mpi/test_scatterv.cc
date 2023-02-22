/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

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
    int    rank, size;
    size_t dt_size, count;

    dt            = params.dt;
    dt_size       = ucc_dt_size(dt);
    count         = msgsize / dt_size;
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
        if (inplace) {
            rbuf_mc_header = NULL;
            rbuf = NULL;
        } else {
            UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, counts[rank] * dt_size,
                                   mem_type));
            rbuf = rbuf_mc_header->addr;
        }
    } else {
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, counts[rank] * dt_size, mem_type));
        rbuf = rbuf_mc_header->addr;
        sbuf_mc_header = NULL;
        sbuf = NULL;
    }

    check_buf = ucc_malloc(count * size * dt_size, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    args.root = root;
    if (rank == root) {
        args.src.info_v.buffer        = sbuf;
        args.src.info_v.counts        = (ucc_count_t*)counts;
        args.src.info_v.displacements = (ucc_aint_t*)displacements;
        args.src.info_v.datatype      = dt;
        args.src.info_v.mem_type      = mem_type;
        if (!inplace) {
            args.dst.info.buffer   = rbuf;
            args.dst.info.count    = counts[rank];
            args.dst.info.datatype = dt;
            args.dst.info.mem_type = mem_type;
        }
    } else {
        args.dst.info.buffer   = rbuf;
        args.dst.info.count    = counts[rank];
        args.dst.info.datatype = dt;
        args.dst.info.mem_type = mem_type;
    }

    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestScatterv::set_input(int iter_persistent)
{
    size_t dt_size = ucc_dt_size(dt);
    size_t count   = msgsize / dt_size;
    int    rank, size;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (rank == root) {
        init_buffer(sbuf, count * size, dt, mem_type,
                    rank * (iter_persistent + 1));
        UCC_CHECK(ucc_mc_memcpy(check_buf, sbuf, count * size * dt_size,
                                UCC_MEMORY_TYPE_HOST, mem_type));
    }
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
    size_t        dt_size = ucc_dt_size(dt);
    size_t        count   = msgsize / dt_size;
    MPI_Datatype  mpi_dt  = ucc_dt_to_mpi(dt);
    MPI_Request   req;
    int           size, rank, completed;

    MPI_Comm_size(team.comm, &size);
    MPI_Comm_rank(team.comm, &rank);

    MPI_Iscatterv(check_buf, (int *)counts, (int *)displacements, mpi_dt,
                  (rank == root) ? MPI_IN_PLACE : check_buf, counts[rank],
                  mpi_dt, root, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    if (rank == root) {
        if (inplace) {
            return compare_buffers(sbuf, check_buf, count * size, dt, mem_type);
        } else {
            return compare_buffers(
                rbuf, PTR_OFFSET(check_buf, displacements[rank] * dt_size),
                counts[rank], dt, mem_type);
        }
    } else {
        return compare_buffers(rbuf, check_buf, counts[rank], dt, mem_type);
    }
}
