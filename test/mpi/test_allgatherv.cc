/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"

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

TestAllgatherv::TestAllgatherv(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_ALLGATHERV, params)
{
    int    rank, size;
    size_t dt_size, count;

    dt            = params.dt;
    dt_size       = ucc_dt_size(dt);
    count         = msgsize / dt_size;
    counts        = NULL;
    displacements = NULL;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &size);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize * size),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    counts = (int *) ucc_malloc(size * sizeof(uint32_t), "counts buf");
    UCC_MALLOC_CHECK(counts);
    displacements = (int *) ucc_malloc(size * sizeof(uint32_t), "displacements buf");
    UCC_MALLOC_CHECK(displacements);
    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, msgsize * size, mem_type));
    rbuf      = rbuf_mc_header->addr;
    check_buf = ucc_malloc(msgsize * size, "check buf");
    UCC_MALLOC_CHECK(check_buf);
    fill_counts_and_displacements(size, count, counts, displacements);

    if (!inplace) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, counts[rank] * dt_size, mem_type));
        sbuf                   = sbuf_mc_header->addr;
        args.src.info.buffer   = sbuf;
        args.src.info.datatype = dt;
        args.src.info.mem_type = mem_type;
        args.src.info.count    = counts[rank];
    }
    args.dst.info_v.buffer        = rbuf;
    args.dst.info_v.counts        = (ucc_count_t*)counts;
    args.dst.info_v.displacements = (ucc_aint_t*)displacements;
    args.dst.info_v.datatype      = dt;
    args.dst.info_v.mem_type      = mem_type;
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAllgatherv::set_input(int iter_persistent)
{
    size_t dt_size = ucc_dt_size(dt);
    int    rank;
    void  *buf, *check;

    MPI_Comm_rank(team.comm, &rank);
    if (inplace) {
        buf = PTR_OFFSET(rbuf, displacements[rank] * dt_size);
    } else {
        buf = sbuf;
    }
    check = PTR_OFFSET(check_buf, displacements[rank] * dt_size);
    init_buffer(buf, counts[rank], dt, mem_type, rank * (iter_persistent + 1));
    UCC_CHECK(ucc_mc_memcpy(check, buf, counts[rank] * dt_size,
                            UCC_MEMORY_TYPE_HOST, mem_type));
    return UCC_OK;
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
    MPI_Datatype mpi_dt      = ucc_dt_to_mpi(dt);
    int          total_count = 0;
    int          size, rank, completed, i;
    MPI_Request  req;

    MPI_Comm_size(team.comm, &size);
    MPI_Comm_rank(team.comm, &rank);
    for (i = 0 ; i < size; i++) {
        total_count += counts[i];
    }
    MPI_Iallgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, check_buf,
                    (int *)counts, (int *)displacements, mpi_dt, team.comm,
                    &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check_buf, total_count, dt, mem_type);
}
