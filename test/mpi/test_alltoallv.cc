/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */
#include <random>
#include <assert.h>

#include "test_mpi.h"
#include "mpi_util.h"

template<typename T>
void * TestAlltoallv::mpi_counts_to_ucc(int *mpi_counts, size_t _ncount)
{
    void *ucc_counts = (T*)malloc(sizeof(T) * _ncount);
    for (size_t i = 0; i < _ncount; i++) {
        ((T*)ucc_counts)[i] = mpi_counts[i];
    }
    return ucc_counts;
}

TestAlltoallv::TestAlltoallv(ucc_test_team_t &_team, TestCaseParams &params) :
    TestCase(_team, UCC_COLL_TYPE_ALLTOALLV, params)
{
    std::default_random_engine eng;
    size_t                     dt_size, count;
    int                        rank, nprocs, rank_count;
    bool                       is_onesided;
    void                      *work_buf;

    dt          = params.dt;
    dt_size     = ucc_dt_size(dt);
    count       = msgsize / dt_size;
    sncounts    = 0;
    rncounts    = 0;
    scounts     = NULL;
    sdispls     = NULL;
    rcounts     = NULL;
    rdispls     = NULL;
    scounts64   = NULL;
    sdispls64   = NULL;
    rcounts64   = NULL;
    rdispls64   = NULL;
    count_bits  = params.count_bits;
    displ_bits  = params.displ_bits;
    is_onesided = (params.buffers != NULL);
    work_buf    = NULL;

    std::uniform_int_distribution<int> urd(count / 2, count);
    eng.seed(test_rand_seed);

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize * nprocs),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
    args.flags |= UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                  UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
    if (is_onesided) {
        args.mask  |= UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;
        args.flags |= UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS;
    }
    if (count_bits == TEST_FLAG_VSIZE_64BIT) {
        args.flags |= UCC_COLL_ARGS_FLAG_COUNT_64BIT;
    }
    if (displ_bits == TEST_FLAG_VSIZE_64BIT) {
        args.flags |= UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
    }

    scounts = (int*)malloc(sizeof(*scounts) * nprocs);
    sdispls = (int*)malloc(sizeof(*sdispls) * nprocs);
    rcounts = (int*)malloc(sizeof(*rcounts) * nprocs);
    rdispls = (int*)malloc(sizeof(*rdispls) * nprocs);

    for (auto i = 0; i < nprocs; i++) {
        rank_count = urd(eng);
        scounts[i] = rank_count;
    }

    MPI_Alltoall((void*)scounts, 1, MPI_INT,
                 (void*)rcounts, 1, MPI_INT, team.comm);

    sncounts = 0;
    rncounts = 0;
    for (auto i = 0; i < nprocs; i++) {
        assert((size_t)rcounts[i] <= count);
        sdispls[i] = sncounts;
        rdispls[i] = rncounts;
        sncounts += scounts[i];
        rncounts += rcounts[i];
    }
    if ((test_max_size < (sncounts * dt_size)) ||
            (test_max_size < (rncounts * dt_size))) {
        test_skip = TEST_SKIP_MEM_LIMIT;
    }
    if (TEST_SKIP_NONE != skip_reduce(test_skip, team.comm)) {
        return;
    }
    check_buf = ucc_malloc(rncounts * dt_size, "check buf");
    UCC_MALLOC_CHECK(check_buf);

    if (!is_onesided) {
        UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, sncounts * dt_size, mem_type));
        UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, rncounts * dt_size, mem_type));
        sbuf = sbuf_mc_header->addr;
        rbuf = rbuf_mc_header->addr;
    } else {
        sbuf                    = params.buffers[MEM_SEND_SEGMENT];
        rbuf                    = params.buffers[MEM_RECV_SEGMENT];
        work_buf                = params.buffers[MEM_WORK_SEGMENT];
        args.global_work_buffer = work_buf;
    }

    args.src.info_v.buffer = sbuf;
    args.src.info_v.datatype = dt;
    args.src.info_v.mem_type = mem_type;

    args.dst.info_v.buffer = rbuf;
    args.dst.info_v.datatype = dt;
    args.dst.info_v.mem_type = mem_type;

    if (TEST_FLAG_VSIZE_64BIT == count_bits ||
        TEST_FLAG_VSIZE_64BIT == displ_bits) {
        if (msgsize % 64 != 0) {
            test_skip = TEST_SKIP_NOT_SUPPORTED;
        }
    } else {
        if (msgsize % 32 != 0) {
            test_skip = TEST_SKIP_NOT_SUPPORTED;
        }
    }
    if (TEST_SKIP_NONE != skip_reduce(test_skip, team.comm)) {
        return;
    }

    if (TEST_FLAG_VSIZE_64BIT == count_bits) {
        args.src.info_v.counts = scounts64 =
                (ucc_count_t*)mpi_counts_to_ucc<uint64_t>(scounts, nprocs);
        args.dst.info_v.counts = rcounts64 =
                (ucc_count_t*)mpi_counts_to_ucc<uint64_t>(rcounts, nprocs);
    } else {
        args.src.info_v.counts = (ucc_count_t*)scounts;
        args.dst.info_v.counts = (ucc_count_t*)rcounts;
    }
    if (TEST_FLAG_VSIZE_64BIT == displ_bits) {
        args.src.info_v.displacements = sdispls64 =
                (ucc_aint_t*)mpi_counts_to_ucc<uint64_t>(sdispls, nprocs);
        args.dst.info_v.displacements = rdispls64 =
                (ucc_aint_t*)mpi_counts_to_ucc<uint64_t>(rdispls, nprocs);
    } else {
        args.src.info_v.displacements = (ucc_aint_t*)sdispls;
        args.dst.info_v.displacements = (ucc_aint_t*)rdispls;
    }
    if (is_onesided) {
        MPI_Datatype datatype;
        size_t       disp_size;
        void        *ldisp;
        int          alltoall_status;

        if (TEST_FLAG_VSIZE_64BIT == displ_bits) {
            datatype  = MPI_LONG;
            disp_size = sizeof(uint64_t);
        } else {
            datatype  = MPI_INT;
            disp_size = sizeof(uint32_t);
        }
        ldisp = ucc_calloc(nprocs, disp_size, "displacements");
        UCC_MALLOC_CHECK(ldisp);
        alltoall_status = MPI_Alltoall(args.dst.info_v.displacements, 1,
                                       datatype, ldisp, 1, datatype, team.comm);
        if (MPI_SUCCESS != alltoall_status) {
            std::cerr << "*** MPI ALLTOALL FAILED" << std::endl;
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        args.dst.info_v.displacements = (ucc_aint_t *)ldisp;
    }
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAlltoallv::set_input(int iter_persistent)
{
    int rank;

    this->iter_persistent = iter_persistent;
    MPI_Comm_rank(team.comm, &rank);
    init_buffer(sbuf, sncounts, dt, mem_type, rank * (iter_persistent + 1));

    return UCC_OK;
}

TestAlltoallv::~TestAlltoallv()
{
    free(scounts);
    free(sdispls);
    free(rcounts);
    free(rdispls);
    free(scounts64);
    free(sdispls64);
    free(rcounts64);
    free(rdispls64);
}

ucc_status_t TestAlltoallv::check()
{
    MPI_Request req;
    int         i, size, rank, completed;

    MPI_Comm_size(team.comm, &size);
    MPI_Comm_rank(team.comm, &rank);

    MPI_Ialltoall(sdispls, 1, MPI_INT, scounts, 1, MPI_INT, team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    for (i = 0; i < size; i++) {
        init_buffer(PTR_OFFSET(check_buf, rdispls[i] * ucc_dt_size(dt)),
                    rcounts[i], dt, UCC_MEMORY_TYPE_HOST,
                    i * (iter_persistent + 1), scounts[i]);
    }

    return compare_buffers(rbuf, check_buf, rncounts, dt, mem_type);
}

std::string TestAlltoallv::str()
{
    return TestCase::str() +
            " counts=" + (count_bits == TEST_FLAG_VSIZE_64BIT ? "64" : "32") +
            " displs=" + (displ_bits == TEST_FLAG_VSIZE_64BIT ? "64" : "32");
}
