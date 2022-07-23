/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include <random>
#include <assert.h>

#include "test_mpi.h"
#include "mpi_util.h"

#define TEST_DT UCC_DT_UINT32

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
    size_t dt_size = ucc_dt_size(TEST_DT);
    size_t count   = msgsize/dt_size;
    std::uniform_int_distribution<int> urd(count/2, count);
    std::default_random_engine         eng;
    int rank;
    int nprocs;
    int rank_count;

    eng.seed(test_rand_seed);
    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    sncounts   = 0;
    rncounts   = 0;
    scounts    = NULL;
    sdispls    = NULL;
    rcounts    = NULL;
    rdispls    = NULL;
    scounts64  = NULL;
    sdispls64  = NULL;
    rcounts64  = NULL;
    rdispls64  = NULL;
    count_bits = params.count_bits;
    displ_bits = params.displ_bits;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    if (TEST_SKIP_NONE != skip_reduce(test_max_size < (msgsize * nprocs),
                                      TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    args.mask  = UCC_COLL_ARGS_FIELD_FLAGS;
    args.flags |= UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                  UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
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

    UCC_CHECK(ucc_mc_alloc(&sbuf_mc_header, sncounts * dt_size, mem_type));
    UCC_CHECK(ucc_mc_alloc(&rbuf_mc_header, rncounts * dt_size, mem_type));
    sbuf      = sbuf_mc_header->addr;
    rbuf      = rbuf_mc_header->addr;
    check_buf = ucc_malloc((sncounts + rncounts) * dt_size, "check buf");
    UCC_MALLOC_CHECK(check_buf);

    args.src.info_v.buffer = sbuf;
    args.src.info_v.datatype = TEST_DT;
    args.src.info_v.mem_type = mem_type;

    args.dst.info_v.buffer = rbuf;
    args.dst.info_v.datatype = TEST_DT;
    args.dst.info_v.mem_type = mem_type;

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
    UCC_CHECK(set_input());
    UCC_CHECK_SKIP(ucc_collective_init(&args, &req, team.team), test_skip);
}

ucc_status_t TestAlltoallv::set_input()
{
    size_t dt_size = ucc_dt_size(TEST_DT);
    int    rank;

    MPI_Comm_rank(team.comm, &rank);
    init_buffer(sbuf, sncounts, TEST_DT, mem_type, rank);
    UCC_CHECK(ucc_mc_memcpy(check_buf, sbuf, sncounts * dt_size,
                            UCC_MEMORY_TYPE_HOST, mem_type));

    return UCC_OK;
}

ucc_status_t TestAlltoallv::reset_sbuf()
{
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
    size_t      dt_size = ucc_dt_size(TEST_DT);
    MPI_Request req;
    int         completed;
    void       *check;

    check = PTR_OFFSET(check_buf, sncounts * dt_size);
    MPI_Ialltoallv(check_buf, scounts, sdispls, ucc_dt_to_mpi(TEST_DT),
                   check, rcounts, rdispls, ucc_dt_to_mpi(TEST_DT),
                   team.comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        ucc_context_progress(team.ctx);
    } while(!completed);

    return compare_buffers(rbuf, check, rncounts, TEST_DT, mem_type);
}

std::string TestAlltoallv::str()
{
    return TestCase::str() +
            " counts=" + (count_bits == TEST_FLAG_VSIZE_64BIT ? "64" : "32") +
            " displs=" + (displ_bits == TEST_FLAG_VSIZE_64BIT ? "64" : "32");
}
