/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#include <random>
#include <assert.h>

#include "test_mpi.h"
#include "mpi_util.h"

TestAlltoallv::TestAlltoallv(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                             ucc_datatype_t _dt, ucc_memory_type_t _mt,
                             ucc_test_team_t &_team,
                             ucc_test_vsize_flag_t _count_bits,
                             ucc_test_vsize_flag_t _displ_bits) :
    TestCase(_team, _mt, _msgsize, _inplace)
{
    size_t dt_size = ucc_dt_size(_dt);
    size_t count = _msgsize/dt_size;
    int rank;
    int nprocs;
    int rank_count;

    std::default_random_engine eng;
    eng.seed(test_rand_seed);
    std::uniform_int_distribution<int> urd(count/2, count);

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    dt = _dt;
    sncounts = 0;
    rncounts = 0;
    scounts = NULL;
    sdispls = NULL;
    rcounts = NULL;
    rdispls = NULL;
    count_bits = _count_bits;
    displ_bits = _displ_bits;

    MPI_Comm_rank(team.comm, &rank);
    MPI_Comm_size(team.comm, &nprocs);

    args.coll_type = UCC_COLL_TYPE_ALLTOALLV;

    if (count_bits == TEST_FLAG_VSIZE_64BIT) {
        args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags |= UCC_COLL_ARGS_FLAG_COUNT_64BIT;
    }
    if (displ_bits == TEST_FLAG_VSIZE_64BIT) {
        args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
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

    if (skip((test_max_size &&
              (test_max_size < (sncounts * dt_size)
               || test_max_size < (rncounts * dt_size))),
        TEST_SKIP_MEM_LIMIT, team.comm)) {
        return;
    }

    UCC_CHECK(ucc_mc_alloc(&sbuf, sncounts * dt_size, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_sbuf, sncounts * dt_size,
                           UCC_MEMORY_TYPE_HOST));
    UCC_CHECK(ucc_mc_alloc(&rbuf, rncounts * dt_size, _mt));
    UCC_CHECK(ucc_mc_alloc(&check_rbuf, rncounts * dt_size,
                           UCC_MEMORY_TYPE_HOST));
    init_buffer(sbuf, sncounts, dt, _mt, rank);
    init_buffer(check_sbuf, sncounts, dt, UCC_MEMORY_TYPE_HOST, rank);

    args.src.info_v.buffer = sbuf;
    args.src.info_v.datatype = _dt;
    args.src.info_v.mem_type = _mt;

    args.dst.info_v.buffer = rbuf;
    args.dst.info_v.datatype = _dt;
    args.dst.info_v.mem_type = _mt;

    if (TEST_FLAG_VSIZE_64BIT == count_bits) {
        args.src.info_v.counts =
                (ucc_count_t*)_mpi_counts_to_ucc<uint64_t>(scounts, nprocs);
        args.dst.info_v.counts =
                (ucc_count_t*)_mpi_counts_to_ucc<uint64_t>(rcounts, nprocs);
    } else {
        args.src.info_v.counts =
                (ucc_count_t*)_mpi_counts_to_ucc<uint32_t>(scounts, nprocs);
        args.dst.info_v.counts =
                (ucc_count_t*)_mpi_counts_to_ucc<uint32_t>(rcounts, nprocs);
    }
    if (TEST_FLAG_VSIZE_64BIT & displ_bits) {
        args.src.info_v.displacements =
                (ucc_aint_t*)_mpi_counts_to_ucc<uint64_t>(sdispls, nprocs);
        args.dst.info_v.displacements =
                (ucc_aint_t*)_mpi_counts_to_ucc<uint64_t>(rdispls, nprocs);
    } else {
        args.src.info_v.displacements =
                (ucc_aint_t*)_mpi_counts_to_ucc<uint32_t>(sdispls, nprocs);
        args.dst.info_v.displacements =
                (ucc_aint_t*)_mpi_counts_to_ucc<uint32_t>(rdispls, nprocs);
    }

    UCC_CHECK(ucc_collective_init(&args, &req, team.team));
}

ucc_status_t TestAlltoallv::check()
{
    MPI_Alltoallv(check_sbuf, scounts, sdispls, ucc_dt_to_mpi(dt), check_rbuf,
                  rcounts, rdispls, ucc_dt_to_mpi(dt), team.comm);
    return compare_buffers(rbuf, check_rbuf, rncounts, dt, mem_type);
}

std::string TestAlltoallv::str()
{
    int rank;
    MPI_Comm_rank(team.comm, &rank);

    std::string _str = std::string("tc=") + ucc_coll_type_str(UCC_COLL_TYPE_ALLTOALLV) +
        " team=" + team_str(team.type) + " msgsize=" + std::to_string(msgsize);
    if (ucc_coll_inplace_supported(UCC_COLL_TYPE_ALLTOALLV)) {
        _str += std::string(" inplace=") + (inplace == TEST_INPLACE ? "1" : "0");
    }
    _str += std::string(" dt=") + ucc_datatype_str(dt) +
            " counts=" + (count_bits == TEST_FLAG_VSIZE_64BIT ? "64" : "32") +
            " displs=" + (displ_bits == TEST_FLAG_VSIZE_64BIT ? "64" : "32");
    return _str;
}
