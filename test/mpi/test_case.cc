/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"

std::shared_ptr<TestCase> TestCase::init(ucc_coll_type_t _type,
                                         ucc_test_team_t &_team,
                                         int root,
                                         size_t msgsize,
                                         ucc_test_mpi_inplace_t inplace,
                                         ucc_memory_type_t mt,
                                         ucc_datatype_t dt,
                                         ucc_reduction_op_t op,
                                         ucc_test_vsize_flag_t count_bits,
                                         ucc_test_vsize_flag_t displ_bits)
{
    switch(_type) {
    case UCC_COLL_TYPE_BARRIER:
        return std::make_shared<TestBarrier>(_team);
    case UCC_COLL_TYPE_ALLREDUCE:
        return std::make_shared<TestAllreduce>(msgsize, inplace, dt,
                                               op, mt, _team);
    case UCC_COLL_TYPE_ALLGATHER:
        return std::make_shared<TestAllgather>(msgsize, inplace, mt, _team);
    case UCC_COLL_TYPE_ALLGATHERV:
        return std::make_shared<TestAllgatherv>(msgsize, inplace, mt, _team);
    case UCC_COLL_TYPE_BCAST:
        return std::make_shared<TestBcast>(msgsize, mt, root, _team);
    case UCC_COLL_TYPE_ALLTOALL:
        return std::make_shared<TestAlltoall>(msgsize, inplace, dt, mt, _team);
    case UCC_COLL_TYPE_ALLTOALLV:
        return std::make_shared<TestAlltoallv>(msgsize, inplace, dt, mt, _team,
                                               count_bits, displ_bits);
    default:
        break;
    }
    return NULL;
}

void TestCase::run()
{
    UCC_CHECK(ucc_collective_post(req));
}

ucc_status_t TestCase::test()
{
    return ucc_collective_test(req);
}

void TestCase::wait()
{
    ucc_status_t status;
    do {
        mpi_progress();
        status = test();
        if (status < 0) {
            std::cerr << "error during coll test\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        ucc_context_progress(team.ctx);
    } while (UCC_OK != status);
}

void TestCase::mpi_progress(void)
{
    MPI_Status status;
    int flag = 0;
    MPI_Test(&progress_request, &flag, &status);
}

std::string TestCase::str() {
    std::string _str = std::string("tc=") + ucc_coll_type_str(args.coll_type) +
        " team=" + team_str(team.type) + " msgsize=" + std::to_string(msgsize);
    if (ucc_coll_inplace_supported(args.coll_type)) {
        _str += std::string(" inplace=") + (inplace == TEST_INPLACE ? "1" : "0");
    }
    if (ucc_coll_is_rooted(args.coll_type)) {
        _str += std::string(" root=") + std::to_string(root);
    }
    return _str;
}

ucc_status_t TestCase::exec()
{
    ucc_status_t status;
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    if (TEST_SKIP_NONE == test_skip) {
        if (0 == world_rank) {
            std::cout << str() << std::endl;
        }
        run();
        wait();
        status = check();
        if (UCC_OK != status) {
            std::cerr << "FAILURE in: " << str() << std::endl;
        }
    } else {
        if (0 == world_rank) {
            std::cout << "SKIPPED: " << skip_str(test_skip) << ": "
                      << str() << " " << std::endl;
        }
        status = UCC_ERR_LAST;
    }
    return status;
}

test_skip_cause_t TestCase::skip(int skip_cond, test_skip_cause_t cause,
                                 MPI_Comm comm)
{
    int rank;
    test_skip_cause_t skip = skip_cond ? cause : TEST_SKIP_NONE;
    MPI_Comm_rank(comm, &rank);
    MPI_Reduce((void*)&skip, (void*)&test_skip, 1,
               MPI_INT, MPI_MAX, 0, comm);
    MPI_Bcast((void*)&test_skip, 1, MPI_INT, 0, comm);
    return test_skip;
}


TestCase::TestCase(ucc_test_team_t &_team, ucc_memory_type_t _mem_type,
                   size_t _msgsize, ucc_test_mpi_inplace_t _inplace) :
    team(_team), mem_type(_mem_type),  msgsize(_msgsize), inplace(_inplace)
{
    int rank;
    sbuf      = NULL;
    rbuf      = NULL;
    check_sbuf = NULL;
    check_rbuf = NULL;
    args.mask = 0;
    args.flags = 0;
    test_skip = TEST_SKIP_NONE;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Irecv((void*)progress_buf, 1, MPI_CHAR, rank, 0, MPI_COMM_WORLD,
              &progress_request);
}

TestCase::~TestCase()
{
    MPI_Status status;
    MPI_Cancel(&progress_request);
    MPI_Wait(&progress_request, &status);

    if (TEST_SKIP_NONE == test_skip) {
        UCC_CHECK(ucc_collective_finalize(req));
    }
    if (sbuf) {
        UCC_CHECK(ucc_mc_free(sbuf, mem_type));
    }
    if (rbuf) {
        UCC_CHECK(ucc_mc_free(rbuf, mem_type));
    }
    if (check_sbuf) {
        UCC_CHECK(ucc_mc_free(check_sbuf, UCC_MEMORY_TYPE_HOST));
    }
    if (check_rbuf) {
        UCC_CHECK(ucc_mc_free(check_rbuf, UCC_MEMORY_TYPE_HOST));
    }
}
