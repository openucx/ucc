/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"

std::vector<std::shared_ptr<TestCase>>
TestCase::init(ucc_coll_type_t _type, ucc_test_team_t &_team, int num_tests,
               int root, size_t msgsize, ucc_test_mpi_inplace_t inplace,
               ucc_memory_type_t mt, size_t max_size, ucc_datatype_t dt,
               ucc_reduction_op_t op, ucc_test_vsize_flag_t count_bits,
               ucc_test_vsize_flag_t displ_bits, void **onesided_buffers)
{
    std::vector<std::shared_ptr<TestCase>> tcs;

    for (int i = 0; i < num_tests; i++) {
        auto tc =
            init_single(_type, _team, root, msgsize, inplace, mt, max_size, dt,
                        op, count_bits, displ_bits, onesided_buffers);
        if (!tc) {
            tcs.clear();
            return tcs;
        }
        tcs.push_back(tc);
    }
    return tcs;
}

std::shared_ptr<TestCase> TestCase::init_single(
        ucc_coll_type_t _type,
        ucc_test_team_t &_team,
        int root,
        size_t msgsize,
        ucc_test_mpi_inplace_t inplace,
        ucc_memory_type_t mt,
        size_t max_size,
        ucc_datatype_t dt,
        ucc_reduction_op_t op,
        ucc_test_vsize_flag_t count_bits,
        ucc_test_vsize_flag_t displ_bits,
        void ** onesided_buffers)
{
    switch(_type) {
    case UCC_COLL_TYPE_BARRIER:
        return std::make_shared<TestBarrier>(_team);
    case UCC_COLL_TYPE_ALLREDUCE:
        return std::make_shared<TestAllreduce>(msgsize, inplace, dt,
                                               op, mt, _team, max_size);
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return std::make_shared<TestReduceScatter>(msgsize, inplace, dt,
                                                   op, mt, _team, max_size);
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return std::make_shared<TestReduceScatterv>(msgsize, inplace, dt, op,
                                                    mt, _team, max_size);
    case UCC_COLL_TYPE_ALLGATHER:
        return std::make_shared<TestAllgather>(msgsize, inplace, mt, _team,
                                               max_size);
    case UCC_COLL_TYPE_ALLGATHERV:
        return std::make_shared<TestAllgatherv>(msgsize, inplace, mt, _team,
                                                max_size);
    case UCC_COLL_TYPE_BCAST:
        return std::make_shared<TestBcast>(msgsize, inplace, mt, root, _team,
                                           max_size);
    case UCC_COLL_TYPE_REDUCE:
        return std::make_shared<TestReduce>(msgsize, inplace, dt, op, mt, root,
                                           _team, max_size);
    case UCC_COLL_TYPE_ALLTOALL:
        if (onesided_buffers) {
            return std::make_shared<TestAlltoall>(msgsize, inplace, mt, _team,
                                                  max_size, onesided_buffers);
        } else {
            return std::make_shared<TestAlltoall>(msgsize, inplace, mt, _team,
                                                  max_size);
        }
    case UCC_COLL_TYPE_ALLTOALLV:
        return std::make_shared<TestAlltoallv>(msgsize, inplace, mt, _team,
                                               max_size, count_bits, displ_bits);
    case UCC_COLL_TYPE_GATHER:
        return std::make_shared<TestGather>(msgsize, inplace, mt, root, _team,
                                            max_size);
    case UCC_COLL_TYPE_GATHERV:
        return std::make_shared<TestGatherv>(msgsize, inplace, mt, root, _team,
                                             max_size);
    case UCC_COLL_TYPE_SCATTER:
        return std::make_shared<TestScatter>(msgsize, inplace, mt, root, _team,
                                             max_size);
    case UCC_COLL_TYPE_SCATTERV:
        return std::make_shared<TestScatterv>(msgsize, inplace, mt, root, _team,
                                              max_size);
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
            std::cerr << "error during coll test: "
                      << ucc_status_string(status)
                      << " ("<<status<<")" << std::endl;
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
    std::string _str = std::string("tc=");
    if (args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS) {
        _str += std::string("Onesided ");
    }
    _str += std::string(ucc_coll_type_str(args.coll_type)) +
            " team=" + team_str(team.type) +
            " mtype=" + ucc_memory_type_names[mem_type] +
            " msgsize=" + std::to_string(msgsize);
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

test_skip_cause_t TestCase::skip_reduce(int skip_cond, test_skip_cause_t cause,
                                        MPI_Comm comm)
{
    test_skip_cause_t test_skip;
    test_skip_cause_t skip = skip_cond ? cause : TestCase::test_skip;
    MPI_Allreduce((void*)&skip, (void*)&test_skip, 1, MPI_INT, MPI_MAX, comm);
    TestCase::test_skip = test_skip;
    return test_skip;
}

test_skip_cause_t TestCase::skip_reduce(test_skip_cause_t cause, MPI_Comm comm)
{
    return skip_reduce(1, cause, comm);
}

void TestCase::tc_progress_ctx()
{
    ucc_context_progress(team.ctx);
}

TestCase::TestCase(ucc_test_team_t &_team, ucc_coll_type_t ct,
                   ucc_memory_type_t _mem_type,
                   size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                   size_t _max_size) :
    team(_team), mem_type(_mem_type),  msgsize(_msgsize), inplace(_inplace),
    test_max_size(_max_size)
{
    int rank;

    sbuf           = NULL;
    rbuf           = NULL;
    check_buf      = NULL;
    sbuf_mc_header = NULL;
    rbuf_mc_header = NULL;
    test_skip      = TEST_SKIP_NONE;
    args.flags     = 0;
    args.mask      = 0;
    args.coll_type = ct;

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
    if (sbuf_mc_header) {
        UCC_CHECK(ucc_mc_free(sbuf_mc_header));
    }
    if (rbuf_mc_header) {
        UCC_CHECK(ucc_mc_free(rbuf_mc_header));
    }
    if (check_buf) {
        ucc_free(check_buf);
    }
}
