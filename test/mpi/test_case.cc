/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"
std::shared_ptr<TestCase> TestCase::init(ucc_coll_type_t _type,
                                         ucc_test_team_t &_team,
                                         int root,
                                         size_t msgsize,
                                         ucc_test_mpi_inplace_t inplace,
                                         ucc_memory_type_t mt,
                                         ucc_datatype_t dt,
                                         ucc_reduction_op_t op)
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
        status = test();
        if (status < 0) {
            std::cerr << "error during coll test\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        ucc_context_progress(team.ctx);
        mpi_progress();
    } while (UCC_OK != status);
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
    if (0 == world_rank) {
        std::cout << str() << std::endl;
    }
    run();
    wait();
    status = check();
    if (UCC_OK != status) {
        std::cerr << "FAILURE in: " << str() << std::endl;
    }
    return status;
}

TestCase::TestCase(ucc_test_team_t &_team, ucc_memory_type_t _mem_type,
                   size_t _msgsize, ucc_test_mpi_inplace_t _inplace) :
    team(_team), mem_type(_mem_type),  msgsize(_msgsize), inplace(_inplace)
{
    sbuf      = NULL;
    rbuf      = NULL;
    check_buf = NULL;
    args.mask = 0;
}

TestCase::~TestCase()
{
    UCC_CHECK(ucc_collective_finalize(req));
    if (sbuf) {
        UCC_CHECK(ucc_mc_free(sbuf, mem_type));
    }
    if (rbuf) {
        UCC_CHECK(ucc_mc_free(rbuf, mem_type));
    }
    if (check_buf) {
        UCC_CHECK(ucc_mc_free(check_buf, mem_type));
    }
}
