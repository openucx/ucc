/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"

std::shared_ptr<TestCase> TestCase::init(ucc_coll_type_t _type,
                                         ucc_test_team_t &_team,
                                         size_t msgsize,
                                         ucc_memory_type_t mt,
                                         ucc_datatype_t dt,
                                         ucc_reduction_op_t op)
{
    switch(_type) {
    case UCC_COLL_TYPE_BARRIER:
        return std::make_shared<TestBarrier>(_team);
    case UCC_COLL_TYPE_ALLREDUCE:
        return std::make_shared<TestAllreduce>(msgsize, dt, op, mt, _team);
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
    } while (UCC_OK != status);
}

std::string TestCase::str() {
    return std::string("tc=")+std::string(ucc_coll_type_str(args.coll_type)) +
        std::string(" team=") + std::string(team_str(team.type)) + " msgsize="
        + std::to_string(msgsize);
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
                   size_t _msgsize) :
    team(_team), mem_type(_mem_type),  msgsize(_msgsize)
{
    sbuf      = NULL;
    rbuf      = NULL;
    check_buf = NULL;
}

TestCase::~TestCase()
{
    UCC_CHECK(ucc_collective_finalize(req));
    UCC_CHECK(ucc_mc_free(sbuf, mem_type));
    UCC_CHECK(ucc_mc_free(rbuf, mem_type));
    UCC_CHECK(ucc_mc_free(check_buf, mem_type));
}
