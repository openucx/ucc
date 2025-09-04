/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"

std::vector<std::shared_ptr<TestCase>>
TestCase::init(ucc_test_team_t &_team, ucc_coll_type_t _type, int num_tests,
               TestCaseParams params)
{
    std::vector<std::shared_ptr<TestCase>> tcs;

    for (int i = 0; i < num_tests; i++) {
        auto tc =
            init_single(_team, _type, params);
        if (!tc) {
            tcs.clear();
            return tcs;
        }
        tcs.push_back(tc);
    }
    return tcs;
}

std::shared_ptr<TestCase> TestCase::init_single(ucc_test_team_t &_team,
                                                ucc_coll_type_t _type,
                                                TestCaseParams params)
{
    switch(_type) {
    case UCC_COLL_TYPE_ALLGATHER:
        return std::make_shared<TestAllgather>(_team, params);
    case UCC_COLL_TYPE_ALLGATHERV:
        return std::make_shared<TestAllgatherv>(_team, params);
    case UCC_COLL_TYPE_ALLREDUCE:
        return std::make_shared<TestAllreduce>(_team, params);
    case UCC_COLL_TYPE_ALLTOALL:
        return std::make_shared<TestAlltoall>(_team, params);
    case UCC_COLL_TYPE_ALLTOALLV:
        return std::make_shared<TestAlltoallv>(_team, params);
    case UCC_COLL_TYPE_BARRIER:
        return std::make_shared<TestBarrier>(_team, params);
    case UCC_COLL_TYPE_BCAST:
        return std::make_shared<TestBcast>(_team, params);
    case UCC_COLL_TYPE_GATHER:
        return std::make_shared<TestGather>(_team, params);
    case UCC_COLL_TYPE_GATHERV:
        return std::make_shared<TestGatherv>(_team, params);
    case UCC_COLL_TYPE_REDUCE:
        return std::make_shared<TestReduce>(_team, params);
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return std::make_shared<TestReduceScatter>(_team, params);
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return std::make_shared<TestReduceScatterv>(_team, params);
    case UCC_COLL_TYPE_SCATTER:
        return std::make_shared<TestScatter>(_team, params);
    case UCC_COLL_TYPE_SCATTERV:
        return std::make_shared<TestScatterv>(_team, params);
    default:
        std::cerr << "collective type is not supported" << std::endl;
        break;
    }
    return NULL;
}

void TestCase::run(bool triggered)
{
    if (triggered) {
        ucc_ee_h ee = nullptr;
        ucc_ev_t comp_ev, *post_ev;
        ucc_ee_type_t ee_type;

        if (mem_type == UCC_MEMORY_TYPE_CUDA) {
            ee_type = UCC_EE_CUDA_STREAM;
        } else {
            UCC_CHECK(UCC_ERR_NOT_SUPPORTED);
        }

        UCC_CHECK(team.get_ee(ee_type, &ee));
        comp_ev.ev_type         = UCC_EVENT_COMPUTE_COMPLETE;
        comp_ev.ev_context      = nullptr;
        comp_ev.ev_context_size = 0;
        comp_ev.req             = req;


        UCC_CHECK(ucc_collective_triggered_post(ee, &comp_ev));
        UCC_CHECK(ucc_ee_get_event(ee, &post_ev));
        UCC_CHECK(ucc_ee_ack_event(ee, post_ev));
    } else {
        UCC_CHECK(ucc_collective_post(req));
    }
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
            " msgsize=" + std::to_string(msgsize) +
            " persistent=" + (persistent ? "1" : "0") +
            " local_registration=" + (local_registration ? "1" : "0");
    if (ucc_coll_inplace_supported(args.coll_type)) {
        _str += std::string(" inplace=") + (inplace ? "1" : "0");
    }
    if (ucc_coll_is_rooted(args.coll_type)) {
        _str += std::string(" root=") + std::to_string(root);
    }
    if (ucc_coll_has_datatype(args.coll_type)) {
        _str += std::string(" dt=") + ucc_datatype_str(dt);
    }

    return _str;
}

test_skip_cause_t TestCase::skip_reduce(int skip_cond, test_skip_cause_t cause,
                                        MPI_Comm comm)
{
    test_skip_cause_t test_skip;
    test_skip_cause_t skip = skip_cond ? cause : TestCase::test_skip;
    MPI_Request req;
    int completed;

    MPI_Iallreduce((void*)&skip, (void*)&test_skip, 1, MPI_INT, MPI_MAX, comm, &req);
    do {
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
        tc_progress_ctx();
    } while(!completed);
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
                   TestCaseParams params) :
    team(_team), mem_type(params.mt), msgsize(params.msgsize),
    inplace(params.inplace), persistent(params.persistent),
    local_registration(params.local_registration),
    test_max_size(params.max_size), dt(params.dt)
{
    int rank;

    sbuf           = NULL;
    rbuf           = NULL;
    check_buf      = NULL;
    sbuf_mc_header = NULL;
    rbuf_mc_header = NULL;
    src_memh       = NULL;
    dst_memh       = NULL;
    src_memh_size  = 0;
    dst_memh_size  = 0;
    test_skip      = TEST_SKIP_NONE;
    args.flags     = 0;
    args.mask      = 0;
    args.coll_type = ct;

    if (inplace) {
        args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if (persistent) {
        args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
    }

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

    if (src_memh) {
        UCC_CHECK(ucc_mem_unmap(&src_memh));
    }
    if (dst_memh) {
        UCC_CHECK(ucc_mem_unmap(&dst_memh));
    }

}
