/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "test_mpi.h"
#include "mpi_util.h"
BEGIN_C_DECLS
#include "utils/ucc_math.h"
END_C_DECLS
#include <algorithm>
#include <assert.h>
UccTestMpi::UccTestMpi(int argc, char *argv[], ucc_thread_mode_t tm,
                       std::vector<ucc_test_mpi_team_t> &test_teams,
                       const char *cls)
{
    int required = (tm == UCC_THREAD_SINGLE) ? MPI_THREAD_SINGLE
        : MPI_THREAD_MULTIPLE;
    int rank, size, provided;
    ucc_lib_config_h lib_config;
    ucc_context_config_h ctx_config;


    MPI_Init_thread(&argc, &argv, required, &provided);
    if (provided != required) {
        std::cerr << "could not initialize MPI in thread multiple\n";
        abort();
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) {
        std::cerr << "test requires at least 2 ranks\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    /* Init ucc library */
    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = tm,
        /* .coll_types = coll_types, */
    };

    /* Init ucc context for a specified UCC_TEST_TLS */
    ucc_context_params_t ctx_params = {
        .mask     = UCC_CONTEXT_PARAM_FIELD_TYPE,
        .ctx_type = UCC_CONTEXT_EXCLUSIVE,
    };
    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
    if (cls) {
        UCC_CHECK(ucc_lib_config_modify(lib_config, "CLS", cls));
    }
    UCC_CHECK(ucc_init(&lib_params, lib_config, &lib));
    ucc_lib_config_release(lib_config);

    UCC_CHECK(ucc_context_config_read(lib, NULL, &ctx_config));
    UCC_CHECK(ucc_context_create(lib, &ctx_params, ctx_config, &ctx));
    ucc_context_config_release(ctx_config);
    for (auto &t : test_teams) {
        if (size < 4 && (t == TEAM_SPLIT_HALF || t == TEAM_SPLIT_ODD_EVEN)) {
            if (rank == 0) {
                std::cout << "size of the world=" << size <<
                    " is too small to create team " << team_str(t) <<
                    ", skipping ...\n";
            }
            continue;
        }
        create_team(t);
    }
    set_msgsizes(8, ((1ULL) << 21), 8);
    dtypes = {UCC_DT_INT32, UCC_DT_INT64, UCC_DT_FLOAT32, UCC_DT_FLOAT64};
    ops = {UCC_OP_SUM, UCC_OP_MAX};
    colls = {UCC_COLL_TYPE_BARRIER, UCC_COLL_TYPE_ALLREDUCE};
    mtypes = {UCC_MEMORY_TYPE_HOST};
    inplace = TEST_NO_INPLACE;
    root_type = ROOT_RANDOM;
    root_value = 10;
}

UccTestMpi::~UccTestMpi()
{
    for (auto &t : teams) {
        destroy_team(t);
    }
    UCC_CHECK(ucc_context_destroy(ctx));
    UCC_CHECK(ucc_finalize(lib));
    MPI_Finalize();
}

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)request;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
    MPI_Request request = (MPI_Request)req;
    int         completed;
    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}

ucc_team_h UccTestMpi::create_ucc_team(MPI_Comm comm)
{
    int rank, size;
    ucc_team_h team;
    ucc_team_params_t team_params;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Create UCC TEAM for comm world */
    team_params.mask               = UCC_TEAM_PARAM_FIELD_EP       |
        UCC_TEAM_PARAM_FIELD_EP_RANGE |
        UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob.allgather      = oob_allgather;
    team_params.oob.req_test       = oob_allgather_test;
    team_params.oob.req_free       = oob_allgather_free;
    team_params.oob.coll_info      = (void*)comm;
    team_params.oob.participants   = size;
    team_params.ep                 = rank;
    team_params.ep_range           = UCC_COLLECTIVE_EP_RANGE_CONTIG;

    UCC_CHECK(ucc_team_create_post(&ctx, 1, &team_params, &team));
    while (UCC_INPROGRESS == ucc_team_create_test(team)) { ; };
    return team;
}


void UccTestMpi::create_team(ucc_test_mpi_team_t t)
{
    MPI_Comm comm = create_mpi_comm(t);
    ucc_team_h team = create_ucc_team(comm);
    teams.push_back(ucc_test_team_t(t, comm, team, ctx));
}

void UccTestMpi::destroy_team(ucc_test_team_t &team)
{
    UCC_CHECK(ucc_team_destroy(team.team));
    if (team.comm != MPI_COMM_WORLD) {
        MPI_Comm_free(&team.comm);
    }
}

void UccTestMpi::set_msgsizes(size_t min, size_t max, size_t power)
{
    size_t m = min;
    msgsizes.clear();
    while (m < max) {
        msgsizes.push_back(m);
        m *= power;
    }
    msgsizes.push_back(max);
}

void UccTestMpi::set_dtypes(std::vector<ucc_datatype_t> &_dtypes)
{
    dtypes = _dtypes;
}

void UccTestMpi::set_mtypes(std::vector<ucc_memory_type_t> &_mtypes)
{
    mtypes = _mtypes;
}

void UccTestMpi::set_colls(std::vector<ucc_coll_type_t> &_colls)
{
    colls = _colls;
}

void UccTestMpi::set_ops(std::vector<ucc_reduction_op_t> &_ops)
{
    ops = _ops;
}

int ucc_coll_inplace_supported(ucc_coll_type_t c)
{
    switch(c) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_BCAST:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return 0;
    default:
        return 1;
    }
}

int ucc_coll_is_rooted(ucc_coll_type_t c)
{
    switch(c) {
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_ALLGATHER:
    case UCC_COLL_TYPE_ALLGATHERV:
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_ALLTOALLV:
    case UCC_COLL_TYPE_BARRIER:
        return 0;
    default:
        return 1;
    }
}

void UccTestMpi::set_count_vsizes(std::vector<ucc_test_vsize_flag_t> &_counts_vsize)
{
    counts_vsize =  _counts_vsize;
}

void UccTestMpi::set_displ_vsizes(std::vector<ucc_test_vsize_flag_t> &_displs_vsize)
{
    displs_vsize = _displs_vsize;
}

void UccTestMpi::run_all()
{
    for (auto &c : colls) {
        for (auto &t : teams) {
            std::vector<int> roots = {0};
            if (ucc_coll_is_rooted(c)) {
                roots = gen_roots(t);
            }
            for (auto r : roots) {
                if (c == UCC_COLL_TYPE_BARRIER) {
                    auto tc = TestCase::init(c, t);
                    results.push_back(tc.get()->exec());
                } else {
                    for (auto mt : mtypes) {
                        for (auto m : msgsizes) {
                            if (c == UCC_COLL_TYPE_ALLREDUCE ||
                                c == UCC_COLL_TYPE_REDUCE) {
                                for (auto dt : dtypes) {
                                    for (auto op : ops) {
                                        auto tc = TestCase::init(c, t, r, m,
                                                                 inplace, mt, dt, op);
                                        results.push_back(tc.get()->exec());
                                    }
                                }
                            } else if (c == UCC_COLL_TYPE_ALLTOALL ||
                                       c == UCC_COLL_TYPE_ALLTOALLV) {
                                if (TEST_INPLACE == inplace &&
                                    ucc_coll_inplace_supported(c)) {
                                    results.push_back(UCC_ERR_NOT_IMPLEMENTED);
                                    continue;
                                }
                                for (auto dt : dtypes) {
                                    switch (c) {
                                    case UCC_COLL_TYPE_ALLTOALL:
                                    {
                                        auto tc = TestCase::init(c, t, r, m, inplace, mt, dt);
                                        results.push_back(tc.get()->exec());
                                        break;
                                    }
                                    case UCC_COLL_TYPE_ALLTOALLV:
                                    {
                                        for (auto count_bits : counts_vsize) {
                                           for (auto displ_bits : displs_vsize) {
                                                auto tc = TestCase::init(c, t, r, m, inplace,
                                                                 mt, dt, (ucc_reduction_op_t)-1,
                                                                 count_bits, displ_bits);
                                                results.push_back(tc.get()->exec());
                                            }
                                        }
                                        break;
                                    }
                                    default:
                                        continue;
                                    }
                                }
                            } else {
                                auto tc = TestCase::init(c, t, r, m, inplace, mt);
                                if (TEST_INPLACE == inplace &&
                                    ucc_coll_inplace_supported(c)) {
                                    results.push_back(UCC_ERR_NOT_IMPLEMENTED);
                                    continue;
                                }
                                results.push_back(tc.get()->exec());
                            }
                        }
                    }
                }
            }
        }
    }
}

std::vector<int> UccTestMpi::gen_roots(ucc_test_team_t &team)
{
    int size;
    std::vector<int> _roots;
    MPI_Comm_size(team.comm, &size);
    std::default_random_engine eng;
    eng.seed(123);
    std::uniform_int_distribution<int> urd(0, size-1);

    switch(root_type) {
    case ROOT_SINGLE:
        _roots = std::vector<int>({ucc_min(root_value, size-1)});
        break;
    case ROOT_RANDOM:
        _roots.resize(root_value);
        for (unsigned i = 0; i < _roots.size(); i++) {
            _roots[i] = urd(eng);
        }
        break;
    case ROOT_ALL:
        _roots.resize(size);
        std::iota(_roots.begin(), _roots.end(), 0);
        break;
    default:
        assert(0);
    }
    return _roots;
}
