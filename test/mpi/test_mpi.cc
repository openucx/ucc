/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <random>
#include <pthread.h>

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm,
                   &request);
    *req = (void *)(uintptr_t)request;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
    MPI_Request request = (MPI_Request)(uintptr_t)req;
    int         completed;
    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req)
{
    return UCC_OK;
}

UccTestMpi::UccTestMpi(int argc, char *argv[], ucc_thread_mode_t _tm,
                       int is_local, bool with_onesided)
{
    ucc_lib_config_h     lib_config;
    ucc_context_config_h ctx_config;
    int                  size, rank;
    char                *prev_env;
    ucc_mem_map_t        segments[UCC_TEST_N_MEM_SEGMENTS];

    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Init ucc library */
    ucc_lib_params_t lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = _tm,
        /* .coll_types = coll_types, */
    };
    tm = _tm; //TODO check ucc provided
    /* Init ucc context for a specified UCC_TEST_TLS */
    ucc_context_params_t ctx_params          = {};
    ucc_context_params_t onesided_ctx_params = {};
    if (!is_local) {
        ctx_params.mask         |= UCC_CONTEXT_PARAM_FIELD_OOB;
        ctx_params.oob.allgather = oob_allgather;
        ctx_params.oob.req_test  = oob_allgather_test;
        ctx_params.oob.req_free  = oob_allgather_free;
        ctx_params.oob.coll_info = (void*)(uintptr_t)MPI_COMM_WORLD;
        ctx_params.oob.n_oob_eps = size;
        ctx_params.oob.oob_ep    = rank;

        if (with_onesided) {
            onesided_ctx_params = ctx_params;
            for (auto i = 0; i < UCC_TEST_N_MEM_SEGMENTS; i++) {
                onesided_buffers[i] = ucc_calloc(UCC_TEST_MEM_SEGMENT_SIZE,
                                                 size, "onesided buffers");
                UCC_MALLOC_CHECK(onesided_buffers[i]);
                segments[i].address = onesided_buffers[i];
                segments[i].len     = UCC_TEST_MEM_SEGMENT_SIZE * size;
            }
            onesided_ctx_params.mask |= UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
            onesided_ctx_params.mem_params.segments   = segments;
            onesided_ctx_params.mem_params.n_segments = UCC_TEST_N_MEM_SEGMENTS;
        }
    }
    if (!with_onesided) {
        for (auto i = 0; i < UCC_TEST_N_MEM_SEGMENTS; i++) {
            onesided_buffers[i] = NULL;
        }
    }
    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
    UCC_CHECK(ucc_init(&lib_params, lib_config, &lib));
    ucc_lib_config_release(lib_config);
    UCC_CHECK(ucc_context_config_read(lib, NULL, &ctx_config));
    UCC_CHECK(ucc_context_create(lib, &ctx_params, ctx_config, &ctx));
    ucc_context_config_release(ctx_config);
    if (with_onesided) {
        prev_env = getenv("UCC_TL_UCP_TUNE");
        setenv("UCC_TL_UCP_TUNE", "alltoall:0-inf:@onesided#alltoallv:0-inf:@onesided", 1);
        UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_config));
        UCC_CHECK(ucc_init(&lib_params, lib_config, &onesided_lib));
        ucc_lib_config_release(lib_config);
        UCC_CHECK(ucc_context_config_read(onesided_lib, NULL, &ctx_config));
        UCC_CHECK(ucc_context_create(onesided_lib, &onesided_ctx_params,
                                     ctx_config, &onesided_ctx));
        ucc_context_config_release(ctx_config);
        if (prev_env) {
            putenv(prev_env);
        } else {
            unsetenv("UCC_TL_UCP_TUNE");
        }
    } else {
        onesided_lib = nullptr;
        onesided_ctx = nullptr;
    }
    set_msgsizes(8, ((1ULL) << 21), 8);
    dtypes             = {UCC_DT_INT16,           UCC_DT_INT32,
                          UCC_DT_INT64,           UCC_DT_UINT16,
                          UCC_DT_UINT32,          UCC_DT_UINT64,
                          UCC_DT_FLOAT32,         UCC_DT_FLOAT64,
                          UCC_DT_FLOAT128,        UCC_DT_FLOAT32_COMPLEX,
                          UCC_DT_FLOAT64_COMPLEX, UCC_DT_FLOAT128_COMPLEX};
    ops                = {UCC_OP_SUM, UCC_OP_MAX};
    colls              = {UCC_COLL_TYPE_BARRIER, UCC_COLL_TYPE_ALLREDUCE};
    mtypes             = {UCC_MEMORY_TYPE_HOST};
    inplace            = false;
    persistent         = false;
    root_type          = ROOT_RANDOM;
    root_value         = 10;
    iterations         = 1;
    triggered          = false;
    local_registration = false;
}

void UccTestMpi::set_iter(int iter)
{
    iterations = iter;
}

void UccTestMpi::set_verbose(bool _verbose)
{
    verbose = _verbose;
}

void UccTestMpi::create_teams(std::vector<ucc_test_mpi_team_t> &test_teams,
                              bool                              is_onesided)
{
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    for (auto &t : test_teams) {
        if (size < 4 && (t == TEAM_SPLIT_HALF || t == TEAM_SPLIT_ODD_EVEN)) {
            if (rank == 0) {
                std::cout << "size of the world=" << size <<
                    " is too small to create team " << team_str(t) <<
                    ", skipping ...\n";
            }
            continue;
        }
        create_team(t, is_onesided);
    }
}

UccTestMpi::~UccTestMpi()
{
    for (auto &t : teams) {
        destroy_team(t);
    }
    for (auto &t : onesided_teams) {
        destroy_team(t);
    }
    if (onesided_buffers[0]) {
        for (auto i = 0; i < UCC_TEST_N_MEM_SEGMENTS; i++) {
            ucc_free(onesided_buffers[i]);
        }
        UCC_CHECK(ucc_context_destroy(onesided_ctx));
        UCC_CHECK(ucc_finalize(onesided_lib));
    }
    UCC_CHECK(ucc_context_destroy(ctx));
    UCC_CHECK(ucc_finalize(lib));
}

ucc_team_h UccTestMpi::create_ucc_team(MPI_Comm comm, bool is_onesided)
{
    ucc_context_h     team_ctx = ctx;
    int               rank, size;
    ucc_team_h        team;
    ucc_team_params_t team_params;
    ucc_status_t      status;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    /* Create UCC TEAM for comm world */
    team_params.mask               = UCC_TEAM_PARAM_FIELD_EP       |
        UCC_TEAM_PARAM_FIELD_EP_RANGE |
        UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test  = oob_allgather_test;
    team_params.oob.req_free  = oob_allgather_free;
    team_params.oob.coll_info = (void*)(uintptr_t)comm;
    team_params.oob.n_oob_eps = size;
    team_params.oob.oob_ep    = rank;
    team_params.ep            = rank;
    team_params.ep_range      = UCC_COLLECTIVE_EP_RANGE_CONTIG;

    if (is_onesided) {
        team_params.mask |= UCC_TEAM_PARAM_FIELD_FLAGS;
        team_params.flags = UCC_TEAM_FLAG_COLL_WORK_BUFFER;
        team_ctx          = onesided_ctx;
    }
    UCC_CHECK(ucc_team_create_post(&team_ctx, 1, &team_params, &team));

    MPI_Request req;
    int tmp;
    int completed;
    MPI_Irecv(&tmp, 1, MPI_INT, rank, 123, comm, &req);
    while (UCC_INPROGRESS == (status = ucc_team_create_test(team))) {
        ucc_context_progress(team_ctx);
        MPI_Test(&req, &completed, MPI_STATUS_IGNORE);
    };
    MPI_Send(&tmp, 1, MPI_INT, rank, 123, comm);
    MPI_Wait(&req, MPI_STATUS_IGNORE);
    if (status < 0) {
        std::cerr << "*** UCC TEST FAIL: ucc_team_create_test failed\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return team;
}

void UccTestMpi::create_team(ucc_test_mpi_team_t t, bool is_onesided)
{
    ucc_team_h team;
    MPI_Comm comm = create_mpi_comm(t);
    if (is_onesided) {
        MPI_Comm comm_dup;
        MPI_Comm_dup(comm, &comm_dup);
        team = create_ucc_team(comm_dup, true);
        onesided_teams.push_back(ucc_test_team_t(t, comm_dup, team, onesided_ctx));
    } else {
        team = create_ucc_team(comm);
        teams.push_back(ucc_test_team_t(t, comm, team, ctx));
    }
}

void UccTestMpi::destroy_team(ucc_test_team_t &team)
{
    ucc_status_t status;

    team.free_ee();
    while (UCC_INPROGRESS == (status = ucc_team_destroy(team.team))) {}
    if (UCC_OK != status) {
        std::cerr << "ucc_team_destroy failed\n";
    }
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

int ucc_coll_reduce_supported(ucc_reduction_op_t op, ucc_datatype_t dt)
{
    switch (dt) {
    case UCC_DT_INT8:
    case UCC_DT_INT16:
    case UCC_DT_INT32:
    case UCC_DT_INT64:
    case UCC_DT_INT128:
    case UCC_DT_UINT8:
    case UCC_DT_UINT16:
    case UCC_DT_UINT32:
    case UCC_DT_UINT64:
    case UCC_DT_UINT128:
        return (op != UCC_OP_AVG);
    case UCC_DT_FLOAT16:
    case UCC_DT_FLOAT32:
    case UCC_DT_FLOAT64:
    case UCC_DT_BFLOAT16:
    case UCC_DT_FLOAT128:
        return (op == UCC_OP_SUM || op == UCC_OP_PROD || op == UCC_OP_MAX ||
                op == UCC_OP_MIN || op == UCC_OP_AVG);
    case UCC_DT_FLOAT32_COMPLEX:
    case UCC_DT_FLOAT64_COMPLEX:
    case UCC_DT_FLOAT128_COMPLEX:
        return (op == UCC_OP_SUM || op == UCC_OP_PROD || op == UCC_OP_AVG);
    default:
        return 0;
    }
}

int ucc_coll_inplace_supported(ucc_coll_type_t c)
{
    switch(c) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_BCAST:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
    /* remove alltoall [v] from here once it starts supporting inplace */
    case UCC_COLL_TYPE_ALLTOALL:
    case UCC_COLL_TYPE_ALLTOALLV:
    /**/
        return 0;
    default:
        return 1;
    }
}

bool ucc_coll_triggered_supported(ucc_memory_type_t mt)
{
    if (mt == UCC_MEMORY_TYPE_CUDA) {
        return true;
    }

    return false;
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
    case UCC_COLL_TYPE_REDUCE_SCATTER:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return 0;
    default:
        return 1;
    }
}

bool ucc_coll_has_memtype(ucc_coll_type_t c)
{
    switch(c) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return false;
    default:
        return true;
    }
}

bool ucc_coll_has_msgrange(ucc_coll_type_t c)
{
    switch(c) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return false;
    default:
        return true;
    }
}

bool ucc_coll_has_datatype(ucc_coll_type_t c)
{
    switch(c) {
    case UCC_COLL_TYPE_BARRIER:
    case UCC_COLL_TYPE_FANIN:
    case UCC_COLL_TYPE_FANOUT:
        return false;
    default:
        return true;
    }
}

bool ucc_coll_has_op(ucc_coll_type_t c)
{
    switch(c) {
    case UCC_COLL_TYPE_ALLREDUCE:
    case UCC_COLL_TYPE_REDUCE:
    case UCC_COLL_TYPE_REDUCE_SCATTER:
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return true;
    default:
        return false;
    }
}

bool ucc_coll_has_bits(ucc_coll_type_t c)
{
    switch(c) {
    case UCC_COLL_TYPE_ALLTOALLV:
        return true;
    default:
        return false;
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

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
test_set_gpu_device_t test_gpu_set_device = TEST_SET_DEV_NONE;
#endif

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
void set_gpu_device(test_set_gpu_device_t set_device)
{
    int local_rank = ucc_test_mpi_data.local_node_rank;
    int gpu_dev_count;
    int device_id;

    if (set_device == TEST_SET_DEV_NONE) {
        return;
    }

#if defined(HAVE_CUDA)
    CUDA_CHECK(cudaGetDeviceCount(&gpu_dev_count));
#elif defined(HAVE_HIP)
    HIP_CHECK(hipGetDeviceCount(&gpu_dev_count));
#endif
    switch (set_device) {
    case TEST_SET_DEV_LRANK:
        if(local_rank >= gpu_dev_count) {
            std::cerr << "*** UCC TEST FAIL: "
                      << "not enough GPU devices on the node to map processes.\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        device_id = local_rank;
        break;
    case TEST_SET_DEV_LRANK_ROUND:
        device_id = local_rank % gpu_dev_count;
        break;
    case TEST_SET_DEV_NONE:
    default:
        return;
    }
#if defined(HAVE_CUDA)
    CUDA_CHECK(cudaSetDevice(device_id));
    // Force CUDA context creation for the device
    // Without this, cuCtxGetCurrent() in TL/CUDA will return NULL
    CUDA_CHECK(cudaFree(0));
#elif defined(HAVE_HIP)
    HIP_CHECK(hipSetDevice(device_id));
    // Force HIP context creation for the device
    HIP_CHECK(hipFree(0));
#endif

}

#endif

std::vector<ucc_test_mpi_result_t> UccTestMpi::exec_tests(
        std::vector<std::shared_ptr<TestCase>> tcs, bool triggered,
                                                    bool persistent)
{
    int n_persistent = persistent ? UCC_TEST_N_PERSISTENT : 1;
    int world_rank, num_done, i;
    ucc_status_t status;

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    std::vector<ucc_test_mpi_result_t> rst;

    for (i = 0; i < n_persistent; i++) {
        for (auto tc: tcs) {
            if (TEST_SKIP_NONE == tc->test_skip) {
                if (verbose && 0 == world_rank) {
                    if (triggered) {
                        std::cout << "Triggered "<<tc->str() << std::endl;
                    } else {
                       std::cout << tc->str() << std::endl;
                    }
                }
                if (tc->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS) {
                    MPI_Barrier(MPI_COMM_WORLD);
                }
                tc->run(triggered);
            } else {
                if (verbose && 0 == world_rank) {
                    std::cout << "SKIPPED: " << skip_str(tc->test_skip) << ": "
                              << tc->str() << " " << std::endl;
                }
                rst.push_back(std::make_tuple(tc->args.coll_type, UCC_ERR_LAST));
                return rst;
            }
        }
        do {
            num_done = 0;
            for (auto tc: tcs) {
                tc->mpi_progress();
                status = tc->test();
                if (status < 0) {
                    std::cerr << "error during coll test: "
                              << ucc_status_string(status)
                              << " ("<<status<<")" << std::endl;
                    MPI_Abort(MPI_COMM_WORLD, -1);
                }
                if (status == UCC_OK) {
                    num_done++;
                }
                tc->tc_progress_ctx();
            }
        } while (num_done != tcs.size());
        for (auto tc: tcs) {
            status = tc->check();
            tc->set_input(i + 1);
            if (UCC_OK != status) {
                std::cerr << "FAILURE in: " << tc->str() << std::endl;
            }
            rst.push_back(std::make_tuple(tc->args.coll_type, status));
       }
    }
    return rst;
}

void UccTestMpi::run_all_at_team(ucc_test_team_t &team,
                                 std::vector<ucc_test_mpi_result_t> &rst)
{
    TestCaseParams params;

    params.max_size           = test_max_size;
    params.inplace            = inplace;
    params.persistent         = persistent;
    params.local_registration = local_registration;

    for (auto i = 0; i < iterations; i++) {
        for (auto &c : colls) {
            std::vector<int> roots = {0};
            std::vector<ucc_memory_type_t> test_memtypes = {UCC_MEMORY_TYPE_LAST};
            std::vector<size_t> test_msgsizes = {0};
            std::vector<ucc_datatype_t> test_dtypes = {(ucc_datatype_t)-1};
            std::vector<ucc_reduction_op_t> test_ops = {(ucc_reduction_op_t)-1};
            std::vector<ucc_test_vsize_flag_t> test_counts_vsize = {TEST_FLAG_VSIZE_64BIT};
            std::vector<ucc_test_vsize_flag_t> test_displ_vsize = {TEST_FLAG_VSIZE_64BIT};
            void **onesided_bufs;

            if (inplace && !ucc_coll_inplace_supported(c)) {
                continue;
            }

            if (ucc_coll_is_rooted(c)) {
                roots = gen_roots(team);
            }

            if (ucc_coll_has_memtype(c)) {
                test_memtypes = mtypes;
            }

            if (ucc_coll_has_msgrange(c)) {
                test_msgsizes = msgsizes;
            }

            if (ucc_coll_has_datatype(c)) {
                test_dtypes = dtypes;
            }

            if (ucc_coll_has_op(c)) {
                test_ops = ops;
            }

            if (ucc_coll_has_bits(c)) {
                test_counts_vsize = counts_vsize;
                test_displ_vsize = displs_vsize;
            }

            for (auto r : roots) {
                for (auto mt: test_memtypes) {
                    if (triggered && !ucc_coll_triggered_supported(mt)) {
                        rst.push_back(std::make_tuple(c, UCC_ERR_NOT_IMPLEMENTED));
                        continue;
                    }

                    if ((c == UCC_COLL_TYPE_ALLTOALL ||
                         c == UCC_COLL_TYPE_ALLTOALLV) &&
                        team.ctx != ctx) {
                        /* onesided alltoall */
                        if (mt != UCC_MEMORY_TYPE_HOST) {
                            continue;
                        } else {
                            onesided_bufs = onesided_buffers;
                        }
                    } else {
                        onesided_bufs = nullptr;
                    }

                    for (auto m: test_msgsizes) {
                        for (auto dt: test_dtypes) {
                            for (auto op: test_ops) {
                                if (ucc_coll_args_is_reduction(c) &&
                                    !ucc_coll_reduce_supported(op, dt)) {
                                    continue;
                                }

                                if (mt != UCC_MEMORY_TYPE_HOST &&
                                    (dt == UCC_DT_FLOAT128 ||
                                     dt == UCC_DT_FLOAT128_COMPLEX)) {
                                    continue;
                                }
                                for (auto count_bits: test_counts_vsize) {
                                    for (auto displ_bits: test_displ_vsize) {
                                        params.root       = r;
                                        params.mt         = mt;
                                        params.msgsize    = m;
                                        params.dt         = dt;
                                        params.op         = op;
                                        params.count_bits = count_bits;
                                        params.displ_bits = displ_bits;
                                        params.buffers    = onesided_bufs;

                                        auto tcs = TestCase::init(team, c, nt, params);
                                        auto res = exec_tests(tcs, triggered, persistent);
                                        rst.insert(rst.end(), res.begin(), res.end());
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

typedef struct ucc_test_thread {
    pthread_t                          thread;
    int                                id;
    UccTestMpi *                       test;
    std::vector<ucc_test_mpi_result_t> rst;
} ucc_test_thread_t;

static void *thread_start(void *arg)
{
    ucc_test_thread_t *t = (ucc_test_thread_t *)arg;

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
    set_gpu_device(test_gpu_set_device);
#endif
    t->test->run_all_at_team(t->test->teams[t->id], t->rst);
    return 0;
}

void UccTestMpi::run_all(bool is_onesided)
{
    if (UCC_THREAD_MULTIPLE == tm) {
        int                            n_threads = teams.size();
        std::vector<ucc_test_thread_t> threads(n_threads);
        void *                         ret;
        for (int i = 0; i < n_threads; i++) {
            threads[i].id   = i;
            threads[i].test = this;
            pthread_create(&threads[i].thread, NULL, thread_start,
                           (void *)&threads[i]);
        }
        for (int i = 0; i < n_threads; i++) {
            pthread_join(threads[i].thread, &ret);
            results.insert(results.end(), threads[i].rst.begin(),
                           threads[i].rst.end());
        }
    } else {
        if (!is_onesided) {
            for (auto &t : teams) {
                run_all_at_team(t, results);
            }
        } else {
            for (auto &t : onesided_teams) {
                run_all_at_team(t, results);
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
