/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef TEST_MPI_H
#define TEST_MPI_H
#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <unistd.h>
#include <mpi.h>
#include <ucc/api/ucc.h>
extern "C" {
#include "utils/ucc_malloc.h"
}
BEGIN_C_DECLS
#include "components/mc/ucc_mc.h"
#include "core/ucc_team.h"
#include "utils/ucc_math.h"
END_C_DECLS
#ifdef HAVE_CUDA
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#ifdef HAVE_HIP
#include <hip/hip_runtime_api.h>
#endif

#define STR(x) #x
#define UCC_CHECK(_call)                                                \
    if (UCC_OK != (_call)) {                                            \
        std::cerr << "*** UCC TEST FAIL: " << STR(_call) << "\n";       \
        MPI_Abort(MPI_COMM_WORLD, -1);                                  \
    }

#define UCC_MALLOC_CHECK(_obj)                                          \
    if (!(_obj)) {                                                      \
        std::cerr << "*** UCC MALLOC FAIL \n";                          \
        MPI_Abort(MPI_COMM_WORLD, -1);                                  \
    }

#define UCC_CHECK_SKIP(_call, _skip_cause)                              \
    {                                                                   \
        ucc_status_t status;                                            \
        status = (_call);                                               \
        if(UCC_ERR_NOT_SUPPORTED == status)  {                          \
            _skip_cause = TEST_SKIP_NOT_SUPPORTED;                      \
        } else if (UCC_ERR_NOT_IMPLEMENTED == status) {                 \
            _skip_cause = TEST_SKIP_NOT_IMPLEMENTED;                    \
        } else if (UCC_OK != status) {                                  \
            std::cerr << "*** UCC TEST FAIL: " << STR(_call) << "\n";   \
            MPI_Abort(MPI_COMM_WORLD, -1);                              \
        }                                                               \
    }

#define TEST_UCC_RANK_BUF_SIZE_MAX (8*1024*1024)

extern int test_rand_seed;

#define UCC_ALLOC_COPY_BUF(_new_buf, _new_mtype, _old_buf, _old_mtype, _size)  \
    {                                                                          \
        UCC_CHECK(ucc_mc_alloc(&(_new_buf), _size, _new_mtype));               \
        UCC_CHECK(ucc_mc_memcpy(_new_buf->addr, _old_buf, _size, _new_mtype,   \
                                _old_mtype));                                  \
    }

#ifdef HAVE_CUDA
#define CUDA_CHECK(_call) {                                         \
    cudaError_t cuda_err = (_call);                                 \
    if (cudaSuccess != (cuda_err)) {                                \
        std::cerr << "*** UCC TEST FAIL: " << STR(_call) << ": "    \
                  << cudaGetErrorString(cuda_err) << "\n" ;         \
        MPI_Abort(MPI_COMM_WORLD, -1);                              \
    }                                                               \
}
#endif

#ifdef HAVE_HIP
#define HIP_CHECK(_call) {                                          \
    hipError_t hip_err = (_call);                                   \
    if (hipSuccess != (hip_err)) {                                  \
        std::cerr << "*** UCC TEST FAIL: " << STR(_call) << ": "    \
                  << hipGetErrorString(hip_err) << "\n" ;           \
        MPI_Abort(MPI_COMM_WORLD, -1);                              \
    }                                                               \
}
#endif

#define UCC_TEST_N_PERSISTENT 4
#define UCC_TEST_N_MEM_SEGMENTS   3
#define UCC_TEST_MEM_SEGMENT_SIZE (1 << 21)

typedef enum {
    MEM_SEND_SEGMENT,
    MEM_RECV_SEGMENT,
    MEM_WORK_SEGMENT
} ucc_test_mem_segments;

typedef enum {
    TEAM_WORLD,
    TEAM_REVERSE,
    TEAM_SPLIT_HALF,
    TEAM_SPLIT_ODD_EVEN,
    TEAM_LAST
} ucc_test_mpi_team_t;

typedef enum {
    ROOT_SINGLE,
    ROOT_RANDOM,
    ROOT_ALL
} ucc_test_mpi_root_t;

typedef enum {
    TEST_FLAG_VSIZE_32BIT,
    TEST_FLAG_VSIZE_64BIT
} ucc_test_vsize_flag_t;

typedef enum {
    TEST_SKIP_NONE,
    TEST_SKIP_NOT_SUPPORTED,
    TEST_SKIP_NOT_IMPLEMENTED,
    TEST_SKIP_MEM_LIMIT,
    TEST_SKIP_LAST
} test_skip_cause_t;

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
typedef enum {
    TEST_SET_DEV_NONE,
    TEST_SET_DEV_LRANK,
    TEST_SET_DEV_LRANK_ROUND
} test_set_gpu_device_t;
#endif

static inline const char* skip_str(test_skip_cause_t s) {
    switch(s) {
    case TEST_SKIP_MEM_LIMIT:
        return "maximum buffer size reached";
    case TEST_SKIP_NOT_SUPPORTED:
        return "not supported";
    case TEST_SKIP_NOT_IMPLEMENTED:
        return "not implemented";
    default:
        return "unknown";
    }
}

static inline const char* team_str(ucc_test_mpi_team_t t) {
    switch(t) {
    case TEAM_WORLD:
        return "world";
    case TEAM_REVERSE:
        return "reverse";
    case TEAM_SPLIT_HALF:
        return "half";
    case TEAM_SPLIT_ODD_EVEN:
        return "odd_even";
    default:
        break;
    }
    return NULL;
}

typedef struct ucc_test_mpi_data {
    int local_node_rank;
} ucc_test_mpi_data_t;
extern ucc_test_mpi_data_t ucc_test_mpi_data;

int init_test_mpi_data(void);

#if defined(HAVE_CUDA) || defined(HAVE_HIP)
void set_gpu_device(test_set_gpu_device_t set_device);
#endif
int ucc_coll_inplace_supported(ucc_coll_type_t c);
int ucc_coll_is_rooted(ucc_coll_type_t c);
bool ucc_coll_has_datatype(ucc_coll_type_t c);

typedef struct ucc_test_team {
#ifdef HAVE_CUDA
    ucc_ee_h cuda_ee;
    cudaStream_t cuda_stream;
#endif
    ucc_test_mpi_team_t type;
    MPI_Comm comm;
    ucc_team_h team;
    ucc_context_h ctx;
    ucc_test_team(ucc_test_mpi_team_t _type, MPI_Comm _comm,
                  ucc_team_h _team, ucc_context_h _ctx) :
    type(_type), comm(_comm), team(_team), ctx(_ctx)
    {
#ifdef HAVE_CUDA
        cuda_stream = nullptr;
        cuda_ee     = nullptr;
#endif
    };

#ifdef HAVE_CUDA
    ucc_status_t get_cuda_ee(ucc_ee_h *ee)
    {
        ucc_ee_params_t ee_params;

        if (!cuda_ee) {
            CUDA_CHECK(cudaStreamCreateWithFlags(&cuda_stream,
                                                 cudaStreamNonBlocking));
            ee_params.ee_type         = UCC_EE_CUDA_STREAM;
            ee_params.ee_context_size = sizeof(cudaStream_t);
            ee_params.ee_context      = cuda_stream;
            UCC_CHECK(ucc_ee_create(team, &ee_params, &cuda_ee));
        }

        *ee = cuda_ee;
        return UCC_OK;
    }

    void free_cuda_ee()
    {
        if (cuda_ee) {
            UCC_CHECK(ucc_ee_destroy(cuda_ee));
            CUDA_CHECK(cudaStreamDestroy(cuda_stream));
        }
    }
#endif

    ucc_status_t get_ee(ucc_ee_type_t ee_type, ucc_ee_h *ee)
    {
        switch (ee_type) {
#ifdef HAVE_CUDA
        case UCC_EE_CUDA_STREAM:
            return get_cuda_ee(ee);
#endif
        default:
            return UCC_ERR_NOT_SUPPORTED;

        }
    }

    void free_ee()
    {
#ifdef HAVE_CUDA
        free_cuda_ee();
#endif
    }

} ucc_test_team_t;

struct TestCaseParams {
    size_t msgsize;
    bool inplace;
    bool persistent;
    bool local_registration;
    ucc_datatype_t dt;
    ucc_reduction_op_t op;
    ucc_memory_type_t mt;
    size_t max_size;
    int root;
    void **buffers;
    ucc_test_vsize_flag_t count_bits;
    ucc_test_vsize_flag_t displ_bits;
};

class TestCase {
protected:
    ucc_test_team_t team;
    ucc_memory_type_t mem_type;
    int root;
    size_t msgsize;
    bool inplace;
    bool persistent;
    bool local_registration;
    ucc_coll_req_h req;
    ucc_mem_map_mem_h src_memh;
    size_t src_memh_size;
    ucc_mem_map_mem_h dst_memh;
    size_t dst_memh_size;
    ucc_mc_buffer_header_t *sbuf_mc_header, *rbuf_mc_header;
    void *sbuf;
    void *rbuf;
    void *check_buf;
    MPI_Request progress_request;
    uint8_t     progress_buf[1];
    size_t test_max_size;
    ucc_datatype_t dt;
    int iter_persistent;
public:
    ucc_coll_args_t args;
    void mpi_progress(void);
    test_skip_cause_t test_skip;
    static std::shared_ptr<TestCase> init_single(
            ucc_test_team_t &_team,
            ucc_coll_type_t _type,
            TestCaseParams params);
    static std::vector<std::shared_ptr<TestCase>> init(
            ucc_test_team_t &_team,
            ucc_coll_type_t _type,
            int num_tests,
            TestCaseParams params);
    TestCase(ucc_test_team_t &_team, ucc_coll_type_t ct, TestCaseParams params);
    virtual ~TestCase();
    virtual void run(bool triggered);
    virtual ucc_status_t set_input(int iter_persistent = 0) = 0;
    virtual ucc_status_t check() = 0;
    virtual std::string str();
    virtual ucc_status_t test();
    void wait();
    void tc_progress_ctx();
    test_skip_cause_t skip_reduce(test_skip_cause_t cause, MPI_Comm comm);
    test_skip_cause_t skip_reduce(int skip_cond, test_skip_cause_t cause,
                                  MPI_Comm comm);
};

typedef std::tuple<ucc_coll_type_t, ucc_status_t> ucc_test_mpi_result_t;
class UccTestMpi {
    ucc_thread_mode_t         tm;
    ucc_context_h             ctx;
    ucc_context_h             onesided_ctx;
    ucc_lib_h                 lib;
    int                       nt;
    ucc_lib_h                 onesided_lib;
    bool                      inplace;
    bool                      persistent;
    ucc_test_mpi_root_t       root_type;
    int                       root_value;
    int                       iterations;
    bool                      verbose;
    void *                    onesided_buffers[3];
    size_t                    test_max_size;
    bool                      triggered;
    bool                      local_registration;
    void create_team(ucc_test_mpi_team_t t, bool is_onesided = false);
    void destroy_team(ucc_test_team_t &team);
    ucc_team_h create_ucc_team(MPI_Comm comm, bool is_onesided = false);
    std::vector<size_t> msgsizes;
    std::vector<ucc_memory_type_t> mtypes;
    std::vector<ucc_datatype_t> dtypes;
    std::vector<ucc_reduction_op_t> ops;
    std::vector<ucc_coll_type_t> colls;
    std::vector<int> gen_roots(ucc_test_team_t &team);
    std::vector<ucc_test_vsize_flag_t> counts_vsize;
    std::vector<ucc_test_vsize_flag_t> displs_vsize;
    std::vector<ucc_test_mpi_result_t> exec_tests(
            std::vector<std::shared_ptr<TestCase>> tcs,
            bool triggered, bool persistent);
public:
    std::vector<ucc_test_team_t> teams;
    std::vector<ucc_test_team_t> onesided_teams;
    void run_all_at_team(ucc_test_team_t &team,
                         std::vector<ucc_test_mpi_result_t> &rst);
    std::vector<ucc_test_mpi_result_t> results;
    UccTestMpi(int argc, char *argv[], ucc_thread_mode_t tm, int is_local,
               bool with_onesided);
    ~UccTestMpi();
    void set_msgsizes(size_t min, size_t max, size_t power);
    void set_dtypes(std::vector<ucc_datatype_t> &_dtypes);
    void set_colls(std::vector<ucc_coll_type_t> &_colls);
    void set_iter(int iter);
    void set_verbose(bool verbose);
    void set_ops(std::vector<ucc_reduction_op_t> &_ops);
    void set_mtypes(std::vector<ucc_memory_type_t> &_mtypes);
    void set_inplace(bool _inplace)
    {
        inplace = _inplace;
    }
    void set_persistent(bool _persistent)
    {
        persistent = _persistent;
    }
    void set_triggered(bool _triggered)
    {
        triggered = _triggered;
    }
    void set_local_registration(bool _local_registration)
    {
        local_registration = _local_registration;
    }
    void set_count_vsizes(std::vector<ucc_test_vsize_flag_t> &_counts_vsize);
    void set_displ_vsizes(std::vector<ucc_test_vsize_flag_t> &_displs_vsize);
    void run_all(bool is_onesided = false);
    void set_root(ucc_test_mpi_root_t _root_type, int _root_value) {
        root_type = _root_type;
        root_value = _root_value;
    };
    void set_max_size(size_t _max_size) {
        test_max_size = _max_size;
    }
    void set_num_tests(int num_tests) {
        nt = num_tests;
    }
    void create_teams(std::vector<ucc_test_mpi_team_t> &test_teams,
                      bool                              is_onesided = false);
    void progress_ctx() {
        ucc_context_progress(ctx);
        if (onesided_ctx) {
            ucc_context_progress(onesided_ctx);
        }
    }
};

class TestAllgather : public TestCase {
public:
    TestAllgather(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
};

class TestAllgatherv : public TestCase {
    int *counts;
    int *displacements;
public:
    TestAllgatherv(ucc_test_team_t &team, TestCaseParams &params);
    ~TestAllgatherv();
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check() override;
};

class TestAllreduce : public TestCase {
    ucc_reduction_op_t op;
public:
    TestAllreduce(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
    std::string str();
};

class TestAlltoall : public TestCase {
    bool is_onesided;
public:
    TestAlltoall(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
};

class TestAlltoallv : public TestCase {
    size_t sncounts;
    size_t rncounts;
    int *scounts;
    int *sdispls;
    int *rcounts;
    int *rdispls;
    ucc_count_t *scounts64;
    ucc_count_t *sdispls64;
    ucc_count_t *rcounts64;
    ucc_count_t *rdispls64;
    ucc_test_vsize_flag_t count_bits;
    ucc_test_vsize_flag_t displ_bits;

    template<typename T>
    void * mpi_counts_to_ucc(int *mpi_counts, size_t _ncount);
public:
    TestAlltoallv(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
    std::string str();
    ~TestAlltoallv();
};

class TestBarrier : public TestCase {
    ucc_status_t status;
public:
    TestBarrier(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
    std::string str();
    void run(bool triggered);
    ucc_status_t test();
};

class TestBcast : public TestCase {
public:
    TestBcast(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
};

class TestGather : public TestCase {
public:
    TestGather(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
};

class TestGatherv : public TestCase {
    uint32_t *counts;
    uint32_t *displacements;
public:
    TestGatherv(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
    ~TestGatherv();
};

class TestReduce : public TestCase {
	ucc_reduction_op_t op;
public:
    TestReduce(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
    std::string  str();
};

class TestReduceScatter : public TestCase {
    ucc_reduction_op_t op;
public:
    TestReduceScatter(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ~TestReduceScatter();
    ucc_status_t check();
    std::string str();
};

class TestReduceScatterv : public TestCase {
    ucc_reduction_op_t op;
    int *              counts;
  public:
    TestReduceScatterv(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ~TestReduceScatterv();
    ucc_status_t check();
    std::string  str();
};

class TestScatter : public TestCase {
public:
    TestScatter(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
};

class TestScatterv : public TestCase {
    uint32_t *counts;
    uint32_t *displacements;
public:
    TestScatterv(ucc_test_team_t &team, TestCaseParams &params);
    ucc_status_t set_input(int iter_persistent = 0) override;
    ucc_status_t check();
    ~TestScatterv();
};

void init_buffer(void *buf, size_t count, ucc_datatype_t dt,
                 ucc_memory_type_t mt, int value, int offset = 0);

ucc_status_t compare_buffers(void *rst, void *expected, size_t count,
                             ucc_datatype_t dt, ucc_memory_type_t mt);

ucc_status_t divide_buffer(void *expected, size_t divider, size_t count,
                           ucc_datatype_t dt);


#endif
