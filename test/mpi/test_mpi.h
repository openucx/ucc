/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
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
BEGIN_C_DECLS
#include "core/ucc_mc.h"
#include "utils/ucc_math.h"
END_C_DECLS

#define STR(x) #x
#define UCC_CHECK(_call)                                            \
    if (UCC_OK != (_call)) {                                        \
        std::cerr << "*** UCC TEST FAIL: " << STR(_call) << "\n";   \
        MPI_Abort(MPI_COMM_WORLD, -1);                              \
    }

extern int test_rand_seed;
extern size_t test_max_size;

#define UCC_ALLOC_COPY_BUF(_new_buf, _new_mtype, _old_buf, _old_mtype, _size) \
{                                                                             \
    UCC_CHECK(ucc_mc_alloc(&(_new_buf), _size, _new_mtype));                  \
    UCC_CHECK(ucc_mc_memcpy(_new_buf, _old_buf, _size,                        \
              _new_mtype, _old_mtype));                                       \
}

typedef enum {
    TEAM_WORLD,
    TEAM_REVERSE,
    TEAM_SPLIT_HALF,
    TEAM_SPLIT_ODD_EVEN,
    TEAM_LAST
} ucc_test_mpi_team_t;

typedef enum {
    TEST_NO_INPLACE,
    TEST_INPLACE
} ucc_test_mpi_inplace_t;

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
    TEST_SKIP_MEM_LIMIT,
    TEST_SKIP_LAST
} test_skip_cause_t;

static inline const char* skip_str(test_skip_cause_t s) {
    switch(s) {
    case TEST_SKIP_MEM_LIMIT:
        return "maximum buffer size reached";
    default:
        return "unknown";
    }
    return NULL;
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

int ucc_coll_inplace_supported(ucc_coll_type_t c);
int ucc_coll_is_rooted(ucc_coll_type_t c);

typedef struct ucc_test_team {
    ucc_test_mpi_team_t type;
    MPI_Comm comm;
    ucc_team_h team;
    ucc_context_h ctx;
    ucc_test_team(ucc_test_mpi_team_t _type, MPI_Comm _comm,
                  ucc_team_h _team, ucc_context_h _ctx) :
    type(_type), comm(_comm), team(_team), ctx(_ctx) {};
} ucc_test_team_t;

class UccTestMpi {
    ucc_context_h ctx;
    ucc_lib_h     lib;
    ucc_test_mpi_inplace_t inplace;
    ucc_test_mpi_root_t    root_type;
    int                    root_value;
    void create_team(ucc_test_mpi_team_t t);
    void destroy_team(ucc_test_team_t &team);
    ucc_team_h create_ucc_team(MPI_Comm comm);
    std::vector<ucc_test_team_t> teams;
    std::vector<size_t> msgsizes;
    std::vector<ucc_memory_type_t> mtypes;
    std::vector<ucc_datatype_t> dtypes;
    std::vector<ucc_reduction_op_t> ops;
    std::vector<ucc_coll_type_t> colls;
    std::vector<int> gen_roots(ucc_test_team_t &team);
    std::vector<ucc_test_vsize_flag_t> counts_vsize;
    std::vector<ucc_test_vsize_flag_t> displs_vsize;
public:
    std::vector<ucc_status_t> results;
    UccTestMpi(int argc, char *argv[], ucc_thread_mode_t tm,
               std::vector<ucc_test_mpi_team_t> &test_teams,
               const char* cls = NULL);
    ~UccTestMpi();
    void set_msgsizes(size_t min, size_t max, size_t power);
    void set_dtypes(std::vector<ucc_datatype_t> &_dtypes);
    void set_colls(std::vector<ucc_coll_type_t> &_colls);
    void set_ops(std::vector<ucc_reduction_op_t> &_ops);
    void set_mtypes(std::vector<ucc_memory_type_t> &_mtypes);
    void set_inplace(ucc_test_mpi_inplace_t _inplace) {
        inplace = _inplace;
    }
    void set_count_vsizes(std::vector<ucc_test_vsize_flag_t> &_counts_vsize);
    void set_displ_vsizes(std::vector<ucc_test_vsize_flag_t> &_displs_vsize);
    void run_all();
    void set_root(ucc_test_mpi_root_t _root_type, int _root_value) {
        root_type = _root_type;
        root_value = _root_value;
    };
};

class TestCase {
protected:
    ucc_test_team_t team;
    ucc_memory_type_t mem_type;
    int root;
    size_t msgsize;
    ucc_test_mpi_inplace_t inplace;
    ucc_coll_args_t args;
    ucc_coll_req_h req;
    void *sbuf;
    void *rbuf;
    void *check_sbuf;
    void *check_rbuf;
    MPI_Request progress_request;
    uint8_t     progress_buf[1];
    void mpi_progress(void);
    test_skip_cause_t test_skip;
public:
    static std::shared_ptr<TestCase> init(ucc_coll_type_t _type,
                                          ucc_test_team_t &_team,
                                          int    root    = 0,
                                          size_t msgsize = 0,
                                          ucc_test_mpi_inplace_t inplace = TEST_NO_INPLACE,
                                          ucc_memory_type_t mt = UCC_MEMORY_TYPE_HOST,
                                          ucc_datatype_t dt = UCC_DT_INT32,
                                          ucc_reduction_op_t op = UCC_OP_SUM,
                                          ucc_test_vsize_flag_t count_vsize = TEST_FLAG_VSIZE_64BIT,
                                          ucc_test_vsize_flag_t displ_vsize = TEST_FLAG_VSIZE_64BIT);

    TestCase(ucc_test_team_t &_team, ucc_memory_type_t _mem_type = UCC_MEMORY_TYPE_UNKNOWN,
             size_t _msgsize = 0, ucc_test_mpi_inplace_t _inplace = TEST_NO_INPLACE);
    virtual ~TestCase();
    virtual void run();
    virtual ucc_status_t check() = 0;
    virtual std::string str();
    virtual ucc_status_t test();
    void wait();
    ucc_status_t exec();
    test_skip_cause_t skip(int skip_cond, test_skip_cause_t cause,
                           MPI_Comm comm);
};

class TestBarrier : public TestCase {
    ucc_status_t status;
public:
    TestBarrier(ucc_test_team_t &team);
    ucc_status_t check();
    std::string str();
    void run();
    ucc_status_t test();
};

class TestAllreduce : public TestCase {
    ucc_datatype_t dt;
    ucc_reduction_op_t op;
public:
    TestAllreduce(size_t _msgsize, ucc_test_mpi_inplace_t inplace,
                  ucc_datatype_t _dt, ucc_reduction_op_t _op,
                  ucc_memory_type_t _mt, ucc_test_team_t &team);
    ucc_status_t check();
    std::string str();
};

class TestAllgather : public TestCase {
public:
    TestAllgather(size_t _msgsize, ucc_test_mpi_inplace_t inplace,
                  ucc_memory_type_t _mt, ucc_test_team_t &team);
    ucc_status_t check();
};

class TestAllgatherv : public TestCase {
    int *counts;
    int *displacements;
public:
    TestAllgatherv(size_t _msgsize, ucc_test_mpi_inplace_t inplace,
                   ucc_memory_type_t _mt, ucc_test_team_t &team);
    ~TestAllgatherv();
    ucc_status_t check() override;
};

class TestBcast : public TestCase {
public:
    TestBcast(size_t _msgsize, ucc_memory_type_t _mt, int root,
              ucc_test_team_t &team);
    ucc_status_t check();
};

class TestAlltoall : public TestCase {
    ucc_datatype_t dt;
public:
    TestAlltoall(size_t _msgsize, ucc_test_mpi_inplace_t _inplace,
                 ucc_datatype_t _dt, ucc_memory_type_t _mt,
                 ucc_test_team_t &_team);
    ucc_status_t check();
    std::string str();
};

class TestAlltoallv : public TestCase {
    ucc_datatype_t dt;
    size_t sncounts;
    size_t rncounts;
    int *scounts;
    int *sdispls;
    int *rcounts;
    int *rdispls;
    ucc_test_vsize_flag_t count_bits;
    ucc_test_vsize_flag_t displ_bits;

    template<typename T>
    static void* _mpi_counts_to_ucc(int *mpi_counts, size_t _ncount)
    {
        void *ucc_counts = (T*)malloc(sizeof(T) * _ncount);
        for (auto i = 0; i < _ncount; i++) {
            ((T*)ucc_counts)[i] = mpi_counts[i];
        }
        return ucc_counts;
    }
public:
    TestAlltoallv(size_t _msgsize, ucc_test_mpi_inplace_t inplace,
                  ucc_datatype_t _dt, ucc_memory_type_t _mt,
                  ucc_test_team_t &_team,
                  ucc_test_vsize_flag_t count_bits,
                  ucc_test_vsize_flag_t displ_bits);
    ucc_status_t check();
    std::string str();
    ~TestAlltoallv()
    {
        if (scounts) {
            free(scounts);
        }
        if (sdispls) {
            free(sdispls);
        }
        if (rcounts) {
            free(rcounts);
        }
        if (rdispls) {
            free(rdispls);
        }
    }
};

void init_buffer(void *buf, size_t count, ucc_datatype_t dt,
                 ucc_memory_type_t mt, int value);

ucc_status_t compare_buffers(void *rst, void *expected, size_t count,
                             ucc_datatype_t dt, ucc_memory_type_t mt);
#endif
