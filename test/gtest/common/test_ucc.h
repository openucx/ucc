/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef TEST_UCC_H
#define TEST_UCC_H
#include "test.h"
#include "ucc/api/ucc.h"
extern "C" {
#include "core/ucc_mc.h"
#include "utils/ucc_malloc.h"
}
#include <vector>
#include <tuple>
#include <memory>

typedef struct {
    ucc_mc_buffer_header_t *dst_mc_header;
    ucc_mc_buffer_header_t *src_mc_header;
    void                   *init_buf;
    size_t                  rbuf_size;
    ucc_coll_args_t        *args;
} gtest_ucc_coll_ctx_t;

typedef std::vector<gtest_ucc_coll_ctx_t*> UccCollCtxVec;

typedef enum {
    TEST_NO_INPLACE,
    TEST_INPLACE
} gtest_ucc_inplace_t;

class UccCollArgs {
protected: 
    ucc_memory_type_t mem_type;
    gtest_ucc_inplace_t inplace;
    void alltoallx_init_buf(int src_rank, int dst_rank, uint8_t *buf, size_t len)
    {
        for (int i = 0; i < len; i++) {
            buf[i] = (uint8_t)(((src_rank + len - i) *
                                (dst_rank + 1)) % UINT8_MAX);
        }
    }
    int alltoallx_validate_buf(int src_rank, int dst_rank, uint8_t *buf, size_t len)
    {
        int err = 0;
        for (int i = 0; i < len; i ++) {
            uint8_t expected = (uint8_t)
                    (((dst_rank + len - i) *
                      (src_rank + 1)) % UINT8_MAX);
            if (buf[i] != expected) {
                err++;
            }
        }
        return err;
    }
public:
    UccCollArgs() {
        // defaults
        mem_type = UCC_MEMORY_TYPE_HOST;
        inplace = TEST_NO_INPLACE;
    }
    virtual ~UccCollArgs() {}
    virtual void data_init(int nprocs, ucc_datatype_t dtype,
                           size_t count, UccCollCtxVec &args) = 0;
    virtual void data_fini(UccCollCtxVec args) = 0;
    virtual bool data_validate(UccCollCtxVec args) = 0;
    void set_mem_type(ucc_memory_type_t _mt);
    void set_inplace(gtest_ucc_inplace_t _inplace);
};

/* A single processes in a Job that runs UCC.
   It has context and lib object */
class UccProcess {
public:
    static constexpr ucc_lib_params_t default_lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE |
                UCC_LIB_PARAM_FIELD_COLL_TYPES,
        .thread_mode = UCC_THREAD_SINGLE,
        .coll_types = UCC_COLL_TYPE_BARRIER |
                      UCC_COLL_TYPE_ALLTOALL |
                      UCC_COLL_TYPE_ALLTOALLV |
                      UCC_COLL_TYPE_ALLREDUCE |
                      UCC_COLL_TYPE_ALLGATHER |
                      UCC_COLL_TYPE_ALLGATHERV |
                      UCC_COLL_TYPE_BCAST
    };
    static constexpr ucc_context_params_t default_ctx_params = {
        .mask = UCC_CONTEXT_PARAM_FIELD_TYPE,
        .type = UCC_CONTEXT_EXCLUSIVE
    };
    ucc_lib_h            lib_h;
    ucc_context_h        ctx_h;
    UccProcess(const ucc_lib_params_t &lp = default_lib_params,
               const ucc_context_params_t &cp = default_ctx_params);
    ~UccProcess();
};
typedef std::shared_ptr<UccProcess> UccProcess_h;

/* Ucc team that consists of several processes. The team
   is created from UccJob environment */
class UccTeam {
    struct proc {
        UccProcess_h p;
        ucc_team_h     team;
        proc(){};
        proc(UccProcess_h _p) : p(_p) {};
    };
    typedef enum {
        AG_INIT,
        AG_READY,
        AG_COPY_DONE,
        AG_COMPLETE
    } allgather_phase_t;
    struct allgather_data {
        void             *sbuf;
        void             *rbuf;
        size_t            len;
        allgather_phase_t phase;
    };
    typedef struct allgather_coll_info {
        int             my_rank;
        UccTeam *self;
    } allgather_coll_info_t;
    std::vector<struct allgather_data> ag;
    void init_team();
    void destroy_team();
    void test_allgather(size_t msglen);
    static ucc_status_t allgather(void *src_buf, void *recv_buf, size_t size,
                                  void *coll_info, void **request);
    static ucc_status_t req_test(void *request);
    static ucc_status_t req_free(void *request);
    int                                copy_complete_count;
public:
    int n_procs;
    void progress();
    std::vector<proc> procs;
    UccTeam(std::vector<UccProcess_h> &_procs);
    ~UccTeam();
};
typedef std::shared_ptr<UccTeam> UccTeam_h;
typedef std::pair<std::string, std::string> ucc_env_var_t;
typedef std::vector<ucc_env_var_t> ucc_job_env_t;
/* UccJob - environent that has n_procs processes.
   Multiple UccTeams can be created from UccJob */
class UccJob {
    static UccJob* staticUccJob;
    static std::vector<UccTeam_h> staticTeams;
public:
    static const int nStaticTeams     = 3;
    static const int staticUccJobSize = 16;
    static constexpr int staticTeamSizes[nStaticTeams] = {2, 11, 16};
    static void cleanup();
    static UccJob* getStaticJob();
    static const std::vector<UccTeam_h> &getStaticTeams();
    int n_procs;
    UccJob(int _n_procs = 2);
    UccJob(int _n_procs, ucc_job_env_t vars);
    ~UccJob();
    std::vector<UccProcess_h> procs;
    UccTeam_h create_team(int n_procs);

};

class UccReq {
    UccTeam_h team;
    /* Make copy constructor and = private,
       to avoid req leak */
public:
    UccReq(const UccReq&) = delete;
    UccReq& operator=(const UccReq&) = delete;
    UccReq(UccReq&& source) : team(source.team) {
        reqs.swap(source.reqs);
    };

    std::vector<ucc_coll_req_h> reqs;
    UccReq(UccTeam_h _team, ucc_coll_args_t *args);
    UccReq(UccTeam_h _team, UccCollCtxVec args);
    ~UccReq();
    void start(void);
    void wait();
    ucc_status_t test(void);
    static void waitall(std::vector<UccReq> &reqs);
    static void startall(std::vector<UccReq> &reqs);
};

#endif

