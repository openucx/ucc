/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef TEST_UCC_H
#define TEST_UCC_H
#include "test.h"
#include <vector>
#include <tuple>
#include <memory>

/* A single processes in a Job that runs UCC.
   It has context and lib object */
class UccProcess {
public:
    static constexpr ucc_lib_params_t default_lib_params = {
        .mask = UCC_LIB_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCC_THREAD_SINGLE
    };
    static constexpr ucc_context_params_t default_ctx_params = {
        .mask = UCC_CONTEXT_PARAM_FIELD_TYPE,
        .ctx_type = UCC_CONTEXT_EXCLUSIVE
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
    static const int staticUccJobSize = 16;
    static constexpr int staticTeamSizes[] = {2, 7, 8};
    static UccJob* staticUccJob;
    static std::vector<UccTeam_h> staticTeams;
public:
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
    UccReq(UccTeam_h _team, ucc_coll_op_args_t *args);
    ~UccReq();
    void start(void);
    void wait();
    ucc_status_t test(void);
    static void waitall(std::vector<UccReq> &reqs);
    static void startall(std::vector<UccReq> &reqs);
};

#endif

