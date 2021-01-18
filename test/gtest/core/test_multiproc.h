/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_MULTIPROC_H
#define UCC_TEST_MULTIPROC_H
#include <common/test.h>
#include <vector>

class test_process {
  public:
    ucc_lib_config_h     lib_config;
    ucc_lib_params_t     lib_params;
    ucc_lib_h            lib_h;
    ucc_context_config_h ctx_config;
    ucc_context_params_t ctx_params;
    ucc_context_h        ctx_h;
    ucc_team_params_t    team_params;
    ucc_team_h           team;
    void                 init_lib();
    void                 init_ctx();
    void                 fini_lib();
    void                 fini_ctx();
};

class test_multiproc {
  public:
    int n_procs;
    test_multiproc(int _n_procs = 2);
    ~test_multiproc();
    std::vector<test_process *> procs;
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
        test_multiproc *self;
    } allgather_coll_info_t;
    std::vector<struct allgather_data> ag;
    int                                copy_complete_count;
    void                               init_team();
    void                               destroy_team();
};

#endif
