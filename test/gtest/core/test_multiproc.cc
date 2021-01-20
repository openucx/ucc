/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "test_multiproc.h"
extern "C" {
#include "core/ucc_team.h"
}
void test_process::init_lib()
{
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &lib_config));
    lib_params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    EXPECT_EQ(UCC_OK, ucc_init(&lib_params, lib_config, &lib_h));
    ucc_lib_config_release(lib_config);
}

void test_process::init_ctx()
{
    EXPECT_EQ(UCC_OK, ucc_context_config_read(lib_h, NULL, &ctx_config));
    ctx_params.mask     = UCC_CONTEXT_PARAM_FIELD_TYPE;
    ctx_params.ctx_type = UCC_CONTEXT_EXCLUSIVE;
    EXPECT_EQ(UCC_OK,
              ucc_context_create(lib_h, &ctx_params, ctx_config, &ctx_h));
}

void test_process::fini_lib()
{
    EXPECT_EQ(UCC_OK, ucc_finalize(lib_h));
}

void test_process::fini_ctx()
{
    EXPECT_EQ(UCC_OK, ucc_context_destroy(ctx_h));
}

test_multiproc::test_multiproc(int _n_procs) : n_procs(_n_procs)
{
    ag.resize(n_procs);
    for (int i = 0; i < n_procs; i++) {
        auto p = new test_process();
        p->init_lib();
        p->init_ctx();
        procs.push_back(p);
        ag[i].phase = AG_INIT;
    }
    copy_complete_count = 0;
}

test_multiproc::~test_multiproc()
{
    for (auto p : procs) {
        p->fini_ctx();
        p->fini_lib();
        delete p;
    }
}

ucc_status_t allgather(void *src_buf, void *recv_buf, size_t size,
                       void *coll_info, void **request)
{
    test_multiproc::allgather_coll_info_t *ci =
        (test_multiproc::allgather_coll_info_t *)coll_info;
    int my_rank                 = ci->my_rank;
    ci->self->ag[my_rank].sbuf  = src_buf;
    ci->self->ag[my_rank].rbuf  = recv_buf;
    ci->self->ag[my_rank].len   = size;
    ci->self->ag[my_rank].phase = test_multiproc::AG_READY;
    *request                    = (void *)ci;
    return UCC_OK;
}

ucc_status_t req_test(void *request)
{
    test_multiproc::allgather_coll_info_t *ci =
        (test_multiproc::allgather_coll_info_t *)request;
    int n_procs = ci->self->n_procs;
    switch (ci->self->ag[ci->my_rank].phase) {
    case test_multiproc::AG_READY:
        for (int i = 0; i < n_procs; i++) {
            if (ci->self->ag[i].phase == test_multiproc::AG_INIT) {
                return UCC_INPROGRESS;
            }
        }
        for (int i = 0; i < n_procs; i++) {
            memcpy((void *)((ptrdiff_t)ci->self->ag[ci->my_rank].rbuf +
                            i * ci->self->ag[i].len),
                   ci->self->ag[i].sbuf, ci->self->ag[i].len);
        }
        ci->self->ag[ci->my_rank].phase = test_multiproc::AG_COPY_DONE;
        ;
        ci->self->copy_complete_count++;
        break;
    case test_multiproc::AG_COPY_DONE:
        if (ci->my_rank == 0 && ci->self->copy_complete_count == n_procs) {
            for (int i = 0; i < n_procs; i++) {
                ci->self->ag[i].phase = test_multiproc::AG_COMPLETE;
            }
            ci->self->copy_complete_count = 0;
        }
        break;
    case test_multiproc::AG_COMPLETE:
        return UCC_OK;
    default:
        break;
    }
    return UCC_INPROGRESS;
}

ucc_status_t req_free(void *request)
{
    return UCC_OK;
}

void test_multiproc::init_team()
{
    ucc_team_params_t                    team_params;
    std::vector<allgather_coll_info_t *> cis;
    ucc_status_t                         status;
    for (int i = 0; i < n_procs; i++) {
        cis.push_back(new allgather_coll_info);
        cis.back()->self             = this;
        cis.back()->my_rank          = i;
        team_params.oob.allgather    = allgather;
        team_params.oob.req_test     = req_test;
        team_params.oob.req_free     = req_free;
        team_params.oob.coll_info    = (void *)cis.back();
        team_params.oob.participants = n_procs;
        team_params.mask             = UCC_TEAM_PARAM_FIELD_OOB;
        EXPECT_EQ(UCC_OK,
                  ucc_team_create_post(&(procs[i]->ctx_h), 1, &team_params,
                                       &(procs[i]->team)));
    }

    int all_done = 0;
    while (!all_done) {
        all_done = 1;
        for (int i = 0; i < n_procs; i++) {
            status = ucc_team_create_test(procs[i]->team);
            EXPECT_GE(status, 0);
            if (UCC_INPROGRESS == status) {
                all_done = 0;
            }
        }
    }
    for (auto c : cis) {
        delete c;
    }
}

void test_multiproc::destroy_team()
{
    ucc_status_t status;
    bool         all_done;
    do {
        all_done = true;
        for (auto p : procs) {
            if (p->team) {
                status = ucc_team_destroy_nb(p->team);
                if (UCC_OK == status) {
                    p->team = NULL;
                } else {
                    all_done = false;
                }
            }
        }
    } while (!all_done);
}

class test_team : public ucc::test {
  public:
    test_multiproc *tm;
};

UCC_TEST_F(test_team, team_create_destroy)
{
    char *env;
    /* Minimal - 2 procs */
    tm = new test_multiproc(2);
    tm->init_team();
    tm->destroy_team();
    delete tm;
    /* Some power of 2 procs  - 8*/
    tm = new test_multiproc(8);
    tm->init_team();
    tm->destroy_team();
    delete tm;

    /* Non power of 2 procs  - 7*/
    tm = new test_multiproc(7);
    tm->init_team();
    tm->destroy_team();
    delete tm;

    /* From env var*/
    env = getenv("UCC_GTEST_TEST_TEAM_NPROCS");
    if (env) {
        tm = new test_multiproc(atoi(env));
        tm->init_team();
        tm->destroy_team();
        delete tm;
    }
}
