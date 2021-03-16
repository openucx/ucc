/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "test_ucc.h"
extern "C" {
#include "core/ucc_team.h"
}
constexpr ucc_lib_params_t UccProcess::default_lib_params;
constexpr ucc_context_params_t UccProcess::default_ctx_params;
constexpr int UccJob::staticTeamSizes[];
UccProcess::UccProcess(const ucc_lib_params_t &lib_params,
                       const ucc_context_params_t &ctx_params)
{
    ucc_lib_config_h     lib_config;
    ucc_context_config_h ctx_config;
    EXPECT_EQ(UCC_OK, ucc_lib_config_read(NULL, NULL, &lib_config));
    EXPECT_EQ(UCC_OK, ucc_init(&lib_params, lib_config, &lib_h));
    ucc_lib_config_release(lib_config);
    EXPECT_EQ(UCC_OK, ucc_context_config_read(lib_h, NULL, &ctx_config));
    EXPECT_EQ(UCC_OK,
              ucc_context_create(lib_h, &ctx_params, ctx_config, &ctx_h));
    ucc_context_config_release(ctx_config);
}

UccProcess::~UccProcess()
{
    EXPECT_EQ(UCC_OK, ucc_context_destroy(ctx_h));
    EXPECT_EQ(UCC_OK, ucc_finalize(lib_h));
}

ucc_status_t UccTeam::allgather(void *src_buf, void *recv_buf, size_t size,
                                void *coll_info, void **request)
{
    UccTeam::allgather_coll_info_t *ci =
        (UccTeam::allgather_coll_info_t *)coll_info;
    int my_rank                 = ci->my_rank;
    ci->self->ag[my_rank].sbuf  = src_buf;
    ci->self->ag[my_rank].rbuf  = recv_buf;
    ci->self->ag[my_rank].len   = size;
    ci->self->ag[my_rank].phase = UccTeam::AG_READY;
    *request                    = (void *)ci;
    return UCC_OK;
}

void UccTeam::test_allgather(size_t msglen)
{
    int *sbufs[n_procs];
    int *rbufs[n_procs];
    size_t count = msglen/sizeof(int);
    std::vector<allgather_coll_info_t> cis;
    std::vector<void *> reqs;
    for (int i=0; i<n_procs; i++) {
        sbufs[i] = new int[count];
        rbufs[i] = new int[count*n_procs];
        memset(rbufs[i], 0, count*n_procs);
        for (int j=0; j<count; j++) {
            sbufs[i][j] = i*count + j;
        }
        cis.push_back(allgather_coll_info_t());
        cis.back().self = this;
        cis.back().my_rank = i;
    }
    for (int i=0; i<n_procs; i++) {
        void *req;
        allgather(sbufs[i], rbufs[i], msglen, (void*)&cis[i], &req);
        reqs.push_back(req);
    }
    int all_done = 0;
    while (!all_done) {
        all_done = 1;
        for (int i=0; i<n_procs; i++) {
            if (!reqs[i]) continue;
            if (UCC_OK != req_test(reqs[i])) {
                all_done = 0;
            } else {
                req_free(reqs[i]);
                reqs[i] = NULL;
            }
        }
    }
    int correct = 1;
    for (int k=0; k<n_procs; k++) {
        for (int i=0; i<n_procs; i++) {
            for (int j=0; j<count; j++) {
                if (rbufs[k][i*count+j] != i*count + j) {
                    correct = 0;
                }
            }
        }
    }
    EXPECT_EQ(1, correct);
}

ucc_status_t UccTeam::req_test(void *request)
{
    UccTeam::allgather_coll_info_t *ci =
        (UccTeam::allgather_coll_info_t *)request;
    int n_procs = ci->self->n_procs;
    switch (ci->self->ag[ci->my_rank].phase) {
    case UccTeam::AG_READY:
        for (int i = 0; i < n_procs; i++) {
            if ((ci->self->ag[i].phase == UccTeam::AG_INIT) ||
                (ci->self->ag[i].phase == UccTeam::AG_COMPLETE)) {
                return UCC_INPROGRESS;
            }
        }
        for (int i = 0; i < n_procs; i++) {
            memcpy((void *)((ptrdiff_t)ci->self->ag[ci->my_rank].rbuf +
                            i * ci->self->ag[i].len),
                   ci->self->ag[i].sbuf, ci->self->ag[i].len);
        }
        ci->self->ag[ci->my_rank].phase = UccTeam::AG_COPY_DONE;
        ;
        ci->self->copy_complete_count++;
        break;
    case UccTeam::AG_COPY_DONE:
        if (ci->my_rank == 0 && ci->self->copy_complete_count == n_procs) {
            for (int i = 0; i < n_procs; i++) {
                ci->self->ag[i].phase = UccTeam::AG_COMPLETE;
            }
            ci->self->copy_complete_count = 0;
        }
        break;
    case UccTeam::AG_COMPLETE:
        return UCC_OK;
    default:
        break;
    }
    return UCC_INPROGRESS;
}

ucc_status_t UccTeam::req_free(void *request)
{
    UccTeam::allgather_coll_info_t *ci =
        (UccTeam::allgather_coll_info_t *)request;
    ci->self->ag[ci->my_rank].phase = UccTeam::AG_INIT;
    return UCC_OK;
}

void UccTeam::init_team()
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
        team_params.ep               = i;
        team_params.ep_range         = UCC_COLLECTIVE_EP_RANGE_CONTIG;
        team_params.mask             = UCC_TEAM_PARAM_FIELD_OOB |
            UCC_TEAM_PARAM_FIELD_EP  |
            UCC_TEAM_PARAM_FIELD_EP_RANGE ;
        EXPECT_EQ(UCC_OK,
                  ucc_team_create_post(&(procs[i].p.get()->ctx_h), 1, &team_params,
                                       &(procs[i].team)));
    }

    int all_done = 0;
    while (!all_done) {
        all_done = 1;
        for (int i = 0; i < n_procs; i++) {
            status = ucc_team_create_test(procs[i].team);
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


void UccTeam::destroy_team()
{
    ucc_status_t status;
    bool         all_done;
    do {
        all_done = true;
        for (auto &p : procs) {
            if (p.team) {
                status = ucc_team_destroy_nb(p.team);
                if (UCC_OK == status) {
                    p.team = NULL;
                } else {
                    all_done = false;
                }
            }
        }
    } while (!all_done);
}

void UccTeam::progress()
{
    for (auto &p : procs) {
        ucc_context_progress(p.p->ctx_h);
    }
}

UccTeam::UccTeam(std::vector<UccProcess_h> &_procs)
{
    n_procs = _procs.size();
    ag.resize(n_procs);
    for (auto &p : _procs) {
        procs.push_back(proc(p));
    }
    for (auto &a : ag) {
        a.phase = AG_INIT;
    }
    copy_complete_count = 0;
    init_team();
    // test_allgather(128);
}

UccTeam::~UccTeam()
{
    destroy_team();
}

UccJob::UccJob(int _n_procs) : n_procs(_n_procs)
{
    for (int i = 0; i < n_procs; i++) {
        procs.push_back(std::make_shared<UccProcess>());
    }
}

UccJob::UccJob(int _n_procs, ucc_job_env_t vars) : n_procs(_n_procs)
{
    ucc_job_env_t env_bkp;
    char *var;
    for (auto &v : vars) {
        var = std::getenv(v.first.c_str());
        if (var) {
            /* found env - back it up for later restore
               after processes creation */
            env_bkp.push_back(ucc_env_var_t(v.first, var));
        }
        setenv(v.first.c_str(), v.second.c_str(), 1);
    }
    for (int i = 0; i < n_procs; i++) {
        procs.push_back(std::make_shared<UccProcess>());
    }

    for (auto &v : env_bkp) {
        /*restore original env */
        setenv(v.first.c_str(), v.second.c_str(), 1);
    }
}

UccJob::~UccJob()
{
}

UccJob* UccJob::staticUccJob = NULL;

UccJob* UccJob::getStaticJob()
{
    if (!staticUccJob) {
        staticUccJob = new UccJob(UccJob::staticUccJobSize);
    }
    return staticUccJob;
}

std::vector<UccTeam_h> UccJob::staticTeams;

const std::vector<UccTeam_h> &UccJob::getStaticTeams()
{
    std::vector<int> teamSizes(std::begin(staticTeamSizes),
                               std::end(staticTeamSizes));
    if (0 == staticTeams.size()) {
        for (auto ts : teamSizes) {
            staticTeams.push_back(getStaticJob()->create_team(ts));
        }
    }
    return staticTeams;
}

void UccJob::cleanup()
{
    staticTeams.clear();
    if (staticUccJob) {
        delete staticUccJob;
    }
}

UccTeam_h UccJob::create_team(int _n_procs)
{
    EXPECT_GE(n_procs, _n_procs);
    std::vector<UccProcess_h> team_procs;
    for (int i=0; i<_n_procs; i++) {
        team_procs.push_back(procs[i]);
    }
    return std::make_shared<UccTeam>(team_procs);
}


UccReq::UccReq(UccTeam_h _team, ucc_coll_args_t *args) :
    team(_team)
{
    ucc_coll_req_h req;
    for (auto &p : team->procs) {
        EXPECT_EQ(UCC_OK, ucc_collective_init(args, &req, p.team));
        reqs.push_back(req);
    }
}

UccReq::UccReq(UccTeam_h _team, UccCollArgsVec args) :
        team(_team)
{
    EXPECT_EQ(team->procs.size(), args.size());
    ucc_coll_req_h req;
    for (auto i = 0; i < team->procs.size(); i++) {
        EXPECT_EQ(UCC_OK, ucc_collective_init(args[i], &req, team->procs[i].team));
        reqs.push_back(req);
    }
}

UccReq::~UccReq()
{
    for (auto r : reqs) {
        EXPECT_EQ(UCC_OK, ucc_collective_finalize(r));
    }
}

void UccReq::start()
{
    for (auto r : reqs) {
        EXPECT_EQ(UCC_OK, ucc_collective_post(r));
    }
}

ucc_status_t UccReq::test()
{
    ucc_status_t status = UCC_OK;
    for (auto r : reqs) {
        status = ucc_collective_test(r);
        if (UCC_OK != status) {
            break;
        }
    }
    EXPECT_GE(status, 0);
    return status;
}

void UccReq::wait()
{
    while (UCC_OK != test()) {
        team->progress();
    }
}

void UccReq::waitall(std::vector<UccReq> &reqs)
{
    bool alldone = false;
    while (!alldone) {
        alldone = true;
        for (auto &r : reqs) {
            if (UCC_OK != r.test()) {
                alldone = false;
                r.team->progress();
            }
        }
    }
}

void UccReq::startall(std::vector<UccReq> &reqs)
{
    for (auto &r : reqs) {
        r.start();
    }
}
