/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#include "test_ucc.h"
extern "C" {
#include "core/ucc_team.h"
#include "components/tl/ucc_tl.h"
}
constexpr ucc_lib_params_t UccProcess::default_lib_params;
constexpr ucc_context_params_t UccProcess::default_ctx_params;
constexpr int UccJob::staticTeamSizes[];

UccProcess::UccProcess(int _job_rank, const ucc_lib_params_t &lib_params,
                       const ucc_context_params_t &_ctx_params)
{
    ucc_lib_config_h     lib_config;
    ucc_status_t         status;
    std::stringstream    err_msg;

    job_rank   = _job_rank;
    ctx_params = _ctx_params;
    status     = ucc_lib_config_read(NULL, NULL, &lib_config);
    if (status != UCC_OK) {
        err_msg << "ucc_lib_config_read failed";
        goto exit_err;
    }
    status = ucc_init(&lib_params, lib_config, &lib_h);
    ucc_lib_config_release(lib_config);
    if (status != UCC_OK) {
        err_msg << "ucc_init failed";
        goto exit_err;
    }
    return;

exit_err:
    err_msg << ": "<< ucc_status_string(status) << " (" << status << ")";
    throw std::runtime_error(err_msg.str());
}

UccProcess::~UccProcess()
{
    EXPECT_EQ(UCC_OK, ucc_context_destroy(ctx_h));
    EXPECT_EQ(UCC_OK, ucc_finalize(lib_h));
    if (ctx_params.mask & UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS) {
        for (auto i = 0; i < UCC_TEST_N_MEM_SEGMENTS; i++) {
            ucc_free(onesided_buf[i]);
        }
    }
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

uint64_t rank_map_cb(uint64_t ep, void *cb_ctx) {
    UccTeam *team = (UccTeam*)cb_ctx;
    return (uint64_t)team->procs[(int)ep].p.get()->job_rank;
}

void UccTeam::init_team(bool use_team_ep_map, bool use_ep_range,
                        bool is_onesided)
{
    ucc_team_params_t                    team_params;
    std::vector<allgather_coll_info_t *> cis;
    ucc_status_t                         status;
    for (int i = 0; i < n_procs; i++) {
        cis.push_back(new allgather_coll_info);
        cis.back()->self     = this;
        cis.back()->my_rank  = i;
        if (use_ep_range) {
            team_params.ep       = i;
            team_params.ep_range = UCC_COLLECTIVE_EP_RANGE_CONTIG;
            team_params.mask     = UCC_TEAM_PARAM_FIELD_EP  |
                UCC_TEAM_PARAM_FIELD_EP_RANGE;
        } else {
            team_params.mask = 0;
        }
        if (use_team_ep_map) {
            team_params.mask |= UCC_TEAM_PARAM_FIELD_EP_MAP;
            team_params.ep_map.type      = UCC_EP_MAP_CB;
            team_params.ep_map.ep_num    = n_procs;
            team_params.ep_map.cb.cb     = rank_map_cb;
            team_params.ep_map.cb.cb_ctx = (void*)this;
        } else {
            team_params.oob.allgather = allgather;
            team_params.oob.req_test  = req_test;
            team_params.oob.req_free  = req_free;
            team_params.oob.coll_info = (void *)cis.back();
            team_params.oob.n_oob_eps = n_procs;
            team_params.oob.oob_ep    = i;
            team_params.mask         |= UCC_TEAM_PARAM_FIELD_OOB;
        }
        if (is_onesided) {
            team_params.mask |= UCC_TEAM_PARAM_FIELD_FLAGS;
            team_params.flags = UCC_TEAM_FLAG_COLL_WORK_BUFFER;
        }
        EXPECT_EQ(UCC_OK,
                  ucc_team_create_post(&(procs[i].p.get()->ctx_h), 1, &team_params,
                                       &(procs[i].team)));
    }

    int all_done = 0;
    while (!all_done) {
        all_done = 1;
        for (int i = 0; i < n_procs; i++) {
            ucc_context_progress(procs[i].p.get()->ctx_h);
            status = ucc_team_create_test(procs[i].team);
            ASSERT_GE(status, 0);
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
                status = ucc_team_destroy(p.team);
                if (UCC_OK == status) {
                    p.team = NULL;
                } else if (status < 0) {
                    return;
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

UccTeam::UccTeam(std::vector<UccProcess_h> &_procs, bool use_team_ep_map,
                 bool use_ep_range, bool is_onesided)
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
    init_team(use_team_ep_map, use_ep_range, is_onesided);
    // test_allgather(128);
}

UccTeam::~UccTeam()
{
    destroy_team();
}

UccJob::UccJob(int _n_procs, ucc_job_ctx_mode_t _ctx_mode, ucc_job_env_t vars) :
    ta(_n_procs), n_procs(_n_procs), ctx_mode(_ctx_mode)

{
    ucc_job_env_t env_bkp;
    char *var;

    /* NCCL TL is disabled since it currently can not support non-blocking
       team creation. */
    vars.push_back({"UCC_TL_NCCL_TUNE", "0"});
    vars.push_back({"UCC_TL_RCCL_TUNE", "0"});
    /* CUDA TL is disabled since cuda context is not initialized in threads. */
    vars.push_back({"UCC_TL_CUDA_TUNE", "0"});
    /* GDR is temporarily disabled due to known issue that may result
       in a hang in the destruction flow */
    vars.push_back({"UCX_IB_GPU_DIRECT_RDMA", "no"});

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
        procs.push_back(std::make_shared<UccProcess>(i));
    }

    create_context();
    for (auto &v : env_bkp) {
        /*restore original env */
        setenv(v.first.c_str(), v.second.c_str(), 1);
    }
}

void thread_allgather(void *src_buf, void *recv_buf, size_t size,
                      ThreadAllgatherReq *ta_req)
{
    ThreadAllgather *ta = ta_req->ta;
    while (ta->ready_count > ta->n_procs) {
        std::this_thread::yield();
    }
    ta->lock.lock();
    if (!ta->buffer) {
        ucc_assert(0 == ta->ready_count);
        ta->buffer = malloc(size * ta->n_procs);
        ta->ready_count = 0;
    }
    memcpy((void*)((ptrdiff_t)ta->buffer + size * ta_req->rank),
           src_buf, size);
    ta->ready_count++;
    ta->lock.unlock();
    while (ta->ready_count < ta->n_procs) {
        std::this_thread::yield();
    }
    memcpy(recv_buf, ta->buffer, size * ta->n_procs);

    ta->lock.lock();
    ta->ready_count++;
    if (ta->ready_count == 2 * ta->n_procs) {
        free(ta->buffer);
        ta->buffer = NULL;
        ta->ready_count = 0;
    }
    ta->lock.unlock();
    ta_req->status = UCC_OK;
}

ucc_status_t thread_allgather_start(void *src_buf, void *recv_buf, size_t size,
                                    void *coll_info, void **request)
{
    ThreadAllgatherReq *ta_req = (ThreadAllgatherReq*)coll_info;
    *request = coll_info;
    while (ta_req->status != UCC_OPERATION_INITIALIZED) {
        std::this_thread::yield();
    }

    ta_req->status = UCC_INPROGRESS;
    ta_req->t = std::thread(thread_allgather, src_buf,
                            recv_buf, size, ta_req);
    return UCC_OK;
}

ucc_status_t thread_allgather_req_test(void *request)
{
    ThreadAllgatherReq *ta_req = (ThreadAllgatherReq*)request;
    return ta_req->status;
}

ucc_status_t thread_allgather_req_free(void *request)
{
    ThreadAllgatherReq *ta_req = (ThreadAllgatherReq*)request;
    ta_req->t.join();
    ta_req->status = UCC_OPERATION_INITIALIZED;
    return UCC_OK;
}

void proc_context_create(UccProcess_h proc, int id, ThreadAllgather *ta, bool is_global)
{
    ucc_status_t status;
    ucc_context_config_h ctx_config;
    std::stringstream    err_msg;

    status = ucc_context_config_read(proc->lib_h, NULL, &ctx_config);
    if (status != UCC_OK) {
        err_msg << "ucc_context_config_read failed";
        goto exit_err;
    }
    if (is_global) {
        proc->ctx_params.mask |= UCC_CONTEXT_PARAM_FIELD_OOB;
        proc->ctx_params.oob.allgather = thread_allgather_start;
        proc->ctx_params.oob.req_test  = thread_allgather_req_test;
        proc->ctx_params.oob.req_free  = thread_allgather_req_free;
        proc->ctx_params.oob.coll_info = (void*) &ta->reqs[id];
        proc->ctx_params.oob.n_oob_eps = ta->n_procs;
        proc->ctx_params.oob.oob_ep    = id;
    }
    status = ucc_context_create(proc->lib_h, &proc->ctx_params, ctx_config, &proc->ctx_h);
    ucc_context_config_release(ctx_config);
    if (status != UCC_OK) {
        err_msg << "ucc_context_create failed";
        goto exit_err;
    }
    return;

exit_err:
    err_msg << ": "<< ucc_status_string(status) << " (" << status << ")";
    throw std::runtime_error(err_msg.str());
}

void proc_context_create_mem_params(UccProcess_h proc, int id,
                                    ThreadAllgather *ta)
{
    ucc_status_t         status;
    ucc_context_config_h ctx_config;
    std::stringstream    err_msg;
    ucc_mem_map_t        map[UCC_TEST_N_MEM_SEGMENTS];

    status = ucc_context_config_read(proc->lib_h, NULL, &ctx_config);
    if (status != UCC_OK) {
        err_msg << "ucc_context_config_read failed";
        goto exit_err;
    }
    for (auto i = 0; i < UCC_TEST_N_MEM_SEGMENTS; i++) {
        proc->onesided_buf[i] =
            ucc_calloc(UCC_TEST_MEM_SEGMENT_SIZE, 1, "onesided_buffer");
        EXPECT_NE(proc->onesided_buf[i], nullptr);
        map[i].address = proc->onesided_buf[i];
        map[i].len     = UCC_TEST_MEM_SEGMENT_SIZE;
    }
    proc->ctx_params.mask = UCC_CONTEXT_PARAM_FIELD_OOB;
    proc->ctx_params.mask |= UCC_CONTEXT_PARAM_FIELD_MEM_PARAMS;
    proc->ctx_params.oob.allgather         = thread_allgather_start;
    proc->ctx_params.oob.req_test          = thread_allgather_req_test;
    proc->ctx_params.oob.req_free          = thread_allgather_req_free;
    proc->ctx_params.oob.coll_info         = (void *)&ta->reqs[id];
    proc->ctx_params.oob.n_oob_eps         = ta->n_procs;
    proc->ctx_params.oob.oob_ep            = id;
    proc->ctx_params.mem_params.segments   = map;
    proc->ctx_params.mem_params.n_segments = UCC_TEST_N_MEM_SEGMENTS;
    status = ucc_context_create(proc->lib_h, &proc->ctx_params, ctx_config,
                                &proc->ctx_h);
    ucc_context_config_release(ctx_config);
    if (status != UCC_OK) {
        err_msg << "ucc_context_create for one-sided context failed";
        goto exit_err;
    }
    return;

exit_err:
    err_msg << ": " << ucc_status_string(status) << " (" << status << ")";
    throw std::runtime_error(err_msg.str());
}

void UccJob::create_context()
{
    std::vector<std::thread> workers;
    for (auto i = 0; i < procs.size(); i++) {
        if (ctx_mode == UCC_JOB_CTX_GLOBAL_ONESIDED) {
            workers.push_back(
                std::thread(proc_context_create_mem_params, procs[i], i, &ta));
        } else {
            workers.push_back(std::thread(proc_context_create, procs[i], i, &ta,
                                          ctx_mode == UCC_JOB_CTX_GLOBAL));
        }
    }
    for (auto i = 0; i < procs.size(); i++) {
        workers[i].join();
    }
}

void thread_proc_destruct(std::vector<UccProcess_h> *procs, int i)
{
    ucc_assert(true == (*procs)[i].unique());
    (*procs)[i] = NULL;
}

UccJob::~UccJob()
{
    std::vector<std::thread> workers;

    if (this == UccJob::staticUccJob) {
        staticTeams.clear();
    }
    for (int i = 0; i < n_procs; i++) {
        workers.push_back(std::thread(thread_proc_destruct, &procs, i));
    }
    for (int i = 0; i < n_procs; i++) {
        workers[i].join();
    }
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
            if (ts == 1 && !tl_self_available()) {
                /* don't use team_size = 1 if there is no tl/self.
                   we can't modify nStaticTeams, so just use some other
                   team_size */
                ts = 3;
            }
            staticTeams.push_back(getStaticJob()->create_team(ts));
        }
        /* Create one more team with reversed ranks order */
        std::vector<int> ranks;
        for (auto r = staticUccJobSize - 1; r >= 0; r--) {
            ranks.push_back(r);
        }
        staticTeams.push_back(getStaticJob()->create_team(ranks, true));
    }

    return staticTeams;
}

void UccJob::cleanup()
{
    if (staticUccJob) {
        delete staticUccJob;
    }
}

UccTeam_h UccJob::create_team(int _n_procs, bool use_team_ep_map,
                              bool use_ep_range, bool is_onesided)
{
    EXPECT_GE(n_procs, _n_procs);
    std::vector<UccProcess_h> team_procs;
    for (int i = 0; i < _n_procs; i++) {
        team_procs.push_back(procs[i]);
    }
    return std::make_shared<UccTeam>(team_procs, use_team_ep_map, use_ep_range,
                                     is_onesided);
}

UccTeam_h UccJob::create_team(std::vector<int> &ranks, bool use_team_ep_map,
                              bool use_ep_range, bool is_onesided)
{
    EXPECT_GE(n_procs, ranks.size());
    std::vector<UccProcess_h> team_procs;
    for (int i = 0; i < ranks.size(); i++) {
        team_procs.push_back(procs[ranks[i]]);
    }
    return std::make_shared<UccTeam>(team_procs, use_team_ep_map, use_ep_range,
                                     is_onesided);
}

UccReq::UccReq(UccTeam_h _team, ucc_coll_args_t *args) :
    team(_team)
{
    ucc_coll_req_h req;
    for (auto &p : team->procs) {
        if (UCC_OK != ucc_collective_init(args, &req, p.team)) {
            goto err;
        }
        reqs.push_back(req);
    }
    return;
err:
    reqs.clear();
}

UccReq::UccReq(UccTeam_h _team, UccCollCtxVec ctxs) :
        team(_team)
{
    std::vector<ucc_status_t> err_st;
    ucc_coll_req_h            req;
    ucc_status_t              st;

    EXPECT_EQ(team->procs.size(), ctxs.size());

    status = UCC_OK;
    for (auto i = 0; i < team->procs.size(); i++) {
        if (!ctxs[i]) {
            continue;
        }
        if (UCC_OK !=(st = ucc_collective_init(ctxs[i]->args, &req,
                                               team->procs[i].team))) {
            err_st.push_back(st);
        } else {
            reqs.push_back(req);
        }
    }
    if (err_st.size() > 0) {
        /* All error status should be equal, otherwise it is
           real fatal error. Only expected error is NOT_SUPPORTED.
           If collective init returns NOT_SUPPORTED it has to be
           symmetric for all ranks */
        if (!std::equal(err_st.begin() + 1, err_st.end(), err_st.begin()) ||
            err_st.size() != team->procs.size() ||
            err_st[0] != UCC_ERR_NOT_SUPPORTED) {
            status = UCC_ERR_NO_MESSAGE;
        } else {
            ucc_assert(err_st[0] = UCC_ERR_NOT_SUPPORTED);
            status = err_st[0];
        }
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
    ucc_status_t st;

    for (auto r : reqs) {
        st = ucc_collective_post(r);
        ASSERT_EQ(UCC_OK, st);
        st = ucc_collective_test(r);
        ASSERT_NE(UCC_OPERATION_INITIALIZED, st);
    }
}

ucc_status_t UccReq::test()
{
    ucc_status_t st = UCC_OK;
    for (auto r : reqs) {
        st = ucc_collective_test(r);
        if (UCC_OK != st) {
            break;
        }
    }
    return st;
}

ucc_status_t UccReq::wait()
{
    ucc_status_t st;
    while (UCC_OK != (st = test())) {
        if (st < 0) {
            break;
        }
        team->progress();
    }
    return st;
}

void UccReq::waitall(std::vector<UccReq> &reqs)
{
    bool alldone = false;
    ucc_status_t status;
    while (!alldone) {
        alldone = true;
        for (auto &r : reqs) {
            if (UCC_OK != (status = r.test())) {
                if (status < 0) {
                    return;
                }
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

void UccCollArgs::set_mem_type(ucc_memory_type_t _mt)
{
    mem_type = _mt;
}

void UccCollArgs::set_inplace(gtest_ucc_inplace_t _inplace)
{
    inplace = _inplace;
}

void clear_buffer(void *_buf, size_t size, ucc_memory_type_t mt, uint8_t value)
{
    void *buf = _buf;
    if (mt != UCC_MEMORY_TYPE_HOST) {
        buf = ucc_malloc(size, "buf");
        ASSERT_NE(0, (uintptr_t)buf);
    }
    memset(buf, value, size);
    if (UCC_MEMORY_TYPE_HOST != mt) {
        UCC_CHECK(ucc_mc_memcpy(_buf, buf, size, mt, UCC_MEMORY_TYPE_HOST));
        ucc_free(buf);
    }
}

bool tl_self_available()
{
    ucc_tl_context_t *tl_ctx;
    ucc_status_t      status;

    status = ucc_tl_context_get(UccJob::getStaticJob()->procs[0]->ctx_h,
                                "self", &tl_ctx);

    if (UCC_OK != status) {
        return false;
    }
    ucc_tl_context_put(tl_ctx);
    return true;
}
