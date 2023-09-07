/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, ucc_datatype_t, ucc_memory_type_t, int, int>;
using Param_1 = std::tuple<ucc_datatype_t, ucc_memory_type_t, int, int>;

class test_bcast : public UccCollArgs, public ucc::test
{
private:
    int root;
public:
    void data_init(int nprocs, ucc_datatype_t dtype, size_t count,
                   UccCollCtxVec &ctxs, bool persistent)
    {
        ctxs.resize(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask = 0;
            coll->coll_type = UCC_COLL_TYPE_BCAST;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dtype;
            coll->root = root;

            ctxs[r]->rbuf_size = ucc_dt_size(dtype) * count;

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header, ctxs[r]->rbuf_size,
                                   mem_type));
            coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
            if (r == root) {
                ctxs[r]->init_buf = ucc_malloc(ctxs[r]->rbuf_size, "init buf");
                EXPECT_NE(ctxs[r]->init_buf, nullptr);
                uint8_t *sbuf = (uint8_t*)ctxs[r]->init_buf;
                for (int i = 0; i < ctxs[r]->rbuf_size; i++) {
                    sbuf[i] = (uint8_t)i;
                }
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                                        ctxs[r]->rbuf_size, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            }
            if (persistent) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
            }
        }
    }
    void reset(UccCollCtxVec ctxs)
    {
        for (auto r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t *coll  = ctxs[r]->args;
            size_t           count = coll->src.info.count;
            ucc_datatype_t   dtype = coll->src.info.datatype;
            if (r != root) {
                clear_buffer(coll->src.info.buffer, count * ucc_dt_size(dtype),
                             mem_type, 0);
            } else {
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                                        ctxs[r]->rbuf_size, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            }
        }
    }

    void data_fini(UccCollCtxVec ctxs)
    {
        for (auto r = 0; r < ctxs.size(); r++) {
            gtest_ucc_coll_ctx_t *ctx = ctxs[r];
            ucc_coll_args_t* coll = ctx->args;
            UCC_CHECK(ucc_mc_free(ctx->src_mc_header));
            if (r == coll->root) {
                ucc_free(ctx->init_buf);
            }
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }
    bool data_validate(UccCollCtxVec ctxs)
    {
        bool     ret  = true;
        int      root = ctxs[0]->args->root;
        uint8_t *dsts;

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
                dsts = (uint8_t*) ucc_malloc(ctxs[root]->rbuf_size, "dsts buf");
                EXPECT_NE(dsts, nullptr);
                UCC_CHECK(ucc_mc_memcpy(dsts, ctxs[root]->args->src.info.buffer,
                                        ctxs[root]->rbuf_size,
                                        UCC_MEMORY_TYPE_HOST, mem_type));
        } else {
            dsts = (uint8_t*)ctxs[root]->args->src.info.buffer;
        }
        for (int r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t* coll = ctxs[r]->args;
            if (coll->root == r) {
                continue;
            }
            for (int i = 0; i < ctxs[r]->rbuf_size; i++) {
                if ((uint8_t)i != dsts[i]) {
                    ret = false;
                    break;
                }
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            ucc_free(dsts);
        }
        return ret;
    }
    void set_root(int _root)
    {
        root = _root;
    }
};

class test_bcast_0 : public test_bcast,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_bcast_0, single)
{
    const int                 team_id  = std::get<0>(GetParam());
    const ucc_datatype_t      dtype    = std::get<1>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<2>(GetParam());
    const int                 count    = std::get<3>(GetParam());
    const int                 root     = std::get<4>(GetParam());
    UccTeam_h                 team     = UccJob::getStaticTeams()[team_id];
    int                       size     = team->procs.size();
    UccCollCtxVec             ctxs;

    SET_MEM_TYPE(mem_type);
    set_root(root);

    data_init(size, dtype, count, ctxs, false);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

UCC_TEST_P(test_bcast_0, single_persistent)
{
    const int               team_id  = std::get<0>(GetParam());
    const ucc_datatype_t    dtype    = std::get<1>(GetParam());
    const ucc_memory_type_t mem_type = std::get<2>(GetParam());
    const int               count    = std::get<3>(GetParam());
    const int               root     = std::get<4>(GetParam());
    UccTeam_h               team     = UccJob::getStaticTeams()[team_id];
    int                     size     = team->procs.size();
    const int               n_calls  = 3;
    UccCollCtxVec           ctxs;

    SET_MEM_TYPE(mem_type);
    set_root(root);

    data_init(size, dtype, count, ctxs, true);
    UccReq req(team, ctxs);

    for (auto i = 0; i < n_calls; i++) {
        req.start();
        req.wait();
        EXPECT_EQ(true, data_validate(ctxs));
        reset(ctxs);
    }

    data_fini(ctxs);
}

INSTANTIATE_TEST_CASE_P(
    , test_bcast_0,
    ::testing::Combine(
        ::testing::Range(1, UccJob::nStaticTeams), // team_ids
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA,
                          UCC_MEMORY_TYPE_CUDA_MANAGED),
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,65536), // count
        ::testing::Values(0,1))); // root

class test_bcast_1 : public test_bcast,
        public ::testing::WithParamInterface<Param_1> {};

UCC_TEST_P(test_bcast_1, multiple)
{
    const ucc_datatype_t       dtype    = std::get<0>(GetParam());
    const ucc_memory_type_t    mem_type = std::get<1>(GetParam());
    const int                  count    = std::get<2>(GetParam());
    const int                  root     = std::get<3>(GetParam());
    std::vector<UccReq>        reqs;
    std::vector<UccCollCtxVec> ctxs;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h       team = UccJob::getStaticTeams()[tid];
        int             size = team->procs.size();
        UccCollCtxVec   ctx;

        if (size == 1 && root > 0) {
            /* skip team size 1 and root > 0, which are invalid */
            continue;
        }

        SET_MEM_TYPE(mem_type);
        set_root(root);

        data_init(size, dtype, count, ctx, false);
        reqs.push_back(UccReq(team, ctx));
        ctxs.push_back(ctx);
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);

    for (auto ctx : ctxs) {
        EXPECT_EQ(true, data_validate(ctx));
        data_fini(ctx);
    }
}

INSTANTIATE_TEST_CASE_P(
    , test_bcast_1,
    ::testing::Combine(
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA,
                          UCC_MEMORY_TYPE_CUDA_MANAGED),
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,65536), // count
        ::testing::Values(0,1))); // root

class test_bcast_alg : public test_bcast
{};

UCC_TEST_F(test_bcast_alg, 2step) {
    int           n_procs = 15;
    ucc_job_env_t env     = {{"UCC_CL_HIER_TUNE", "bcast:@2step:0-inf:inf"},
                             {"UCC_CLS", "all"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 1;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }
    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA_MANAGED)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA_MANAGED);
    }

    for (auto count : {8, 65536}) {
        for (int root = 0; root < n_procs; root++) {
            for (auto m : mt) {
                this->set_root(root);
                SET_MEM_TYPE(m);
                this->data_init(n_procs, UCC_DT_INT8, count, ctxs, false);
                UccReq req(team, ctxs);

                for (auto i = 0; i < repeat; i++) {
                    req.start();
                    req.wait();
                    EXPECT_EQ(true, this->data_validate(ctxs));
                    this->reset(ctxs);
                }
                this->data_fini(ctxs);
            }
        }
    }
}

UCC_TEST_F(test_bcast_alg, two_tree_odd_shift) {
    int           n_procs = 15;
    ucc_job_env_t env     = {{"UCC_TL_UCP_TUNE", "bcast:@two_tree:0-inf:inf"},
                             {"UCC_CLS", "basic"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 1;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }
    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA_MANAGED)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA_MANAGED);
    }

    for (auto count : {8, 65536}) {
        for (int root = 0; root < n_procs; root++) {
            for (auto m : mt) {
                this->set_root(root);
                SET_MEM_TYPE(m);
                this->data_init(n_procs, UCC_DT_INT8, count, ctxs, false);
                UccReq req(team, ctxs);

                for (auto i = 0; i < repeat; i++) {
                    req.start();
                    req.wait();
                    EXPECT_EQ(true, this->data_validate(ctxs));
                    this->reset(ctxs);
                }
                this->data_fini(ctxs);
            }
        }
    }
}

UCC_TEST_F(test_bcast_alg, two_tree_even_mirror) {
    int           n_procs = 16;
    ucc_job_env_t env     = {{"UCC_TL_UCP_TUNE", "bcast:@two_tree:0-inf:inf"},
                             {"UCC_CLS", "basic"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 1;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }
    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA_MANAGED)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA_MANAGED);
    }

    for (auto count : {8, 65536}) {
        for (int root = 0; root < n_procs; root++) {
            for (auto m : mt) {
                this->set_root(root);
                SET_MEM_TYPE(m);
                this->data_init(n_procs, UCC_DT_INT8, count, ctxs, false);
                UccReq req(team, ctxs);

                for (auto i = 0; i < repeat; i++) {
                    req.start();
                    req.wait();
                    EXPECT_EQ(true, this->data_validate(ctxs));
                    this->reset(ctxs);
                }
                this->data_fini(ctxs);
            }
        }
    }
}
