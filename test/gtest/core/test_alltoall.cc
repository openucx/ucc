/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, ucc_datatype_t, ucc_memory_type_t, gtest_ucc_inplace_t, int>;
using Param_1 = std::tuple<ucc_datatype_t, ucc_memory_type_t, gtest_ucc_inplace_t, int>;

class test_alltoall : public UccCollArgs, public ucc::test
{
public:
    void data_init(int nprocs, ucc_datatype_t dtype, size_t single_rank_count,
                   UccCollCtxVec &ctxs)
    {
        ctxs.resize(nprocs);
        for (auto i = 0; i < nprocs; i++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[i] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[i]->args = coll;

            coll->mask = 0;
            coll->coll_type = UCC_COLL_TYPE_ALLTOALL;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count    = (ucc_count_t)single_rank_count * nprocs;
            coll->src.info.datatype = dtype;
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count    = (ucc_count_t)single_rank_count * nprocs;
            coll->dst.info.datatype = dtype;

            ctxs[i]->init_buf = ucc_malloc(
                ucc_dt_size(dtype) * single_rank_count * nprocs, "init buf");
            EXPECT_NE(ctxs[i]->init_buf, nullptr);
           for (int r = 0; r < nprocs; r++) {
                size_t rank_size = ucc_dt_size(dtype) * single_rank_count;
                alltoallx_init_buf(r, i,
                                   (uint8_t*)ctxs[i]->init_buf + r * rank_size,
                                   rank_size);
            }

            UCC_CHECK(ucc_mc_alloc(
                &ctxs[i]->dst_mc_header,
                ucc_dt_size(dtype) * single_rank_count * nprocs, mem_type));
            coll->dst.info.buffer = ctxs[i]->dst_mc_header->addr;
            if (TEST_INPLACE == inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                UCC_CHECK(ucc_mc_memcpy(coll->dst.info.buffer, ctxs[i]->init_buf,
                                        ucc_dt_size(dtype) * single_rank_count * nprocs,
                                        mem_type, UCC_MEMORY_TYPE_HOST));
            } else {
                UCC_CHECK(ucc_mc_alloc(
                    &ctxs[i]->src_mc_header,
                    ucc_dt_size(dtype) * single_rank_count * nprocs, mem_type));
                coll->src.info.buffer = ctxs[i]->src_mc_header->addr;
                UCC_CHECK(
                    ucc_mc_memcpy(coll->src.info.buffer, ctxs[i]->init_buf,
                                  ucc_dt_size(dtype) * single_rank_count * nprocs,
                                  mem_type, UCC_MEMORY_TYPE_HOST));
            }
        }
    }
    void data_init(int nprocs, UccJob *job, ucc_datatype_t dtype,
                   size_t single_rank_count, UccCollCtxVec &ctxs)
    {
        void *sbuf;
        void *rbuf;
        void *work_buf;

        ctxs.resize(nprocs);
        for (auto i = 0; i < nprocs; i++) {
            ucc_coll_args_t *coll =
                (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));

            ctxs[i] =
                (gtest_ucc_coll_ctx_t *)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[i]->args = coll;

            work_buf = job->procs[i]->onesided_buf[0];
            sbuf     = job->procs[i]->onesided_buf[1];
            rbuf     = job->procs[i]->onesided_buf[2];

            coll->mask = UCC_COLL_ARGS_FIELD_FLAGS |
                         UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;
            coll->coll_type          = UCC_COLL_TYPE_ALLTOALL;
            coll->src.info.mem_type  = mem_type;
            coll->src.info.count     = (ucc_count_t)single_rank_count * nprocs;
            coll->src.info.datatype  = dtype;
            coll->dst.info.mem_type  = mem_type;
            coll->dst.info.count     = (ucc_count_t)single_rank_count * nprocs;
            coll->dst.info.datatype  = dtype;
            coll->src.info.buffer    = sbuf;
            coll->dst.info.buffer    = rbuf;
            coll->flags              = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS,
            coll->global_work_buffer = work_buf;

            ctxs[i]->init_buf = ucc_malloc(
                ucc_dt_size(dtype) * single_rank_count * nprocs, "init buf");
            EXPECT_NE(ctxs[i]->init_buf, nullptr);
            for (int r = 0; r < nprocs; r++) {
                size_t rank_size = ucc_dt_size(dtype) * single_rank_count;
                alltoallx_init_buf(r, i,
                                   (uint8_t *)ctxs[i]->init_buf + r * rank_size,
                                   rank_size);
            }

            if (TEST_INPLACE == inplace) {
                coll->mask |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                UCC_CHECK(ucc_mc_memcpy(
                    coll->dst.info.buffer, ctxs[i]->init_buf,
                    ucc_dt_size(dtype) * single_rank_count * nprocs, mem_type,
                    UCC_MEMORY_TYPE_HOST));
            } else {
                UCC_CHECK(ucc_mc_memcpy(
                    coll->src.info.buffer, ctxs[i]->init_buf,
                    ucc_dt_size(dtype) * single_rank_count * nprocs, mem_type,
                    UCC_MEMORY_TYPE_HOST));
            }
        }
    }
    void reset(UccCollCtxVec ctxs)
    {
        for (auto r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t *coll  = ctxs[r]->args;
            size_t single_rank_count = coll->dst.info.count / ctxs.size();
            ucc_datatype_t   dtype = coll->dst.info.datatype;
            clear_buffer(coll->dst.info.buffer,
                         single_rank_count * ucc_dt_size(dtype), mem_type, 0);
            if (TEST_INPLACE == inplace) {
                UCC_CHECK(ucc_mc_memcpy(
                    (void *)(ptrdiff_t)coll->dst.info.buffer, ctxs[r]->init_buf,
                    ucc_dt_size(dtype) * single_rank_count, mem_type,
                    UCC_MEMORY_TYPE_HOST));
            }
        }
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            if (coll->src.info.buffer) { /* no inplace */
                UCC_CHECK(ucc_mc_free(ctx->src_mc_header));
            }
            UCC_CHECK(ucc_mc_free(ctx->dst_mc_header));
            ucc_free(ctx->init_buf);
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }
    void data_fini_onesided(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            ucc_free(ctx->init_buf);
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }   
    bool data_validate(UccCollCtxVec ctxs)
    {
        bool                   ret = true;
        std::vector<uint8_t *> dsts(ctxs.size());

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                size_t buf_size =
                    ucc_dt_size(ctxs[r]->args->dst.info.datatype) *
                    (size_t)ctxs[r]->args->dst.info.count;
                dsts[r] = (uint8_t *) ucc_malloc(buf_size, "dsts buf");
                EXPECT_NE(dsts[r], nullptr);
                UCC_CHECK(ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info.buffer,
                                        buf_size, UCC_MEMORY_TYPE_HOST, mem_type));
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *)(ctxs[r]->args->dst.info.buffer);
            }
        }
        for (int r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t* coll = ctxs[r]->args;

            for (int i = 0; i < ctxs.size(); i++) {
                size_t rank_size = ucc_dt_size(coll->dst.info.datatype) *
                                   (size_t)(coll->dst.info.count / ctxs.size());
                if (0 != alltoallx_validate_buf(i, r, (uint8_t*)dsts[r] +
                                                rank_size * i, rank_size)) {
                    ret = false;
                    break;
                }
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                ucc_free(dsts[r]);
            }
        }
        return ret;
    }
};

class test_alltoall_0 : public test_alltoall,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoall_0, single)
{
    const int            team_id  = std::get<0>(GetParam());
    const ucc_datatype_t dtype    = std::get<1>(GetParam());
    ucc_memory_type_t    mem_type = std::get<2>(GetParam());
    gtest_ucc_inplace_t  inplace  = std::get<3>(GetParam());
    const int            count    = std::get<4>(GetParam());
    UccTeam_h            team     = UccJob::getStaticTeams()[team_id];
    int                  size     = team->procs.size();
    UccCollCtxVec        ctxs;

    this->set_inplace(inplace);
    this->set_mem_type(mem_type);

    data_init(size, dtype, count, ctxs);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

UCC_TEST_P(test_alltoall_0, single_onesided)
{
    const int            team_id        = std::get<0>(GetParam());
    const ucc_datatype_t dtype          = std::get<1>(GetParam());
    ucc_memory_type_t    mem_type       = std::get<2>(GetParam());
    gtest_ucc_inplace_t  inplace        = std::get<3>(GetParam());
    const int            count          = std::get<4>(GetParam());
    UccTeam_h            reference_team = UccJob::getStaticTeams()[team_id];
    int                  size           = reference_team->procs.size();
    ucc_job_env_t        env = {{"UCC_TL_UCP_TUNE", "alltoall:0-inf:@1"}};
    UccJob               job(size, UccJob::UCC_JOB_CTX_GLOBAL_ONESIDED, env);
    UccTeam_h            team = job.create_team(size, false, true, true);
    UccCollCtxVec        ctxs;

    this->set_inplace(inplace);
    this->set_mem_type(mem_type);

    data_init(size, &job, dtype, count, ctxs);
    UccReq req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini_onesided(ctxs);
}

UCC_TEST_P(test_alltoall_0, single_persistent)
{
    const int            team_id  = std::get<0>(GetParam());
    const ucc_datatype_t dtype    = std::get<1>(GetParam());
    ucc_memory_type_t    mem_type = std::get<2>(GetParam());
    gtest_ucc_inplace_t  inplace  = std::get<3>(GetParam());
    const int            count    = std::get<4>(GetParam());
    UccTeam_h            team     = UccJob::getStaticTeams()[team_id];
    int                  size     = team->procs.size();
    const int            n_calls  = 3;
    UccCollCtxVec        ctxs;

    this->set_inplace(inplace);
    this->set_mem_type(mem_type);

    data_init(size, dtype, count, ctxs);
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
    , test_alltoall_0,
    ::testing::Combine(
        ::testing::Range(1, UccJob::nStaticTeams), // team_ids
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(/*TEST_INPLACE,*/ TEST_NO_INPLACE), // inplace
        ::testing::Values(1,3))); // count

class test_alltoall_1 : public test_alltoall,
        public ::testing::WithParamInterface<Param_1> {};

UCC_TEST_P(test_alltoall_1, multiple)
{
    const ucc_datatype_t       dtype    = std::get<0>(GetParam());
    ucc_memory_type_t          mem_type = std::get<1>(GetParam());
    gtest_ucc_inplace_t        inplace  = std::get<2>(GetParam());
    const int                  count    = std::get<3>(GetParam());
    std::vector<UccReq>        reqs;
    std::vector<UccCollCtxVec> ctxs;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h       team = UccJob::getStaticTeams()[tid];
        int             size = team->procs.size();
        UccCollCtxVec   ctx;

        this->set_inplace(inplace);
        this->set_mem_type(mem_type);

        data_init(size, dtype, count, ctx);
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
    , test_alltoall_1,
    ::testing::Combine(
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(/*TEST_INPLACE,*/ TEST_NO_INPLACE), // inplace
        ::testing::Values(1,3,8192))); // count
