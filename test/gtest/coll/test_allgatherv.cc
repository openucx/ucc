/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, ucc_datatype_t, ucc_memory_type_t, int, gtest_ucc_inplace_t, bool>;
using Param_1 = std::tuple<ucc_datatype_t, ucc_memory_type_t, int, gtest_ucc_inplace_t, bool>;
using Param_2 = std::tuple<ucc_datatype_t, ucc_memory_type_t, int, gtest_ucc_inplace_t, std::string, bool>;

size_t noncontig_padding = 1; // # elements worth of space in between each rank's contribution to the dst buf

class test_allgatherv : public UccCollArgs, public ucc::test
{
public:
    void  data_init(int nprocs, ucc_datatype_t dtype, size_t count,
                    UccCollCtxVec &ctxs, bool persistent) {
        ctxs.resize(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            int *counts;
            int *displs;
            size_t my_count = (nprocs - r) * count;
            size_t disp_counter = 0;
            ucc_coll_args_t *coll = (ucc_coll_args_t*)calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            counts = (int*)malloc(sizeof(int) * nprocs);
            displs = (int*)malloc(sizeof(int) * nprocs);

            if (is_contig) {
                for (int i = 0; i < nprocs; i++) {
                    counts[i] = (nprocs - i) * count;
                    displs[i] = disp_counter;
                    disp_counter += counts[i];
                }
                coll->flags = UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
            } else {
                for (int i = 0; i < nprocs; i++) {
                    counts[i] = (nprocs - i) * count;
                    displs[i] = disp_counter;
                    disp_counter += counts[i] + noncontig_padding; // Add noncontig_padding elemnts of space between the bufs
                }
            }
            coll->mask = UCC_COLL_ARGS_FIELD_FLAGS;
            coll->coll_type = UCC_COLL_TYPE_ALLGATHERV;

            coll->src.info.mem_type = mem_type;
            coll->src.info.count = my_count;
            coll->src.info.datatype = dtype;

            coll->dst.info_v.mem_type = mem_type;
            coll->dst.info_v.counts   = (ucc_count_t*)counts;
            coll->dst.info_v.displacements = (ucc_aint_t*)displs;
            coll->dst.info_v.datatype = dtype;

            ctxs[r]->init_buf = ucc_malloc(ucc_dt_size(dtype) * my_count, "init buf");
            EXPECT_NE(ctxs[r]->init_buf, nullptr);
            for (int i = 0; i < (ucc_dt_size(dtype) * my_count); i++) {
                uint8_t *sbuf = (uint8_t*)ctxs[r]->init_buf;
                sbuf[i] = r;
            }

            ctxs[r]->rbuf_size = ucc_dt_size(dtype) * disp_counter;
            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header, ctxs[r]->rbuf_size,
                                   mem_type));
            coll->dst.info_v.buffer = ctxs[r]->dst_mc_header->addr;
            if (TEST_INPLACE == inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                UCC_CHECK(ucc_mc_memcpy((void*)((ptrdiff_t)coll->dst.info_v.buffer +
                                        displs[r] * ucc_dt_size(dtype)),
                                        ctxs[r]->init_buf, ucc_dt_size(dtype) * my_count,
                                        mem_type, UCC_MEMORY_TYPE_HOST));
            } else {
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                       ucc_dt_size(dtype) * my_count,
                                       mem_type));
                coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                                        ucc_dt_size(dtype) * my_count, mem_type,
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
            ucc_coll_args_t *coll     = ctxs[r]->args;
            size_t           my_count = coll->src.info.count;
            ucc_datatype_t   dtype    = coll->dst.info_v.datatype;
            int *            displs   = (int *)coll->dst.info_v.displacements;

            clear_buffer(coll->dst.info_v.buffer, ctxs[r]->rbuf_size, mem_type,
                         0);
            if (TEST_INPLACE == inplace) {
                UCC_CHECK(ucc_mc_memcpy(
                    (void *)((ptrdiff_t)coll->dst.info_v.buffer +
                             displs[r] * ucc_dt_size(dtype)),
                    ctxs[r]->init_buf, ucc_dt_size(dtype) * my_count, mem_type,
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
            free(coll->dst.info_v.displacements);
            free(coll->dst.info_v.counts);
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
                dsts[r] = (uint8_t *) ucc_malloc(ctxs[r]->rbuf_size, "ctxs buf");
                EXPECT_NE(dsts[r], nullptr);
                UCC_CHECK(ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info_v.buffer,
                                        ctxs[r]->rbuf_size, UCC_MEMORY_TYPE_HOST,
                                        mem_type));
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *)(ctxs[r]->args->dst.info_v.buffer);
            }
        }

        for (int i = 0; i < ctxs.size(); i++) {
            size_t rank_size = 0;
            uint8_t *rbuf = dsts[i];
            int is_contig = UCC_COLL_IS_DST_CONTIG(ctxs[i]->args);
            for (int r = 0; r < ctxs.size(); r++) {
                rbuf += rank_size;
                rank_size = ucc_dt_size((ctxs[r])->args->src.info.datatype) *
                        (ctxs[r])->args->src.info.count;
                for (int j = 0; j < rank_size; j++) {
                    if (r != rbuf[j]) {
                        ret = false;
                        break;
                    }
                }
                if (!is_contig) {
                    rbuf += noncontig_padding * ucc_dt_size((ctxs[r])->args->src.info.datatype);
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

class test_allgatherv_0 : public test_allgatherv,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_allgatherv_0, single)
{
    const int                 team_id  = std::get<0>(GetParam());
    const ucc_datatype_t      dtype    = std::get<1>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<2>(GetParam());
    const int                 count    = std::get<3>(GetParam());
    const gtest_ucc_inplace_t inplace  = std::get<4>(GetParam());
    const bool                contig   = std::get<5>(GetParam());
    UccTeam_h                 team     = UccJob::getStaticTeams()[team_id];
    int                       size     = team->procs.size();
    UccCollCtxVec             ctxs;

    set_inplace(inplace);
    set_contig(contig);
    SET_MEM_TYPE(mem_type);

    data_init(size, dtype, count, ctxs, false);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));;
    data_fini(ctxs);
}

UCC_TEST_P(test_allgatherv_0, single_persistent)
{
    const int                 team_id = std::get<0>(GetParam());
    const ucc_datatype_t      dtype   = std::get<1>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<2>(GetParam());
    const int                 count    = std::get<3>(GetParam());
    const gtest_ucc_inplace_t inplace  = std::get<4>(GetParam());
    const bool                contig   = std::get<5>(GetParam());
    UccTeam_h                 team     = UccJob::getStaticTeams()[team_id];
    int                       size     = team->procs.size();
    const int                 n_calls  = 3;
    UccCollCtxVec             ctxs;

    set_inplace(inplace);
    set_contig(contig);
    SET_MEM_TYPE(mem_type);

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
    , test_allgatherv_0,
    ::testing::Combine(
        ::testing::Range(1, UccJob::nStaticTeams), // team_ids
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA,
                          UCC_MEMORY_TYPE_CUDA_MANAGED), // mem type
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,8192), // count
        ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE), // inplace
        ::testing::Bool() // contig dst buf displacements
        )); 

class test_allgatherv_1 : public test_allgatherv,
        public ::testing::WithParamInterface<Param_1> {};

UCC_TEST_P(test_allgatherv_1, multiple)
{
    const ucc_datatype_t       dtype    = std::get<0>(GetParam());
    const ucc_memory_type_t    mem_type = std::get<1>(GetParam());
    const int                  count    = std::get<2>(GetParam());
    const gtest_ucc_inplace_t  inplace  = std::get<3>(GetParam());
    const bool                 contig   = std::get<4>(GetParam());
    std::vector<UccReq>        reqs;
    std::vector<UccCollCtxVec> ctxs;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h       team = UccJob::getStaticTeams()[tid];
        int             size = team->procs.size();
        UccCollCtxVec   ctx;

        this->set_inplace(inplace);
        this->set_contig(contig);
        SET_MEM_TYPE(mem_type);

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
    , test_allgatherv_1,
    ::testing::Combine(
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA,
                          UCC_MEMORY_TYPE_CUDA_MANAGED),
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,8192), // count
        ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE),
        ::testing::Bool()) // dst buf contig
    );

class test_allgatherv_alg : public test_allgatherv,
        public ::testing::WithParamInterface<Param_2> {};

UCC_TEST_P(test_allgatherv_alg, alg)
{
    const ucc_datatype_t      dtype    = std::get<0>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<1>(GetParam());
    const int                 count    = std::get<2>(GetParam());
    const gtest_ucc_inplace_t inplace  = std::get<3>(GetParam());
    const bool                contig   = std::get<5>(GetParam());
    int                       n_procs  = 5;
    char                      tune[32];

    sprintf(tune, "allgatherv:@%s:inf", std::get<4>(GetParam()).c_str());
    ucc_job_env_t env     = {{"UCC_CL_BASIC_TUNE", "inf"},
                             {"UCC_TL_UCP_TUNE", tune}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team    = job.create_team(n_procs);
    UccCollCtxVec ctxs;

    set_inplace(inplace);
    set_contig(contig);
    SET_MEM_TYPE(mem_type);

    data_init(n_procs, dtype, count, ctxs, false);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

INSTANTIATE_TEST_CASE_P(
    , test_allgatherv_alg,
    ::testing::Combine(
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA,
                          UCC_MEMORY_TYPE_CUDA_MANAGED),
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,8192), // count
        ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE),
        ::testing::Values("knomial", "ring"),
        ::testing::Bool()), // dst buf contig
        [](const testing::TestParamInfo<test_allgatherv_alg::ParamType>& info) {
            std::string name;
            name += ucc_datatype_str(std::get<0>(info.param));
            name += std::string("_") + std::string(ucc_mem_type_str(std::get<1>(info.param)));
            name += std::string("_count_")+std::to_string(std::get<2>(info.param));
            name += std::string("_inplace_")+std::to_string(std::get<3>(info.param));
            name += std::string("_contig_")+std::to_string(std::get<5>(info.param));
            name += std::string("_")+std::get<4>(info.param);
            return name;
        }
    );
