/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "core/test_mc_reduce.h"
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

// For sliding window allreduce
#include "test_allreduce_sliding_window.h"

#include <array>

template<typename T>
class test_allreduce : public UccCollArgs, public testing::Test {
  public:
    void data_init(int nprocs, ucc_datatype_t dt, size_t count,
                   UccCollCtxVec &ctxs, bool persistent)
    {
        ctxs.resize(nprocs);
        for (int r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->coll_type          = UCC_COLL_TYPE_ALLREDUCE;
            coll->op                 = T::redop;
            coll->global_work_buffer = NULL;

            ctxs[r]->init_buf = ucc_malloc(ucc_dt_size(dt) * count, "init buf");
            EXPECT_NE(ctxs[r]->init_buf, nullptr);
            for (int i = 0; i < count; i++) {
                typename T::type * ptr;
                ptr = (typename T::type *)ctxs[r]->init_buf;
                /* need to limit the init value so that "prod" operation
                   would not grow too large. We have teams up to 16 procs
                   in gtest, this would result in prod ~2**48 */
                /* bFloat16 will be assigned with the floats matching the
                   uint16_t bit pattern*/
                ptr[i] = (typename T::type)((i + r + 1) % 8);
            }

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
                                   ucc_dt_size(dt) * count, mem_type));
            coll->dst.info.buffer = ctxs[r]->dst_mc_header->addr;
            if (TEST_INPLACE == inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                UCC_CHECK(ucc_mc_memcpy(coll->dst.info.buffer, ctxs[r]->init_buf,
                                        ucc_dt_size(dt) * count, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
                coll->src.info.mem_type = UCC_MEMORY_TYPE_UNKNOWN;
                coll->src.info.count    = SIZE_MAX;
                coll->src.info.datatype = (ucc_datatype_t)-1;
            } else {
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                       ucc_dt_size(dt) * count, mem_type));
                coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                                        ucc_dt_size(dt) * count, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
                coll->src.info.mem_type = mem_type;
                coll->src.info.count    = (ucc_count_t)count;
                coll->src.info.datatype = dt;
            }
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count    = (ucc_count_t)count;
            coll->dst.info.datatype = dt;
            if (persistent) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
            }
        }
    }
    void data_fini(UccCollCtxVec ctxs) {
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
    void reset(UccCollCtxVec ctxs)
    {
        for (auto r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t *coll  = ctxs[r]->args;
            size_t           count = coll->dst.info.count;
            ucc_datatype_t   dtype = coll->dst.info.datatype;
            clear_buffer(coll->dst.info.buffer, count * ucc_dt_size(dtype),
                         mem_type, 0);

            if (TEST_INPLACE == inplace) {
                UCC_CHECK(ucc_mc_memcpy(coll->dst.info.buffer,
                                        ctxs[r]->init_buf,
                                        ucc_dt_size(dtype) * count, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            }
        }
    }
    bool data_validate(UccCollCtxVec ctxs)
    {
        size_t count = (ctxs[0])->args->dst.info.count;
        std::vector<typename T::type *> dsts(ctxs.size());

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (typename T::type *) ucc_malloc(count * sizeof(typename T::type), "dsts buf");
                EXPECT_NE(dsts[r], nullptr);
                UCC_CHECK(ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info.buffer,
                                        count * sizeof(typename T::type), UCC_MEMORY_TYPE_HOST,
                                        mem_type));
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (typename T::type *)(ctxs[r]->args->dst.info.buffer);
            }
        }
        for (int i = 0; i < count; i++) {
            typename T::type res =
                    ((typename T::type *)((ctxs[0])->init_buf))[i];
            for (int r = 1; r < ctxs.size(); r++) {
                res = T::do_op(res, ((typename T::type *)((ctxs[r])->init_buf))[i]);
            }
            if (T::redop == UCC_OP_AVG) {
                if (T::dt == UCC_DT_BFLOAT16){
                    float32tobfloat16(bfloat16tofloat32(&res) / (float)ctxs.size(),
                                      &res);
                } else {
                    res = res / (typename T::type)ctxs.size();
                }
            }
            for (int r = 0; r < ctxs.size(); r++) {
                T::assert_equal(res, dsts[r][i]);
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                ucc_free(dsts[r]);
            }
        }
        return true;
    }
};
template<typename T>
class test_allreduce_host : public test_allreduce<T> {};

template<typename T>
class test_allreduce_cuda : public test_allreduce<T> {};

TYPED_TEST_CASE(test_allreduce_host, CollReduceTypeOpsHost);
TYPED_TEST_CASE(test_allreduce_cuda, CollReduceTypeOpsCuda);

#define TEST_DECLARE(_mem_type, _inplace, _repeat, _persistent)                \
    {                                                                          \
        std::array<int, 3> counts{4, 256, 65536};                              \
        for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {                 \
            for (int count : counts) {                                         \
                UccTeam_h     team = UccJob::getStaticTeams()[tid];            \
                int           size = team->procs.size();                       \
                UccCollCtxVec ctxs;                                            \
                SET_MEM_TYPE(_mem_type);                                       \
                this->set_inplace(_inplace);                                   \
                this->data_init(size, TypeParam::dt, count, ctxs, _persistent);\
                UccReq req(team, ctxs);                                        \
                for (auto i = 0; i < _repeat; i++) {                           \
                    req.start();                                               \
                    req.wait();                                                \
                    EXPECT_EQ(true, this->data_validate(ctxs));                \
                    this->reset(ctxs);                                         \
                }                                                              \
                this->data_fini(ctxs);                                         \
            }                                                                  \
        }                                                                      \
    }

TYPED_TEST(test_allreduce_host, single) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE, 1, 0);
}

TYPED_TEST(test_allreduce_host, single_persistent)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE, 3, 1);
}

TYPED_TEST(test_allreduce_host, single_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE, 1, 0);
}

TYPED_TEST(test_allreduce_host, single_persistent_inplace)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE, 3, 1);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_allreduce_cuda, single) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE, 1, 0);
}

TYPED_TEST(test_allreduce_cuda, single_persistent)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE, 3, 1);
}

TYPED_TEST(test_allreduce_cuda, single_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE, 1, 0);
}

TYPED_TEST(test_allreduce_cuda, single_persistent_inplace)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE, 3, 1);
}
TYPED_TEST(test_allreduce_cuda, single_managed) {
    TEST_DECLARE( UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_NO_INPLACE, 1, 0);
}

TYPED_TEST(test_allreduce_cuda, single_persistent_managed)
{
    TEST_DECLARE( UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_NO_INPLACE, 3, 1);
}

TYPED_TEST(test_allreduce_cuda, single_inplace_managed) {
    TEST_DECLARE( UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_INPLACE, 1, 0);
}

TYPED_TEST(test_allreduce_cuda, single_persistent_inplace_managed)
{
    TEST_DECLARE( UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_INPLACE, 3, 1);
}
#endif

#define TEST_DECLARE_MULTIPLE(_mem_type, _inplace)                             \
    {                                                                          \
        std::array<int, 3> counts{4, 256, 65536};                              \
        for (int count : counts) {                                             \
            std::vector<UccReq>        reqs;                                   \
            std::vector<UccCollCtxVec> ctxs;                                   \
            for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {             \
                UccTeam_h     team = UccJob::getStaticTeams()[tid];            \
                int           size = team->procs.size();                       \
                UccCollCtxVec ctx;                                             \
                this->set_inplace(_inplace);                                   \
                SET_MEM_TYPE(_mem_type);                                       \
                this->data_init(size, TypeParam::dt, count, ctx, false);       \
                reqs.push_back(UccReq(team, ctx));                             \
                ctxs.push_back(ctx);                                           \
            }                                                                  \
            UccReq::startall(reqs);                                            \
            UccReq::waitall(reqs);                                             \
            for (auto ctx : ctxs) {                                            \
                EXPECT_EQ(true, this->data_validate(ctx));                     \
                this->data_fini(ctx);                                          \
            }                                                                  \
        }                                                                      \
    }

TYPED_TEST(test_allreduce_host, multiple) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE);
}

TYPED_TEST(test_allreduce_host, multiple_inplace) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_allreduce_cuda, multiple) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE);
}

TYPED_TEST(test_allreduce_cuda, multiple_inplace) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE);
}
TYPED_TEST(test_allreduce_cuda, multiple_managed) {
    TEST_DECLARE_MULTIPLE( UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_NO_INPLACE);
}

TYPED_TEST(test_allreduce_cuda, multiple_inplace_managed) {
    TEST_DECLARE_MULTIPLE( UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_INPLACE);
}
#endif

template<typename T>
class test_allreduce_alg : public test_allreduce<T>
{};

// Expanded type list for allreduce algorithm tests to cover more data types and operations
using test_allreduce_alg_type = ::testing::Types<
    TypeOpPair<UCC_DT_INT32, sum>,
    TypeOpPair<UCC_DT_FLOAT32, sum>,
    TypeOpPair<UCC_DT_INT32, prod>,
    TypeOpPair<UCC_DT_INT32, max>,
    TypeOpPair<UCC_DT_INT32, min>,
    TypeOpPair<UCC_DT_FLOAT64, sum>
>;
TYPED_TEST_CASE(test_allreduce_alg, test_allreduce_alg_type);

TYPED_TEST(test_allreduce_alg, sra_knomial_pipelined) {
    int           n_procs = 15;
    ucc_job_env_t env     = {{"UCC_CL_BASIC_TUNE", "inf"},
                             {"UCC_TL_UCP_TUNE", "allreduce:@sra_knomial:inf"},
                             {"UCC_TL_UCP_ALLREDUCE_SRA_KN_PIPELINE", "thresh=1024:nfrags=11"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 3;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }
    if (UCC_OK == ucc_mc_available( UCC_MEMORY_TYPE_CUDA_MANAGED)) {
        mt.push_back( UCC_MEMORY_TYPE_CUDA_MANAGED);
    }

    for (auto count : {65536, 123567}) {
        for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {
            for (auto m : mt) {
                SET_MEM_TYPE(m);
                this->set_inplace(inplace);
                this->data_init(n_procs, TypeParam::dt, count, ctxs, true);
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

TYPED_TEST(test_allreduce_alg, dbt) {
    int           n_procs = 15;
    ucc_job_env_t env     = {{"UCC_CL_BASIC_TUNE", "inf"},
                             {"UCC_TL_UCP_TUNE", "allreduce:@dbt:inf"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 3;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }
    if (UCC_OK == ucc_mc_available( UCC_MEMORY_TYPE_CUDA_MANAGED)) {
        mt.push_back( UCC_MEMORY_TYPE_CUDA_MANAGED);
    }

    for (auto count : {65536, 123567}) {
        for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {
            for (auto m : mt) {
                SET_MEM_TYPE(m);
                this->set_inplace(inplace);
                this->data_init(n_procs, TypeParam::dt, count, ctxs, true);
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

TYPED_TEST(test_allreduce_alg, rab) {
    int           n_procs = 15;
    ucc_job_env_t env     = {{"UCC_CL_HIER_TUNE", "allreduce:@rab:0-inf:inf"},
                             {"UCC_CLS", "all"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 3;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) { //add cuda_managed for cl hier?
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }

    for (auto count : {8, 65536, 123567}) {
        for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {
            for (auto m : mt) {
                SET_MEM_TYPE(m);
                this->set_inplace(inplace);
                this->data_init(n_procs, TypeParam::dt, count, ctxs, true);
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

TYPED_TEST(test_allreduce_alg, rab_pipelined) {
    int           n_procs = 15;
    ucc_job_env_t env     = {{"UCC_CL_HIER_TUNE", "allreduce:@rab:0-inf:inf"},
                             {"UCC_CL_HIER_ALLREDUCE_RAB_PIPELINE", "thresh=1024:nfrags=11"},
                             {"UCC_CLS", "all"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 3;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) { //add cuda_managed for cl hier?
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }

    for (auto count : {65536, 123567}) {
        for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {
            for (auto m : mt) {
                SET_MEM_TYPE(m);
                this->set_inplace(inplace);
                this->data_init(n_procs, TypeParam::dt, count, ctxs, true);
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

TYPED_TEST(test_allreduce_alg, ring) {
    int           n_procs = 15;
    ucc_job_env_t env     = {{"UCC_CL_BASIC_TUNE", "inf"},
                             {"UCC_TL_UCP_TUNE", "allreduce:0-inf:@ring"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 3;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }
    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA_MANAGED)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA_MANAGED);
    }

    // Test with various data sizes: small, medium, large
    for (auto count : {8, 65536, 123567}) {
        for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {
            for (auto m : mt) {
                SET_MEM_TYPE(m);
                this->set_inplace(inplace);
                this->data_init(n_procs, TypeParam::dt, count, ctxs, true);
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

TYPED_TEST(test_allreduce_alg, ring_edge_cases) {
    // Test with non-power-of-two team sizes and edge cases
    for (auto team_size : {3, 7, 13}) {
        ucc_job_env_t env     = {{"UCC_CL_BASIC_TUNE", "inf"},
                                 {"UCC_TL_UCP_TUNE", "allreduce:0-inf:@ring"}};
        UccJob        job(team_size, UccJob::UCC_JOB_CTX_GLOBAL, env);
        UccTeam_h     team = job.create_team(team_size);
        UccCollCtxVec ctxs;

        for (auto count : {0, 1, 3, 17}) {
            SET_MEM_TYPE(UCC_MEMORY_TYPE_HOST);
            this->set_inplace(TEST_NO_INPLACE);
            this->data_init(team_size, TypeParam::dt, count, ctxs, false);
            UccReq req(team, ctxs);

            req.start();
            req.wait();
            EXPECT_EQ(true, this->data_validate(ctxs));
            this->data_fini(ctxs);
        }
    }
}

TYPED_TEST(test_allreduce_alg, ring_persistent) {
    // Test persistent operation - results should be consistent across multiple calls
    int           n_procs = 8;
    ucc_job_env_t env     = {{"UCC_CL_BASIC_TUNE", "inf"},
                             {"UCC_TL_UCP_TUNE", "allreduce:0-inf:@ring"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    UccCollCtxVec ctxs;

    SET_MEM_TYPE(UCC_MEMORY_TYPE_HOST);
    this->set_inplace(TEST_NO_INPLACE);
    // Use a larger buffer for persistent test
    size_t count = 1024;
    this->data_init(n_procs, TypeParam::dt, count, ctxs, true);
    UccReq req(team, ctxs);
    // Run multiple iterations to verify persistence
    for (int i = 0; i < 5; i++) {
        req.start();
        req.wait();
        EXPECT_EQ(true, this->data_validate(ctxs));
        this->reset(ctxs);
    }
    this->data_fini(ctxs);
}

#ifdef HAVE_UCX
TYPED_TEST(test_allreduce_alg, sliding_window)
{
    int              n_procs = 8;
    ucc_job_env_t    env     = {{"UCC_TL_UCP_TUNE", "allreduce:@sliding_window"},
                                {"UCC_CLS", "all"}};
    UccJob           job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL_ONESIDED, env);
    UccTeam_h        team      = job.create_team(n_procs);
    int              repeat    = 3;
    test_ucp_info_t *ucp_infos = NULL;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};
    ucs_status_t     ucs_status = UCS_OK;
    UccCollCtxVec    ctxs;

    if (UCC_OK == ucc_mc_available(
                      UCC_MEMORY_TYPE_CUDA)) { //add cuda_managed for cl hier?
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }

    for (auto count : {65536, 123567}) {
        for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {
            for (auto m : mt) {
                SET_MEM_TYPE(m);
                this->set_inplace(inplace);
                this->data_init(n_procs, TypeParam::dt, count, ctxs, true);

                // set args->global_work_buffer on each ctx
                ucs_status = setup_gwbi(n_procs, ctxs, &ucp_infos, inplace == TEST_INPLACE);
                if (ucs_status != UCS_OK) {
                    free_gwbi(n_procs, ctxs, ucp_infos, inplace == TEST_INPLACE);
                    this->data_fini(ctxs);
                    if (ucs_status == UCS_ERR_UNSUPPORTED) {
                        GTEST_SKIP() << "Exported memory key not supported";
                    } else {
                        GTEST_FAIL() << ucs_status_string(ucs_status);
                    }
                }

                for (auto i = 0; i < repeat; i++) {
                    this->reset(ctxs);
                }

                free_gwbi(n_procs, ctxs, ucp_infos, inplace == TEST_INPLACE);
                ucp_infos = NULL;
                this->data_fini(ctxs);
            }
        }
    }
}
#endif

template <typename T>
class test_allreduce_avg_order : public test_allreduce<T> {
};

using test_allreduce_avg_order_type = ::testing::Types<
    TypeOpPair<UCC_DT_FLOAT32, avg>, TypeOpPair<UCC_DT_FLOAT64, avg>,
    TypeOpPair<UCC_DT_FLOAT128, avg>, TypeOpPair<UCC_DT_FLOAT32_COMPLEX, avg>,
    TypeOpPair<UCC_DT_FLOAT64_COMPLEX, avg>,
    TypeOpPair<UCC_DT_FLOAT128_COMPLEX, avg>, TypeOpPair<UCC_DT_BFLOAT16, avg>>;

TYPED_TEST_CASE(test_allreduce_avg_order, test_allreduce_avg_order_type);

TYPED_TEST(test_allreduce_avg_order, avg_post_op)
{
    int           n_procs = 15;
    ucc_job_env_t env     = {{"UCC_TL_UCP_REDUCE_AVG_PRE_OP", "0"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 3;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }
    if (UCC_OK == ucc_mc_available( UCC_MEMORY_TYPE_CUDA_MANAGED)) {
        mt.push_back( UCC_MEMORY_TYPE_CUDA_MANAGED);
    }

    for (auto count : {4, 256, 65536}) {
        for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {
            for (auto m : mt) {
                CHECK_TYPE_OP_SKIP(TypeParam::dt, TypeParam::redop, m);
                SET_MEM_TYPE(m);
                this->set_inplace(inplace);
                this->data_init(n_procs, TypeParam::dt, count, ctxs, true);
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
