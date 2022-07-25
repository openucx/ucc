/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "core/test_mc_reduce.h"
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

#include <array>

template <typename T>
class test_reduce_scatter : public UccCollArgs, public testing::Test {
  public:
    virtual void TestBody(){};
    void data_init(int nprocs, ucc_datatype_t dt, size_t count,
                   UccCollCtxVec &ctxs, bool persistent)
    {
        size_t rcount;
        ctxs.resize(nprocs);
        if (count < nprocs) {
            count = nprocs;
        }
        count  = count - (count % nprocs);
        rcount = count / nprocs;
        if (TEST_INPLACE == inplace) {
            rcount = count;
        }

        for (int r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll =
                (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] =
                (gtest_ucc_coll_ctx_t *)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask      = 0;
            coll->coll_type = UCC_COLL_TYPE_REDUCE_SCATTER;
            coll->op        = T::redop;

            ctxs[r]->init_buf = ucc_malloc(ucc_dt_size(dt) * count, "init buf");
            EXPECT_NE(ctxs[r]->init_buf, nullptr);
            for (int i = 0; i < count; i++) {
                typename T::type *ptr;
                ptr = (typename T::type *)ctxs[r]->init_buf;
                /* need to limit the init value so that "prod" operation
                   would not grow too large. We have teams up to 16 procs
                   in gtest, this would result in prod ~2**48 */
                ptr[i] = (typename T::type)((i + r + 1) % 8);
            }

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
                                   ucc_dt_size(dt) * rcount, mem_type));
            coll->dst.info.buffer = ctxs[r]->dst_mc_header->addr;
            coll->src.info.buffer = NULL;
            if (TEST_INPLACE == inplace) {
                coll->mask |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                UCC_CHECK(ucc_mc_memcpy(
                    coll->dst.info.buffer, ctxs[r]->init_buf,
                    ucc_dt_size(dt) * rcount, mem_type, UCC_MEMORY_TYPE_HOST));
                coll->src.info.mem_type = UCC_MEMORY_TYPE_UNKNOWN;
                coll->src.info.count    = SIZE_MAX;
                coll->src.info.datatype = (ucc_datatype_t)-1;
            } else {
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                       ucc_dt_size(dt) * count, mem_type));
                coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(
                    coll->src.info.buffer, ctxs[r]->init_buf,
                    ucc_dt_size(dt) * count, mem_type, UCC_MEMORY_TYPE_HOST));
                coll->src.info.mem_type = mem_type;
                coll->src.info.count    = (ucc_count_t)count;
                coll->src.info.datatype = dt;
            }
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count    = (ucc_count_t)rcount;
            coll->dst.info.datatype = dt;
            if (persistent) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
            }
        }
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t *ctx : ctxs) {
            ucc_coll_args_t *coll = ctx->args;
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
        size_t            total_count, rcount, offset;
        typename T::type *dst, *dst_p;

        if (TEST_INPLACE != inplace) {
            total_count = (ctxs[0])->args->src.info.count;
            rcount      = (ctxs[0])->args->dst.info.count;
        } else {
            total_count = (ctxs[0])->args->dst.info.count;
            rcount      = total_count / ctxs.size();
        }

        ucc_assert(rcount * ctxs.size() == total_count);

        dst = (typename T::type *)ucc_malloc(
            total_count * sizeof(typename T::type), "dst buf");
        dst_p = dst;
        for (int r = 0; r < ctxs.size(); r++) {
            offset = (TEST_INPLACE == inplace)
                         ? rcount * r * sizeof(typename T::type)
                         : 0;
            UCC_CHECK(ucc_mc_memcpy(
                dst_p, PTR_OFFSET(ctxs[r]->args->dst.info.buffer, offset),
                rcount * sizeof(typename T::type), UCC_MEMORY_TYPE_HOST,
                mem_type));
            dst_p += rcount;
        }

        for (int i = 0; i < total_count; i++) {
            typename T::type res =
                ((typename T::type *)((ctxs[0])->init_buf))[i];
            for (int r = 1; r < ctxs.size(); r++) {
                res = T::do_op(res,
                               ((typename T::type *)((ctxs[r])->init_buf))[i]);
            }
            if (T::redop == UCC_OP_AVG) {
                res = res / (typename T::type)ctxs.size();
            }
            T::assert_equal(res, dst[i]);
        }
        ucc_free(dst);

        return true;
    }
};

template<typename T>
class test_reduce_scatter_host : public test_reduce_scatter<T> {};

template<typename T>
class test_reduce_scatter_cuda : public test_reduce_scatter<T> {};

TYPED_TEST_CASE(test_reduce_scatter_host, CollReduceTypeOpsHost);
TYPED_TEST_CASE(test_reduce_scatter_cuda, CollReduceTypeOpsCuda);

#define TEST_DECLARE(_mem_type, _inplace, _repeat, _persistent)                \
    {                                                                          \
        std::array<int, 1> counts{123};                                        \
        CHECK_TYPE_OP_SKIP(TypeParam::dt, TypeParam::redop, _mem_type);        \
        for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {                 \
            for (int count : counts) {                                         \
                UccTeam_h     team = UccJob::getStaticTeams()[tid];            \
                int           size = team->procs.size();                       \
                UccCollCtxVec ctxs;                                            \
                SET_MEM_TYPE(_mem_type);                                       \
                this->set_inplace(_inplace);                                   \
                this->data_init(size, TypeParam::dt, count, ctxs, _persistent);\
                UccReq req(team, ctxs);                                        \
                CHECK_REQ_NOT_SUPPORTED_SKIP(req, this->data_fini(ctxs));      \
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

TYPED_TEST(test_reduce_scatter_host, single)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_scatter_host, single_persistent)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE, 3, 1);
}

TYPED_TEST(test_reduce_scatter_host, single_inplace)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_scatter_host, single_persistent_inplace)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE, 3, 1);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_reduce_scatter_cuda, single)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_scatter_cuda, single_persistent)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE, 3, 1);
}

TYPED_TEST(test_reduce_scatter_cuda, single_inplace)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_scatter_cuda, single_persistent_inplace)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE, 3, 1);
}
#endif

#define TEST_DECLARE_MULTIPLE(_mem_type, _inplace)                             \
    {                                                                          \
        std::array<int, 3> counts{4, 256, 65536};                              \
        CHECK_TYPE_OP_SKIP(TypeParam::dt, TypeParam::redop, _mem_type);        \
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
                ctxs.push_back(ctx);                                           \
                reqs.push_back(UccReq(team, ctx));                             \
                CHECK_REQ_NOT_SUPPORTED_SKIP(reqs.back(),                      \
                                             DATA_FINI_ALL(this, ctxs));       \
            }                                                                  \
            UccReq::startall(reqs);                                            \
            UccReq::waitall(reqs);                                             \
            for (auto ctx : ctxs) {                                            \
                EXPECT_EQ(true, this->data_validate(ctx));                     \
            }                                                                  \
            DATA_FINI_ALL(this, ctxs);                                         \
        }                                                                      \
    }

TYPED_TEST(test_reduce_scatter_host, multiple)
{
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE);
}

TYPED_TEST(test_reduce_scatter_host, multiple_inplace)
{
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_reduce_scatter_cuda, multiple)
{
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE);
}

TYPED_TEST(test_reduce_scatter_cuda, multiple_inplace)
{
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE);
}
#endif

class test_reduce_scatter_alg
    : public ucc::test,
      public ::testing::WithParamInterface<std::string> {
};

UCC_TEST_P(test_reduce_scatter_alg, ring)
{
    test_reduce_scatter<TypeOpPair<UCC_DT_INT32, sum>> rs_test;
    int                                                n_procs = 15;
    std::string                                        bidir   = GetParam();
    ucc_job_env_t                  env = {{"UCC_CL_BASIC_TUNE", "inf"},
                         {"UCC_TL_UCP_TUNE", "reduce_scatter:@ring:inf"},
                         {"REDUCE_SCATTER_RING_BIDIRECTIONAL",
                          bidir == "bidirectional" ? "y" : "n"}};
    UccJob        job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h     team   = job.create_team(n_procs);
    int           repeat = 3;
    UccCollCtxVec ctxs;
    std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};

    if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {
        mt.push_back(UCC_MEMORY_TYPE_CUDA);
    }

    for (auto count : {65536, 123567}) {
        for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {
            for (auto m : mt) {
                rs_test.set_mem_type(m);
                rs_test.set_inplace(inplace);
                rs_test.data_init(n_procs, UCC_DT_INT32, count, ctxs, true);
                UccReq req(team, ctxs);

                for (auto i = 0; i < repeat; i++) {
                    req.start();
                    req.wait();
                    EXPECT_EQ(true, rs_test.data_validate(ctxs));
                    rs_test.reset(ctxs);
                }
                rs_test.data_fini(ctxs);
            }
        }
    }
}
INSTANTIATE_TEST_CASE_P(, test_reduce_scatter_alg,
                        ::testing::Values("bidirectional", "unidirectional"));
