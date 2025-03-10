/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "core/test_mc_reduce.h"
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

#include <array>

template<typename T>
class test_reduce : public UccCollArgs, public testing::Test {
  private:
    int root = 0;
  public:
    void data_init(int nprocs, ucc_datatype_t dt, size_t count,
                   UccCollCtxVec &ctxs, bool persistent)
    {
        ctxs.resize(nprocs);
        for (int r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r]           = (gtest_ucc_coll_ctx_t*)calloc(1,
                                sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args     = coll;
            ctxs[r]->init_buf = ucc_malloc(ucc_dt_size(dt) * count,
                                                        "init buf");
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

            coll->coll_type = UCC_COLL_TYPE_REDUCE;
            coll->op        = T::redop;
            coll->root      = root;
            if (r != root || !inplace) {
                coll->src.info.mem_type = mem_type;
                coll->src.info.count    = (ucc_count_t)count;
                coll->src.info.datatype = dt;
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                       ucc_dt_size(dt) * count, mem_type));
                coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer,
                                        ctxs[r]->init_buf,
                                        ucc_dt_size(dt) * count, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            }
            if (r == root) {
                coll->dst.info.mem_type = mem_type;
                coll->dst.info.count = (ucc_count_t)count;
                coll->dst.info.datatype = dt;
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
                                       ucc_dt_size(dt) * count, mem_type));
                coll->dst.info.buffer = ctxs[r]->dst_mc_header->addr;
                if (inplace) {
                    UCC_CHECK(ucc_mc_memcpy(coll->dst.info.buffer,
                              ctxs[r]->init_buf, ucc_dt_size(dt) * count,
                              mem_type, UCC_MEMORY_TYPE_HOST));
                }
            }
            if (inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
            }
            if (persistent) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
            }
        }
    }
    void data_fini(UccCollCtxVec ctxs) {
    	for (auto r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t* coll = ctxs[r]->args;
            if (r == root) {
                UCC_CHECK(ucc_mc_free(ctxs[r]->dst_mc_header));
            }
            if (r != root || !inplace) {
            	UCC_CHECK(ucc_mc_free(ctxs[r]->src_mc_header));
            }
            ucc_free(ctxs[r]->init_buf);
            free(coll);
            free(ctxs[r]);
        }
        ctxs.clear();
    }
    void reset(UccCollCtxVec ctxs)
    {
        ucc_coll_args_t *coll  = ctxs[root]->args;
        size_t           count = coll->dst.info.count;
        ucc_datatype_t   dtype = coll->dst.info.datatype;
        clear_buffer(coll->dst.info.buffer, count * ucc_dt_size(dtype),
                     mem_type, 0);
		if (TEST_INPLACE == inplace) {
			UCC_CHECK(ucc_mc_memcpy(coll->dst.info.buffer,
                  ctxs[root]->init_buf,
                  ucc_dt_size(dtype) * count, mem_type, UCC_MEMORY_TYPE_HOST));
		}
    }
    bool data_validate(UccCollCtxVec ctxs)
    {
        size_t count = (ctxs[0])->args->src.info.count;
        typename T::type * dsts;

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            dsts = (typename T::type *)
                    ucc_malloc(count * sizeof(typename T::type), "dsts buf");
            EXPECT_NE(dsts, nullptr);
            UCC_CHECK(ucc_mc_memcpy(dsts, ctxs[root]->args->dst.info.buffer,
                                    count * sizeof(typename T::type),
                                    UCC_MEMORY_TYPE_HOST, mem_type));
        } else {
            dsts = (typename T::type *)ctxs[root]->args->dst.info.buffer;
        }
        for (int i = 0; i < count; i++) {
            typename T::type res =
                    ((typename T::type *)((ctxs[0])->init_buf))[i];
            for (int r = 1; r < ctxs.size(); r++) {
                res = T::do_op(res,
                              ((typename T::type *)((ctxs[r])->init_buf))[i]);
            }
            if (T::redop == UCC_OP_AVG) {
                if (T::dt == UCC_DT_BFLOAT16){
                    float32tobfloat16(bfloat16tofloat32(&res) / (float)ctxs.size(),
                    &res);
                } else {
                    res = res / (typename T::type)ctxs.size();
                }
            }
            T::assert_equal(res, dsts[i]);
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            ucc_free(dsts);
        }
        return true;
    }
};

template<typename T>
class test_reduce_host : public test_reduce<T> {};

template<typename T>
class test_reduce_cuda : public test_reduce<T> {};

TYPED_TEST_CASE(test_reduce_host, CollReduceTypeOpsHost);
TYPED_TEST_CASE(test_reduce_cuda, CollReduceTypeOpsCuda);

#define TEST_DECLARE(_mem_type, _inplace, _repeat, _persistent)                \
    {                                                                          \
        std::array<int, 3> counts{4, 256, 65536};                              \
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

TYPED_TEST(test_reduce_host, single) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_host, single_persistent) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE, 3, 1);
}

TYPED_TEST(test_reduce_host, single_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_host, single_persistent_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE, 3, 1);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_reduce_cuda, single) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_cuda, single_persistent) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE, 3, 1);
}
TYPED_TEST(test_reduce_cuda, single_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_cuda, single_persistent_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE, 3, 1);
}
TYPED_TEST(test_reduce_cuda, single_managed) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_NO_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_cuda, single_persistent_managed) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_NO_INPLACE, 3, 1);
}
TYPED_TEST(test_reduce_cuda, single_inplace_managed) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_INPLACE, 1, 0);
}

TYPED_TEST(test_reduce_cuda, single_persistent_inplace_managed) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_INPLACE, 3, 1);
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
                reqs.push_back(UccReq(team, ctx));                             \
                CHECK_REQ_NOT_SUPPORTED_SKIP(reqs.back(),                      \
                                             DATA_FINI_ALL(this, ctxs));       \
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

TYPED_TEST(test_reduce_host, multiple) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE);
}

TYPED_TEST(test_reduce_host, multiple_inplace) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_reduce_cuda, multiple) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE);
}
TYPED_TEST(test_reduce_cuda, multiple_inplace) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE);
}
TYPED_TEST(test_reduce_cuda, multiple_managed) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_NO_INPLACE);
}
TYPED_TEST(test_reduce_cuda, multiple_inplace_managed) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA_MANAGED, TEST_INPLACE);
}
#endif

template <typename T> class test_reduce_avg_order : public test_reduce<T> {
};

template <typename T> class test_reduce_dbt : public test_reduce<T> {
};

template <typename T> class test_reduce_2step : public test_reduce<T> {
};

template <typename T> class test_reduce_srg : public test_reduce<T> {
};

#define TEST_DECLARE_WITH_ENV(_env, _n_procs, _persistent)                     \
    {                                                                          \
        UccJob        job(_n_procs, UccJob::UCC_JOB_CTX_GLOBAL, _env);         \
        UccTeam_h     team   = job.create_team(_n_procs);                      \
        int           repeat = _persistent ? 3 : 1;                            \
        UccCollCtxVec ctxs;                                                    \
        std::vector<ucc_memory_type_t> mt = {UCC_MEMORY_TYPE_HOST};            \
        if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA)) {                \
            mt.push_back(UCC_MEMORY_TYPE_CUDA);                                \
        }                                                                      \
        if (UCC_OK == ucc_mc_available(UCC_MEMORY_TYPE_CUDA_MANAGED)) {        \
            mt.push_back(UCC_MEMORY_TYPE_CUDA_MANAGED);                        \
        }                                                                      \
        for (auto count : {5, 256, 65536}) {                                   \
            for (auto inplace : {TEST_NO_INPLACE, TEST_INPLACE}) {             \
                for (auto m : mt) {                                            \
                    CHECK_TYPE_OP_SKIP(TypeParam::dt, TypeParam::redop, m);    \
                    SET_MEM_TYPE(m);                                           \
                    this->set_inplace(inplace);                                \
                    this->data_init(_n_procs, TypeParam::dt, count, ctxs,      \
                                    _persistent);                              \
                    UccReq req(team, ctxs);                                    \
                    CHECK_REQ_NOT_SUPPORTED_SKIP(req, this->data_fini(ctxs));  \
                    for (auto i = 0; i < repeat; i++) {                        \
                        req.start();                                           \
                        req.wait();                                            \
                        EXPECT_EQ(true, this->data_validate(ctxs));            \
                        this->reset(ctxs);                                     \
                    }                                                          \
                    this->data_fini(ctxs);                                     \
                }                                                              \
            }                                                                  \
        }                                                                      \
    }

TYPED_TEST_CASE(test_reduce_avg_order, CollReduceTypeOpsAvg);
TYPED_TEST_CASE(test_reduce_dbt, CollReduceTypeOpsHost);
TYPED_TEST_CASE(test_reduce_2step, CollReduceTypeOpsHost);
TYPED_TEST_CASE(test_reduce_srg, CollReduceTypeOpsHost);

ucc_job_env_t post_op_env      = {{"UCC_TL_UCP_REDUCE_AVG_PRE_OP", "0"}};
ucc_job_env_t reduce_dbt_env   = {{"UCC_TL_UCP_TUNE", "reduce:@dbt:0-inf:inf"},
                                  {"UCC_CLS", "basic"}};
ucc_job_env_t reduce_2step_env = {{"UCC_CL_HIER_TUNE", "reduce:@2step:0-inf:inf"},
                                  {"UCC_CLS", "all"}};
ucc_job_env_t reduce_srg_env   = {{"UCC_TL_UCP_TUNE", "reduce:@srg:0-inf:inf"},
                                  {"UCC_CLS", "basic"}};
TYPED_TEST(test_reduce_avg_order, avg_post_op) {
    TEST_DECLARE_WITH_ENV(post_op_env, 15, true);
}

TYPED_TEST(test_reduce_dbt, reduce_dbt_shift) {
    TEST_DECLARE_WITH_ENV(reduce_dbt_env, 15, true);
}

TYPED_TEST(test_reduce_dbt, reduce_dbt_mirror) {
    TEST_DECLARE_WITH_ENV(reduce_dbt_env, 16, true);
}

TYPED_TEST(test_reduce_2step, 2step) {
    TEST_DECLARE_WITH_ENV(reduce_2step_env, 16, false);
}

TYPED_TEST(test_reduce_srg, srg) {
    TEST_DECLARE_WITH_ENV(reduce_srg_env, 15, false);
}
