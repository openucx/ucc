/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "test_mc_reduce.h"
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

#include <array>

template<typename T>
class test_allreduce : public UccCollArgs, public testing::Test {
  public:
    void data_init(int nprocs, ucc_datatype_t dt, size_t count,
                   UccCollCtxVec &ctxs)
    {
        ctxs.resize(nprocs);
        for (int r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS;
            coll->coll_type = UCC_COLL_TYPE_ALLREDUCE;
            coll->reduce.predefined_op = T::redop;

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->init_buf, ucc_dt_size(dt) * count,
                                   UCC_MEMORY_TYPE_HOST));
            for (int i = 0; i < count; i++) {
                typename T::type * ptr;
                ptr = (typename T::type *)ctxs[r]->init_buf;
                ptr[i] = (typename T::type)(2 * i + r + 1);
            }

            UCC_CHECK(ucc_mc_alloc(&coll->dst.info.buffer,
                      ucc_dt_size(dt) * count, mem_type));
            if (TEST_INPLACE == inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                UCC_CHECK(ucc_mc_memcpy(coll->dst.info.buffer, ctxs[r]->init_buf,
                                        ucc_dt_size(dt) * count, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            } else {
                UCC_CHECK(ucc_mc_alloc(&coll->src.info.buffer,
                          ucc_dt_size(dt) * count, mem_type));
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                                        ucc_dt_size(dt) * count, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            }
            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dt;

            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count   = (ucc_count_t)count;
            coll->dst.info.datatype = dt;
        }
    }
    void data_fini(UccCollCtxVec ctxs) {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            if (coll->src.info.buffer) { /* no inplace */
                UCC_CHECK(ucc_mc_free(coll->src.info.buffer, mem_type));
            }
            UCC_CHECK(ucc_mc_free(coll->dst.info.buffer, mem_type));
            UCC_CHECK(ucc_mc_free(ctx->init_buf, UCC_MEMORY_TYPE_HOST));
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }
    void data_validate(UccCollCtxVec ctxs)
    {
        size_t count = (ctxs[0])->args->src.info.count;
        std::vector<typename T::type *> dsts(ctxs.size());

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                UCC_CHECK(ucc_mc_alloc((void**)&dsts[r],
                                       count * sizeof(typename T::type),
                                       UCC_MEMORY_TYPE_HOST));
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
            for (int r = 0; r < ctxs.size(); r++) {
                T::assert_equal(res, dsts[r][i]);
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                UCC_CHECK(ucc_mc_free((void*)dsts[r], UCC_MEMORY_TYPE_HOST));
            }
        }
        return;
    }
};

TYPED_TEST_CASE(test_allreduce, ReductionTypesOps);

#define TEST_DECLARE(_mem_type, _inplace)                                      \
{                                                                              \
    std::array<int,3> counts {1,2,4};                                          \
    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {                     \
        for (int count : counts) {                                             \
            UccTeam_h team = UccJob::getStaticTeams()[tid];                    \
            int       size = team->procs.size();                               \
            UccCollCtxVec ctxs;                                                \
            this->set_mem_type(_mem_type);                                     \
            this->set_inplace(_inplace);                                       \
            this->data_init(size, TypeParam::dt, count, ctxs);                 \
            UccReq    req(team, ctxs);                                         \
            req.start();                                                       \
            req.wait();                                                        \
            this->data_validate(ctxs);                                         \
            this->data_fini(ctxs);                                             \
        }                                                                      \
    }                                                                          \
}

TYPED_TEST(test_allreduce, single_host) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE);
}

TYPED_TEST(test_allreduce, single_host_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_allreduce, single_cuda) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE);
}

TYPED_TEST(test_allreduce, single_cuda_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE);
}
#endif

/* TODO: enable parallel teams once it is supported */
/*
TYPED_TEST(test_allreduce, multi) {
    const int count = 2;

    std::vector<UccReq> reqs;
    std::vector<UccCollArgsVec> args;
    for (auto &team : UccJob::getStaticTeams()) {
        UccCollArgsVec arg = this->data_init(team->procs.size(),
                                       TypeParam::dt, count);
        args.push_back(arg);
        reqs.push_back(UccReq(team, arg));
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);
    for (auto arg : args) {
        this->data_validate(arg);
        this->data_fini(arg);
    }
}
*/
