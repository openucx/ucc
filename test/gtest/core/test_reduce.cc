/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "test_mc_reduce.h"
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

#include <array>

template<typename T>
class test_reduce : public UccCollArgs, public testing::Test {
  private:
    int root = 0;
  public:
    void data_init(int nprocs, ucc_datatype_t dt, size_t count,
                   UccCollCtxVec &ctxs)
    {
        ctxs.resize(nprocs);
        for (int r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1,
                          sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask = UCC_COLL_ARGS_FIELD_PREDEFINED_REDUCTIONS;
            coll->coll_type = UCC_COLL_TYPE_REDUCE;
            coll->reduce.predefined_op = T::redop;
            coll->root = root;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dt;

            ctxs[r]->init_buf = ucc_malloc(ucc_dt_size(dt) * count,
                                                        "init buf");
            EXPECT_NE(ctxs[r]->init_buf, nullptr);
            for (int i = 0; i < count; i++) {
                typename T::type * ptr;
                ptr = (typename T::type *)ctxs[r]->init_buf;
                /* need to limit the init value so that "prod" operation
                   would not grow too large. We have teams up to 16 procs
                   in gtest, this would result in prod ~2**48 */
                ptr[i] = (typename T::type)((i + r + 1) % 8);
            }

            if (r == root) {
			    coll->dst.info.mem_type = mem_type;
			    coll->dst.info.count   = (ucc_count_t)count;
			    coll->dst.info.datatype = dt;
			    UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
			                           ucc_dt_size(dt) * count, mem_type));
			    coll->dst.info.buffer = ctxs[r]->dst_mc_header->addr;
			    if (inplace) {
			        UCC_CHECK(ucc_mc_memcpy(coll->dst.info.buffer,
			    		  ctxs[r]->init_buf, ucc_dt_size(dt) * count, mem_type,
			              UCC_MEMORY_TYPE_HOST));
			    }
            }
            if (inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;

            }
            if (r != root || !inplace) {
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                       ucc_dt_size(dt) * count, mem_type));
                coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer,
                                        ctxs[r]->init_buf,
                                        ucc_dt_size(dt) * count, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
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
            T::assert_equal(res, dsts[i]);
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            ucc_free(dsts);
        }
        return true;
    }
};

TYPED_TEST_CASE(test_reduce, ReductionTypesOps);

#define TEST_DECLARE(_mem_type, _inplace, _repeat)                             \
    {                                                                          \
        std::array<int, 3> counts{4, 256, 65536};                              \
        for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {                 \
            for (int count : counts) {                                         \
                UccTeam_h     team = UccJob::getStaticTeams()[tid];            \
                int           size = team->procs.size();                       \
                UccCollCtxVec ctxs;                                            \
                this->set_mem_type(_mem_type);                                 \
                this->set_inplace(_inplace);                                   \
                this->data_init(size, TypeParam::dt, count, ctxs);             \
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

TYPED_TEST(test_reduce, single_host) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE, 1);
}

TYPED_TEST(test_reduce, single_host_persistent)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE, 3);
}

TYPED_TEST(test_reduce, single_host_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE, 1);
}

TYPED_TEST(test_reduce, single_host_persistent_inplace)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE, 3);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_reduce, single_cuda) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE, 1);
}

TYPED_TEST(test_reduce, single_cuda_persistent)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE, 3);
}

TYPED_TEST(test_reduce, single_cuda_inplace) {
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE, 1);
}

TYPED_TEST(test_reduce, single_cuda_persistent_inplace)
{
    TEST_DECLARE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE, 3);
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
                this->set_mem_type(_mem_type);                                 \
                this->data_init(size, TypeParam::dt, count, ctx);              \
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

TYPED_TEST(test_reduce, multiple) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_HOST, TEST_NO_INPLACE);
}

TYPED_TEST(test_reduce, multiple_inplace) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_HOST, TEST_INPLACE);
}

#ifdef HAVE_CUDA
TYPED_TEST(test_reduce, multiple_cuda) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA, TEST_NO_INPLACE);
}

TYPED_TEST(test_reduce, multiple_cuda_inplace) {
    TEST_DECLARE_MULTIPLE(UCC_MEMORY_TYPE_CUDA, TEST_INPLACE);
}
#endif
