/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, int, ucc_memory_type_t, int, gtest_ucc_inplace_t>;
using Param_1 = std::tuple<int, ucc_memory_type_t, int, gtest_ucc_inplace_t>;

class test_allgather : public UccCollArgs, public ucc::test
{
public:
    void data_init(int nprocs, ucc_datatype_t dtype, size_t count,
                   UccCollCtxVec &ctxs)
    {
        ctxs.resize(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));
            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask = 0;
            coll->flags = 0;
            coll->coll_type = UCC_COLL_TYPE_ALLGATHER;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dtype;
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count   = (ucc_count_t)count;
            coll->dst.info.datatype = dtype;

            ctxs[r]->init_buf = ucc_malloc(ucc_dt_size(dtype) * count, "init buf");
            EXPECT_NE(ctxs[r]->init_buf, nullptr);
            uint8_t *sbuf = (uint8_t*)ctxs[r]->init_buf;
            for (int i = 0; i < ucc_dt_size(dtype) * count; i++) {
                sbuf[i] = r;
            }

            ctxs[r]->rbuf_size = ucc_dt_size(dtype) * count * nprocs;
            UCC_CHECK(ucc_mc_alloc(&coll->dst.info.mc_header,
                                   ctxs[r]->rbuf_size, mem_type));
            coll->dst.info.buffer = coll->dst.info.mc_header->addr;
            if (TEST_INPLACE == inplace) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                UCC_CHECK(ucc_mc_memcpy((void*)((ptrdiff_t)coll->dst.info.buffer +
                                        r * count * ucc_dt_size(dtype)),
                                        ctxs[r]->init_buf, ucc_dt_size(dtype) * count,
                                        mem_type, UCC_MEMORY_TYPE_HOST));
            } else {
                UCC_CHECK(ucc_mc_alloc(&coll->src.info.mc_header,
                                       ucc_dt_size(dtype) * count, mem_type));
                coll->src.info.buffer = coll->src.info.mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                                        ucc_dt_size(dtype) * count, mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            }
        }
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            if (coll->src.info.buffer) { /* no inplace */
                UCC_CHECK(ucc_mc_free(coll->src.info.mc_header, mem_type));
            }
            UCC_CHECK(ucc_mc_free(coll->dst.info.buffer, mem_type));
            ucc_free(ctx->init_buf);
            UCC_CHECK(ucc_mc_free(coll->dst.info.mc_header, mem_type));
            UCC_CHECK(ucc_mc_free(ctx->mc_header, UCC_MEMORY_TYPE_HOST));
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }
    bool data_validate(UccCollCtxVec ctxs)
    {
        bool                   ret = true;
        std::vector<uint8_t *> dsts(ctxs.size());
        std::vector<ucc_mc_buffer_header_t *> dsts_mc_headers(ctxs.size());
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *) ucc_malloc(ctxs[r]->rbuf_size, "dsts buf");
                EXPECT_NE(dsts[r], nullptr);
                UCC_CHECK(ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info.buffer,
                                        ctxs[r]->rbuf_size, UCC_MEMORY_TYPE_HOST,
                                        mem_type));
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *)(ctxs[r]->args->dst.info.buffer);
            }
        }
        for (int i = 0; i < ctxs.size(); i++) {
            uint8_t *rbuf = dsts[i];
            for (int r = 0; r < ctxs.size(); r++) {
                size_t rank_size = ucc_dt_size((ctxs[r])->args->src.info.datatype) *
                        (ctxs[r])->args->src.info.count;
                for (int j = 0; j < rank_size; j++) {
                    if (r != rbuf[r*rank_size + j]) {
                        ret = false;
                        break;
                    }
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

class test_allgather_0 : public test_allgather,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_allgather_0, single_host)
{
    const int                 team_id  = std::get<0>(GetParam());
    const ucc_datatype_t      dtype    = (ucc_datatype_t)std::get<1>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<2>(GetParam());
    const int                 count    = std::get<3>(GetParam());
    const gtest_ucc_inplace_t inplace  = std::get<4>(GetParam());
    UccTeam_h                 team     = UccJob::getStaticTeams()[team_id];
    int                       size     = team->procs.size();
    UccCollCtxVec             ctxs;

    set_inplace(inplace);
    set_mem_type(mem_type);

    data_init(size, dtype, count, ctxs);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

INSTANTIATE_TEST_CASE_P(
    , test_allgather_0,
    ::testing::Combine(
        ::testing::Range(1, UccJob::nStaticTeams), // team_ids
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1, 4), // dtype
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,8192), // count
        ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE)));  // inplace


class test_allgather_1 : public test_allgather,
        public ::testing::WithParamInterface<Param_1> {};

UCC_TEST_P(test_allgather_1, multiple_host)
{
    const ucc_datatype_t      dtype    = (ucc_datatype_t)std::get<0>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<1>(GetParam());
    const int                 count    = std::get<2>(GetParam());
    const gtest_ucc_inplace_t inplace  = std::get<3>(GetParam());
    std::vector<UccReq>        reqs;
    std::vector<UccCollCtxVec> ctxs;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h       team = UccJob::getStaticTeams()[tid];
        int             size = team->procs.size();
        UccCollCtxVec   ctx;

        this->set_inplace(inplace);
        this->set_mem_type(mem_type);

        data_init(size, (ucc_datatype_t)dtype, count, ctx);
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
    , test_allgather_1,
    ::testing::Combine(
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1, 4), // dtype
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,8192), // count
        ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE)));  // inplace
