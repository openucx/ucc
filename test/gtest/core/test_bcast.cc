/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, int, ucc_memory_type_t, int, int>;
using Param_1 = std::tuple<int, ucc_memory_type_t, int, int>;

class test_bcast : public UccCollArgs, public ucc::test
{
private:
    int root;
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
            coll->coll_type = UCC_COLL_TYPE_BCAST;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dtype;
            coll->root = root;

            ctxs[r]->rbuf_size = ucc_dt_size(dtype) * count;

            UCC_CHECK(ucc_mc_alloc(&coll->src.info.mc_header,
                                   ctxs[r]->rbuf_size, mem_type));
            coll->src.info.buffer = coll->src.info.mc_header->addr;
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
        }
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (auto r = 0; r < ctxs.size(); r++) {
            gtest_ucc_coll_ctx_t *ctx = ctxs[r];
            ucc_coll_args_t* coll = ctx->args;
            UCC_CHECK(ucc_mc_free(coll->src.info.mc_header, mem_type));
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
        ucc_mc_buffer_header_t *dsts_mc_header;

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

UCC_TEST_P(test_bcast_0, single_host)
{
    const int                 team_id  = std::get<0>(GetParam());
    const ucc_datatype_t      dtype    = (ucc_datatype_t)std::get<1>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<2>(GetParam());
    const int                 count    = std::get<3>(GetParam());
    const int                 root     = std::get<4>(GetParam());
    UccTeam_h                 team     = UccJob::getStaticTeams()[team_id];
    int                       size     = team->procs.size();
    UccCollCtxVec             ctxs;

    set_mem_type(mem_type);
    set_root(root);

    data_init(size, (ucc_datatype_t)dtype, count, ctxs);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

INSTANTIATE_TEST_CASE_P(
    , test_bcast_0,
    ::testing::Combine(
        ::testing::Range(1, UccJob::nStaticTeams), // team_ids
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_UINT32 + 1, 3), // dtype
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,65536), // count
        ::testing::Values(0,1))); // root

class test_bcast_1 : public test_bcast,
        public ::testing::WithParamInterface<Param_1> {};

UCC_TEST_P(test_bcast_1, multiple)
{
    const ucc_datatype_t       dtype    = (ucc_datatype_t)std::get<0>(GetParam());
    const ucc_memory_type_t    mem_type = std::get<1>(GetParam());
    const int                  count    = std::get<2>(GetParam());
    const int                  root     = std::get<3>(GetParam());
    std::vector<UccReq>        reqs;
    std::vector<UccCollCtxVec> ctxs;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h       team = UccJob::getStaticTeams()[tid];
        int             size = team->procs.size();
        UccCollCtxVec   ctx;

        set_mem_type(mem_type);
        set_root(root);

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
    , test_bcast_1,
    ::testing::Combine(
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_UINT32 + 1, 3), // dtype
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(1,3,65536), // count
        ::testing::Values(0,1))); // root
