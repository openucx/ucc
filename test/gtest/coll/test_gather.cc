/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, ucc_datatype_t, ucc_memory_type_t, int, int,
                           gtest_ucc_inplace_t>;
using Param_1 = std::tuple<ucc_datatype_t, ucc_memory_type_t, int, int,
                           gtest_ucc_inplace_t>;

class test_gather : public UccCollArgs, public ucc::test {
  private:
    int root;

  public:
    void data_init(int nprocs, ucc_datatype_t dtype, size_t single_rank_count,
                   UccCollCtxVec &ctxs, bool persistent)
    {
        ctxs.resize(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll =
                (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));
            ctxs[r] =
                (gtest_ucc_coll_ctx_t *)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask              = 0;
            coll->flags             = 0;
            coll->coll_type         = UCC_COLL_TYPE_GATHER;
            coll->root              = root;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count    = (ucc_count_t)single_rank_count;
            coll->src.info.datatype = dtype;

            ctxs[r]->init_buf =
                ucc_malloc(ucc_dt_size(dtype) * single_rank_count, "init buf");
            EXPECT_NE(ctxs[r]->init_buf, nullptr);
            for (int i = 0; i < single_rank_count * ucc_dt_size(dtype); i++) {
                uint8_t *ptr = (uint8_t *)ctxs[r]->init_buf;
                ptr[i]       = ((i + r) % 256);
            }

            if (r == root) {
                coll->dst.info.mem_type = mem_type;
                coll->dst.info.count = (ucc_count_t)single_rank_count * nprocs;
                coll->dst.info.datatype = dtype;
                ctxs[r]->rbuf_size =
                    ucc_dt_size(dtype) * single_rank_count * nprocs;
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
                                       ctxs[r]->rbuf_size, mem_type));
                coll->dst.info.buffer = ctxs[r]->dst_mc_header->addr;
                if (inplace) {
                    UCC_CHECK(ucc_mc_memcpy(
                        (void *)((ptrdiff_t)coll->dst.info.buffer +
                                 r * single_rank_count * ucc_dt_size(dtype)),
                        ctxs[r]->init_buf,
                        ucc_dt_size(dtype) * single_rank_count, mem_type,
                        UCC_MEMORY_TYPE_HOST));
                }
            }
            if (r != root || !inplace) {
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                       ucc_dt_size(dtype) * single_rank_count,
                                       mem_type));
                coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer,
                                        ctxs[r]->init_buf,
                                        ucc_dt_size(dtype) * single_rank_count,
                                        mem_type, UCC_MEMORY_TYPE_HOST));
            }
            if (inplace) {
                coll->mask |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
            }
            if (persistent) {
                coll->mask |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
            }
        }
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (auto r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t *coll = ctxs[r]->args;
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
        ucc_coll_args_t *coll              = ctxs[root]->args;
        size_t           single_rank_count = coll->dst.info.count / ctxs.size();
        ucc_datatype_t   dtype             = coll->dst.info.datatype;
        clear_buffer(coll->dst.info.buffer,
                     single_rank_count * ucc_dt_size(dtype) * ctxs.size(),
                     mem_type, 0);
        if (TEST_INPLACE == inplace) {
            UCC_CHECK(ucc_mc_memcpy(
                (void *)((ptrdiff_t)coll->dst.info.buffer +
                         root * single_rank_count * ucc_dt_size(dtype)),
                ctxs[root]->init_buf, ucc_dt_size(dtype) * single_rank_count,
                mem_type, UCC_MEMORY_TYPE_HOST));
        }
    }
    bool data_validate(UccCollCtxVec ctxs)
    {
        bool     ret               = true;
        int      root              = ctxs[0]->args->root;
        size_t   count             = (ctxs[root])->args->dst.info.count;
        size_t   single_rank_count = count / ctxs.size();
        size_t   dt_size = ucc_dt_size(ctxs[0]->args->src.info.datatype);
        uint8_t *dsts;

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            dsts = (uint8_t *)ucc_malloc(count * dt_size, "dsts buf");
            EXPECT_NE(dsts, nullptr);
            UCC_CHECK(ucc_mc_memcpy(dsts, ctxs[root]->args->dst.info.buffer,
                                    count * dt_size,
                                    UCC_MEMORY_TYPE_HOST, mem_type));
        } else {
            dsts = (uint8_t *)ctxs[root]->args->dst.info.buffer;
        }
        for (int r = 0; r < ctxs.size(); r++) {
            for (int i = 0; i < single_rank_count * dt_size; i++) {
                if ((uint8_t)((i + r) % 256) !=
                    dsts[(r * single_rank_count * dt_size + i)]) {
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

class test_gather_0 : public test_gather,
                      public ::testing::WithParamInterface<Param_0> {
};

UCC_TEST_P(test_gather_0, single)
{
    const int                 team_id  = std::get<0>(GetParam());
    const ucc_datatype_t      dtype    = std::get<1>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<2>(GetParam());
    const int                 count    = std::get<3>(GetParam());
    const int                 root     = std::get<4>(GetParam());
    const gtest_ucc_inplace_t inplace  = std::get<5>(GetParam());
    UccTeam_h                 team     = UccJob::getStaticTeams()[team_id];
    int                       size     = team->procs.size();
    UccCollCtxVec             ctxs;

    set_inplace(inplace);
    SET_MEM_TYPE(mem_type);
    set_root(root);

    data_init(size, dtype, count, ctxs, false);
    UccReq req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

UCC_TEST_P(test_gather_0, single_persistent)
{
    const int                 team_id  = std::get<0>(GetParam());
    const ucc_datatype_t      dtype    = std::get<1>(GetParam());
    const ucc_memory_type_t   mem_type = std::get<2>(GetParam());
    const int                 count    = std::get<3>(GetParam());
    const int                 root     = std::get<4>(GetParam());
    const gtest_ucc_inplace_t inplace  = std::get<5>(GetParam());
    UccTeam_h                 team     = UccJob::getStaticTeams()[team_id];
    int                       size     = team->procs.size();
    const int                 n_calls  = 3;
    UccCollCtxVec             ctxs;

    set_inplace(inplace);
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
    , test_gather_0,
    ::testing::Combine(::testing::Range(1, UccJob::nStaticTeams), // team_ids
                       PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
                       ::testing::Values(UCC_MEMORY_TYPE_HOST,
                                         UCC_MEMORY_TYPE_CUDA),
#else
                       ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
                       ::testing::Values(1, 3, 8192), // count
                       ::testing::Values(0, 1),       // root
                       ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE)));

class test_gather_1 : public test_gather,
                      public ::testing::WithParamInterface<Param_1> {
};

UCC_TEST_P(test_gather_1, multiple_host)
{
    const ucc_datatype_t       dtype    = std::get<0>(GetParam());
    const ucc_memory_type_t    mem_type = std::get<1>(GetParam());
    const int                  count    = std::get<2>(GetParam());
    const int                  root     = std::get<3>(GetParam());
    const gtest_ucc_inplace_t  inplace  = std::get<4>(GetParam());
    std::vector<UccReq>        reqs;
    std::vector<UccCollCtxVec> ctxs;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h     team = UccJob::getStaticTeams()[tid];
        int           size = team->procs.size();
        UccCollCtxVec ctx;

        if (size == 1 && root > 0) {
            /* skip team size 1 and root > 0, which are invalid */
            continue;
        }

        this->set_inplace(inplace);
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
    , test_gather_1,
    ::testing::Combine(PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
                       ::testing::Values(UCC_MEMORY_TYPE_HOST,
                                         UCC_MEMORY_TYPE_CUDA),
#else
                       ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
                       ::testing::Values(1, 3, 8192), // count
                       ::testing::Values(0, 1),       // root
                       ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE)));
