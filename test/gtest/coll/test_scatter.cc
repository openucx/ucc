/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, ucc_datatype_t, ucc_memory_type_t, int, int,
                           gtest_ucc_inplace_t>;
using Param_1 = std::tuple<ucc_datatype_t, ucc_memory_type_t, int, int,
                           gtest_ucc_inplace_t>;

class test_scatter : public UccCollArgs, public ucc::test {
  private:
    int root;

  public:
    void data_init(int nprocs, ucc_datatype_t dtype, size_t single_rank_count,
                   UccCollCtxVec &ctxs, bool persistent)
    {
        ucc_coll_args_t *coll;
        ucc_count_t count;

        ctxs.resize(nprocs);
        count = single_rank_count * nprocs;

        for (auto r = 0; r < nprocs; r++) {
            coll = (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));
            ctxs[r] =
                (gtest_ucc_coll_ctx_t *)calloc(1,
                                               sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask              = 0;
            coll->flags             = 0;
            coll->coll_type         = UCC_COLL_TYPE_SCATTER;
            coll->root              = root;
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count    = (ucc_count_t)single_rank_count;
            coll->dst.info.datatype = dtype;

            if (r == root) {
                ctxs[r]->init_buf =
                    ucc_malloc(ucc_dt_size(dtype) * count,
                               "init buf");
                EXPECT_NE(ctxs[r]->init_buf, nullptr);
                uint8_t *sbuf = (uint8_t*)ctxs[r]->init_buf;
                for (int p = 0; p < nprocs; p++) {
                    for (int i = 0; i < ucc_dt_size(dtype) * single_rank_count; i++) {
                        sbuf[(p * ucc_dt_size(dtype) * single_rank_count + i)] = (uint8_t)((i+p) % 256);
                    }
                }

                coll->src.info.mem_type = mem_type;
                coll->src.info.count = (ucc_count_t)count;
                coll->src.info.datatype = dtype;
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                       ucc_dt_size(dtype) * count, mem_type));
                coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer,
                          ctxs[r]->init_buf, ucc_dt_size(dtype) * count,
                          mem_type, UCC_MEMORY_TYPE_HOST));
            }
            if (r != root || !inplace) {
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
                                       ucc_dt_size(dtype) * single_rank_count,
                                       mem_type));
                coll->dst.info.buffer = ctxs[r]->dst_mc_header->addr;
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
                UCC_CHECK(ucc_mc_free(ctxs[r]->src_mc_header));
            }
            if (r != root || !inplace) {
                UCC_CHECK(ucc_mc_free(ctxs[r]->dst_mc_header));
            }
            ucc_free(ctxs[r]->init_buf);
            free(coll);
            free(ctxs[r]);
        }
        ctxs.clear();
    }
    void reset(UccCollCtxVec ctxs)
    {
        ucc_coll_args_t *coll;
        size_t           single_rank_count = (ctxs[0])->args->dst.info.count;
        ucc_datatype_t   dtype = (ctxs[0])->args->dst.info.datatype;

        for (auto r = 0; r < ctxs.size(); r++) {
            coll = ctxs[r]->args;
            if (r != root || !inplace) {
                clear_buffer(coll->dst.info.buffer,
                    single_rank_count * ucc_dt_size(dtype),
                    mem_type, 0);
            }
        }
    }
    bool data_validate(UccCollCtxVec ctxs)
    {
        bool     ret               = true;
        int      root              = ctxs[0]->args->root;
        size_t   single_rank_count = (ctxs[0])->args->dst.info.count;
        size_t   dt_size = ucc_dt_size((ctxs[0])->args->dst.info.datatype);
        uint8_t *dsts = nullptr;

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            dsts = (uint8_t *)ucc_malloc(single_rank_count * dt_size, "dsts buf");
            EXPECT_NE(dsts, nullptr);
        }
        for (auto r = 0; r < ctxs.size(); r++) {
            if (r == root && inplace) {
                continue;
            }
            if (UCC_MEMORY_TYPE_HOST != mem_type) {
                UCC_CHECK(ucc_mc_memcpy(dsts, ctxs[r]->args->dst.info.buffer,
                                        single_rank_count * dt_size,
                                        UCC_MEMORY_TYPE_HOST, mem_type));
            } else {
                dsts = (uint8_t *)ctxs[r]->args->dst.info.buffer;
            }

            for (int i = 0; i < single_rank_count * dt_size; i++) {
                if ((uint8_t)((i + r) % 256) !=
                    dsts[i]) {
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

class test_scatter_0 : public test_scatter,
                      public ::testing::WithParamInterface<Param_0> {
};

UCC_TEST_P(test_scatter_0, single)
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
    GTEST_SKIP(); //scatter not implemented as stand alone in TL/UCP

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

UCC_TEST_P(test_scatter_0, single_persistent)
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
    GTEST_SKIP(); //scatter not implemented as stand alone in TL/UCP

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
    , test_scatter_0,
    ::testing::Combine(::testing::Range(1, UccJob::nStaticTeams), // team_ids
                       PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
                       ::testing::Values(UCC_MEMORY_TYPE_HOST,
                                         UCC_MEMORY_TYPE_CUDA,
                                         UCC_MEMORY_TYPE_CUDA_MANAGED),
#else
                       ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
                       ::testing::Values(1, 3, 8192), // count
                       ::testing::Values(0, 1),       // root
                       ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE)));

class test_scatter_1 : public test_scatter,
                      public ::testing::WithParamInterface<Param_1> {
};

UCC_TEST_P(test_scatter_1, multiple_host)
{
    const ucc_datatype_t       dtype    = std::get<0>(GetParam());
    const ucc_memory_type_t    mem_type = std::get<1>(GetParam());
    const int                  count    = std::get<2>(GetParam());
    const int                  root     = std::get<3>(GetParam());
    const gtest_ucc_inplace_t  inplace  = std::get<4>(GetParam());
    std::vector<UccReq>        reqs;
    std::vector<UccCollCtxVec> ctxs;
    GTEST_SKIP(); //scatter not implemented as stand alone in TL/UCP

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
    , test_scatter_1,
    ::testing::Combine(PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
                       ::testing::Values(UCC_MEMORY_TYPE_HOST,
                                         UCC_MEMORY_TYPE_CUDA,
                                         UCC_MEMORY_TYPE_CUDA_MANAGED),
#else
                       ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
                       ::testing::Values(1, 3, 8192), // count
                       ::testing::Values(0, 1),       // root
                       ::testing::Values(TEST_INPLACE, TEST_NO_INPLACE)));
