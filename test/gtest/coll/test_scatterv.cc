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

class test_scatterv : public UccCollArgs, public ucc::test {
  private:
    int root;

  public:
    void data_init(int nprocs, ucc_datatype_t dtype, size_t count,
                   UccCollCtxVec &ctxs, bool persistent)
    {
        ucc_coll_args_t *coll;
        int             *counts, *displs;
        size_t           my_count, all_counts;

        ctxs.resize(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            coll = (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));
            my_count = (nprocs - r) * count;
            ctxs[r] =
                (gtest_ucc_coll_ctx_t *)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->mask              = 0;
            coll->flags             = 0;
            coll->coll_type         = UCC_COLL_TYPE_SCATTERV;
            coll->root              = root;
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count    = (ucc_count_t)my_count;
            coll->dst.info.datatype = dtype;

            if (r == root) {
                all_counts = 0;
                counts = (int*)malloc(sizeof(int) * nprocs);
                ASSERT_NE(counts, nullptr);
                displs = (int*)malloc(sizeof(int) * nprocs);
                ASSERT_NE(displs, nullptr);

                for (int i = 0; i < nprocs; i++) {
                    counts[i] = (nprocs - i) * count;
                    displs[i] = all_counts;
                    all_counts += counts[i];
                }
                
                ctxs[r]->init_buf =
                    ucc_malloc(ucc_dt_size(dtype) * all_counts, "init buf");
                ASSERT_NE(ctxs[r]->init_buf, nullptr);
                uint8_t *sbuf = (uint8_t*)ctxs[r]->init_buf;
                for (int p = 0; p < nprocs; p++) {
                    for (int i = 0; i < ucc_dt_size(dtype) * counts[p]; i++) {
                        sbuf[(displs[p] * ucc_dt_size(dtype) + i)] =
                            (uint8_t)((i + p) % 256);
                    }
                }

                coll->src.info_v.mem_type      = mem_type;
                coll->src.info_v.counts        = (ucc_count_t *)counts;
                coll->src.info_v.displacements = (ucc_aint_t *)displs;
                coll->src.info_v.datatype      = dtype;

                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                       ucc_dt_size(dtype) * all_counts,
                                       mem_type));
                coll->src.info_v.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info_v.buffer,
                          ctxs[r]->init_buf, ucc_dt_size(dtype) * all_counts,
                          mem_type, UCC_MEMORY_TYPE_HOST));
            }
            if (r != root || !inplace) {
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
                                       ucc_dt_size(dtype) * my_count,
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
                free(coll->src.info_v.counts);
                free(coll->src.info_v.displacements);
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
        ucc_datatype_t   dtype = (ctxs[0])->args->dst.info.datatype;
        size_t           my_count;

        for (auto r = 0; r < ctxs.size(); r++) {
            if (r != root || !inplace) {
                coll = ctxs[r]->args;
                my_count = coll->dst.info.count;
                clear_buffer(coll->dst.info.buffer,
                    my_count * ucc_dt_size(dtype),
                    mem_type, 0);
            }
        }
    }
    bool data_validate(UccCollCtxVec ctxs)
    {
        bool     ret  = true;
        int      root = ctxs[0]->args->root;
        size_t   dt_size;
        size_t   my_count;
        uint8_t *dsts;

        for (auto r = 0; r < ctxs.size(); r++) {
            if (r == root && inplace) {
                continue;
            }
            dt_size  = ucc_dt_size((ctxs[r])->args->dst.info.datatype);
            my_count = (ctxs[r])->args->dst.info.count;
            if (UCC_MEMORY_TYPE_HOST != mem_type) {
                dsts = (uint8_t *)ucc_malloc(my_count * dt_size, "dsts buf");
                ucc_assert(dsts != nullptr);
                UCC_CHECK(ucc_mc_memcpy(dsts, ctxs[r]->args->dst.info.buffer,
                                        my_count * dt_size,
                                        UCC_MEMORY_TYPE_HOST, mem_type));
            } else {
                dsts = (uint8_t *)ctxs[r]->args->dst.info.buffer;
            }

            for (int i = 0; i < my_count * dt_size; i++) {
                if ((uint8_t)((i + r) % 256) !=
                    dsts[i]) {
                    ret = false;
                    break;
                }
            }

            if (UCC_MEMORY_TYPE_HOST != mem_type) {
                ucc_free(dsts);
                if (!ret) {
                    break;
                }
            }
        }
        return ret;
    }
    void set_root(int _root)
    {
        root = _root;
    }
};

class test_scatterv_0 : public test_scatterv,
                      public ::testing::WithParamInterface<Param_0> {
};

UCC_TEST_P(test_scatterv_0, single)
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

    if (size <= root) {
        GTEST_SKIP();
    }

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

UCC_TEST_P(test_scatterv_0, single_persistent)
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

    if (size <= root) {
        GTEST_SKIP();
    }

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
    , test_scatterv_0,
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

class test_scatterv_1 : public test_scatterv,
                      public ::testing::WithParamInterface<Param_1> {
};

UCC_TEST_P(test_scatterv_1, multiple_host)
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

        if (size <= root) {
            /* skip invalid */
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
    , test_scatterv_1,
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
