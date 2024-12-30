/**
 * Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"

#ifdef HAVE_CUDA

using Param = std::tuple<ucc_coll_type_t, ucc_memory_type_t,
                         ucc_memory_type_t, int>;

class test_asymmetric_memory : public ucc::test,
                       public ::testing::WithParamInterface<Param>
{
public:
    UccCollCtxVec ctxs;
    void data_init(ucc_coll_type_t coll_type, ucc_memory_type_t src_mem_type,
                   ucc_memory_type_t dst_mem_type, UccTeam_h team, bool persistent = false) {
        ucc_rank_t tsize = team->procs.size();
        int root = 0;
        size_t msglen = 2048;
        size_t src_modifier = 1;
        size_t dst_modifier = 1;
        ctxs.resize(tsize);

        if (coll_type == UCC_COLL_TYPE_GATHER) {
            dst_modifier = tsize;
        } else if (coll_type == UCC_COLL_TYPE_SCATTER) {
            src_modifier = tsize;
        }

        for (int i = 0; i < tsize; i++) {
            ctxs[i] = (gtest_ucc_coll_ctx_t*)
                            calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                calloc(1, sizeof(ucc_coll_args_t));

            ctxs[i]->args           = coll;
            coll->coll_type         = coll_type;
            coll->src.info.mem_type = src_mem_type;
            coll->src.info.count    = (ucc_count_t)msglen * src_modifier;
            coll->src.info.datatype = UCC_DT_INT8;
            coll->root              = root;
            if (persistent) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
            }

            if (i == root || coll_type != UCC_COLL_TYPE_SCATTER) {
                UCC_CHECK(ucc_mc_alloc(&ctxs[i]->src_mc_header,
                                        msglen * src_modifier, src_mem_type));
                coll->src.info.buffer = ctxs[i]->src_mc_header->addr;

                ctxs[i]->init_buf = ucc_malloc(msglen * src_modifier,
                                                    "init buf");
                EXPECT_NE(ctxs[i]->init_buf, nullptr);
                uint8_t *sbuf = (uint8_t*)ctxs[i]->init_buf;
                for (int j = 0; j < msglen * src_modifier; j++) {
                    sbuf[j] = (uint8_t) 1;
                }
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer,
                                        ctxs[i]->init_buf,
                                        msglen * src_modifier, src_mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            }

            ctxs[i]->rbuf_size = msglen * dst_modifier;
            if (i == root || coll_type == UCC_COLL_TYPE_SCATTER) {
                UCC_CHECK(ucc_mc_alloc(&ctxs[i]->dst_mc_header,
                                        ctxs[i]->rbuf_size, dst_mem_type));
                coll->dst.info.buffer   = ctxs[i]->dst_mc_header->addr;
                coll->dst.info.count    = (ucc_count_t)ctxs[i]->rbuf_size;
                coll->dst.info.datatype = UCC_DT_INT8;
                coll->dst.info.mem_type = dst_mem_type;
            }
        }
    }

    void data_fini()
    {
        for (int i = 0; i < ctxs.size(); i++) {
            gtest_ucc_coll_ctx_t *ctx = ctxs[i];
            if (!ctx) {
                continue;
            }
            ucc_coll_args_t* coll = ctx->args;
            if (i == coll->root || coll->coll_type != UCC_COLL_TYPE_SCATTER) {
                ucc_free(ctx->init_buf);
                UCC_CHECK(ucc_mc_free(ctx->src_mc_header));
            }
            if (i == coll->root || coll->coll_type == UCC_COLL_TYPE_SCATTER) {
                UCC_CHECK(ucc_mc_free(ctx->dst_mc_header));
            }
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }

    bool data_validate(uint8_t data = 1)
    {
        bool              ret  = true;
        int               root = 0;
        uint8_t           result = data;
        ucc_memory_type_t dst_mem_type;
        uint8_t          *rst;

        if (ctxs[0]->args->coll_type == UCC_COLL_TYPE_REDUCE) {
            result *= (uint8_t) ctxs.size();
        }

        for (int i = 0; i < ctxs.size(); i++) {
            if (!ctxs[i]) {
                continue;
            }

            root = ctxs[i]->args->root;

            if (i == root || ctxs[i]->args->coll_type == UCC_COLL_TYPE_SCATTER) {
                dst_mem_type = ctxs[i]->args->dst.info.mem_type;

                rst = (uint8_t*) ucc_malloc(ctxs[i]->rbuf_size, "validation buf");
                EXPECT_NE(rst, nullptr);

                UCC_CHECK(ucc_mc_memcpy(rst, ctxs[i]->args->dst.info.buffer,
                                        ctxs[i]->rbuf_size,
                                        UCC_MEMORY_TYPE_HOST, dst_mem_type));

                for (int j = 0; j < ctxs[i]->rbuf_size; j++) {
                    if (result != rst[j]) {
                        ret = false;
                        break;
                    }
                }

                ucc_free(rst);
            }
        }
        
        return ret;
    }

    void data_update(uint8_t data) {
        ucc_rank_t tsize = ctxs.size();
        size_t msglen = 2048;
        size_t src_modifier = 1;
        ucc_coll_type_t coll_type = ctxs[0]->args->coll_type;
        int root = ctxs[0]->args->root;
        ucc_memory_type_t src_mem_type = ctxs[0]->args->src.info.mem_type;

        if (coll_type == UCC_COLL_TYPE_SCATTER) {
            src_modifier = tsize;
        }

        for (int i = 0; i < tsize; i++) {
            if (i == root || coll_type != UCC_COLL_TYPE_SCATTER) {
                ucc_coll_args_t *coll = ctxs[i]->args;
                uint8_t *sbuf = (uint8_t*)ctxs[i]->init_buf;
                for (int j = 0; j < msglen * src_modifier; j++) {
                    sbuf[j] = (uint8_t) data;
                }
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer,
                                        ctxs[i]->init_buf,
                                        msglen * src_modifier, src_mem_type,
                                        UCC_MEMORY_TYPE_HOST));
            }
        }
    }
};


class test_asymmetric_memory_v : public ucc::test,
                       public ::testing::WithParamInterface<Param>
{
public:
    UccCollCtxVec ctxs;
    void data_init(ucc_coll_type_t coll_type, ucc_memory_type_t src_mem_type,
                   ucc_memory_type_t dst_mem_type, UccTeam_h team) {
        int              nprocs = team->n_procs;
        size_t           count  = 2048;
        ucc_rank_t       root   = 0;
        ucc_coll_args_t *coll;
        int             *counts, *displs;
        size_t           my_count, all_counts;

        ctxs.resize(nprocs);
        for (auto r = 0; r < nprocs; r++) {
            coll = (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));
            my_count = (nprocs - r) * count;
            ctxs[r] =
                (gtest_ucc_coll_ctx_t *)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args           = coll;
            coll->mask              = 0;
            coll->flags             = 0;
            coll->coll_type         = coll_type;
            coll->root              = root;
            if (coll_type == UCC_COLL_TYPE_GATHERV) {
                coll->src.info.mem_type = src_mem_type;
                coll->src.info.count    = (ucc_count_t)my_count;
                coll->src.info.datatype = UCC_DT_INT8;

                ctxs[r]->init_buf =
                    ucc_malloc(ucc_dt_size(UCC_DT_INT8) * my_count, "init buf");
                EXPECT_NE(ctxs[r]->init_buf, nullptr);
                for (int i = 0; i < my_count * ucc_dt_size(UCC_DT_INT8); i++) {
                    uint8_t *sbuf = (uint8_t *)ctxs[r]->init_buf;
                    sbuf[i]       = ((i + r) % 256);
                }

                if (r == root) {
                    all_counts = 0;
                    counts = (int*)malloc(sizeof(int) * nprocs);
                    EXPECT_NE(counts, nullptr);
                    displs = (int*)malloc(sizeof(int) * nprocs);
                    EXPECT_NE(displs, nullptr);

                    for (int i = 0; i < nprocs; i++) {
                        counts[i] = (nprocs - i) * count;
                        displs[i] = all_counts;
                        all_counts += counts[i];
                    }

                    coll->dst.info_v.mem_type      = dst_mem_type;
                    coll->dst.info_v.counts        = (ucc_count_t *)counts;
                    coll->dst.info_v.displacements = (ucc_aint_t *)displs;
                    coll->dst.info_v.datatype      = UCC_DT_INT8;

                    ctxs[r]->rbuf_size = ucc_dt_size(UCC_DT_INT8) * all_counts;
                    UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
                                        ctxs[r]->rbuf_size, dst_mem_type));
                    coll->dst.info_v.buffer = ctxs[r]->dst_mc_header->addr;
                }

                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                        ucc_dt_size(UCC_DT_INT8) * my_count,
                                        src_mem_type));
                coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
                UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer,
                                        ctxs[r]->init_buf,
                                        ucc_dt_size(UCC_DT_INT8) * my_count,
                                        src_mem_type, UCC_MEMORY_TYPE_HOST));
            } else {
                // scatterv
                coll->dst.info.mem_type = dst_mem_type;
                coll->dst.info.count    = (ucc_count_t)my_count;
                coll->dst.info.datatype = UCC_DT_INT8;

                if (r == root) {
                    all_counts = 0;
                    counts = (int*)malloc(sizeof(int) * nprocs);
                    EXPECT_NE(counts, nullptr);
                    displs = (int*)malloc(sizeof(int) * nprocs);
                    EXPECT_NE(displs, nullptr);

                    for (int i = 0; i < nprocs; i++) {
                        counts[i] = (nprocs - i) * count;
                        displs[i] = all_counts;
                        all_counts += counts[i];
                    }
                    
                    ctxs[r]->init_buf =
                        ucc_malloc(ucc_dt_size(UCC_DT_INT8) * all_counts, "init buf");
                    EXPECT_NE(ctxs[r]->init_buf, nullptr);
                    uint8_t *sbuf = (uint8_t*)ctxs[r]->init_buf;
                    for (int p = 0; p < nprocs; p++) {
                        for (int i = 0; i < ucc_dt_size(UCC_DT_INT8) * counts[p]; i++) {
                            sbuf[(displs[p] * ucc_dt_size(UCC_DT_INT8) + i)] =
                                (uint8_t)((i + p) % 256);
                        }
                    }

                    coll->src.info_v.mem_type      = src_mem_type;
                    coll->src.info_v.counts        = (ucc_count_t *)counts;
                    coll->src.info_v.displacements = (ucc_aint_t *)displs;
                    coll->src.info_v.datatype      = UCC_DT_INT8;

                    UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header,
                                        ucc_dt_size(UCC_DT_INT8) * all_counts,
                                        src_mem_type));
                    coll->src.info_v.buffer = ctxs[r]->src_mc_header->addr;
                    UCC_CHECK(ucc_mc_memcpy(coll->src.info_v.buffer,
                            ctxs[r]->init_buf, ucc_dt_size(UCC_DT_INT8) * all_counts,
                            src_mem_type, UCC_MEMORY_TYPE_HOST));
                }
                UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header,
                                    ucc_dt_size(UCC_DT_INT8) * my_count,
                                    dst_mem_type));
                coll->dst.info.buffer = ctxs[r]->dst_mc_header->addr;
            }
        }
    }

    bool data_validate()
    {
        bool   ret      = true;
        int    root     = ctxs[0]->args->root;
        int   *displs   = (int*)ctxs[root]->args->dst.info_v.displacements;
        size_t dt_size;
        ucc_memory_type_t dst_mem_type;
        ucc_count_t my_count;
        uint8_t    *dsts;

        if (ctxs[root]->args->coll_type == UCC_COLL_TYPE_GATHERV) {
            dt_size  = ucc_dt_size(ctxs[root]->args->src.info.datatype);
            dst_mem_type = ctxs[root]->args->dst.info_v.mem_type;
            if (UCC_MEMORY_TYPE_HOST != dst_mem_type) {
                dsts = (uint8_t *)ucc_malloc(ctxs[root]->rbuf_size, "dsts buf");
                ucc_assert(dsts != nullptr);
                UCC_CHECK(ucc_mc_memcpy(dsts, ctxs[root]->args->dst.info_v.buffer,
                                        ctxs[root]->rbuf_size,
                                        UCC_MEMORY_TYPE_HOST, dst_mem_type));
            } else {
                dsts = (uint8_t *)ctxs[root]->args->dst.info_v.buffer;
            }

            for (int r = 0; r < ctxs.size(); r++) {
                my_count = ctxs[r]->args->src.info.count;
                for (int i = 0; i < my_count * dt_size; i++) {
                    if ((uint8_t)((i + r) % 256) !=
                        dsts[(displs[r] * dt_size + i)]) {
                        ret = false;
                        break;
                    }
                }
            }

            if (UCC_MEMORY_TYPE_HOST != dst_mem_type) {
                ucc_free(dsts);
            }
        } else {
            // scatterv
            dst_mem_type = ctxs[root]->args->dst.info.mem_type;
            for (auto r = 0; r < ctxs.size(); r++) {
                dt_size  = ucc_dt_size((ctxs[r])->args->dst.info.datatype);
                my_count = (ctxs[r])->args->dst.info.count;
                if (UCC_MEMORY_TYPE_HOST != dst_mem_type) {
                    dsts = (uint8_t *)ucc_malloc(my_count * dt_size, "dsts buf");
                    ucc_assert(dsts != nullptr);
                    UCC_CHECK(ucc_mc_memcpy(dsts, ctxs[r]->args->dst.info.buffer,
                                            my_count * dt_size,
                                            UCC_MEMORY_TYPE_HOST, dst_mem_type));
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

                if (UCC_MEMORY_TYPE_HOST != dst_mem_type) {
                    ucc_free(dsts);
                    if (!ret) {
                        break;
                    }
                }
            }
        }

        return ret;
    }

    void data_fini()
    {
        int root = ctxs[0]->args->root;
        for (auto r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t *coll = ctxs[r]->args;
            if (coll->coll_type == UCC_COLL_TYPE_GATHERV) {
                if (r == root) {
                    UCC_CHECK(ucc_mc_free(ctxs[r]->dst_mc_header));
                    free(coll->dst.info_v.counts);
                    free(coll->dst.info_v.displacements);
                }
                UCC_CHECK(ucc_mc_free(ctxs[r]->src_mc_header));
            } else {
                // scatterv
                if (r == root) {
                    UCC_CHECK(ucc_mc_free(ctxs[r]->src_mc_header));
                    free(coll->src.info_v.counts);
                    free(coll->src.info_v.displacements);
                }
                UCC_CHECK(ucc_mc_free(ctxs[r]->dst_mc_header));
            }
            ucc_free(ctxs[r]->init_buf);
            free(coll);
            free(ctxs[r]);
        }
        ctxs.clear();
    }
};

#define TEST_ASYM_DECLARE                                                      \
    const ucc_coll_type_t   coll_type    = std::get<0>(GetParam());            \
    const ucc_memory_type_t src_mem_type = std::get<1>(GetParam());            \
    const ucc_memory_type_t dst_mem_type = std::get<2>(GetParam());            \
    const int               n_procs      = std::get<3>(GetParam());            \
                                                                               \
    UccJob    job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL);                        \
    UccTeam_h team = job.create_team(n_procs);                                 \
                                                                               \
    data_init(coll_type, src_mem_type, dst_mem_type, team);                    \
    UccReq req(team, ctxs);                                                    \
    if (req.status != UCC_OK) {                                                \
        data_fini();                                                           \
        GTEST_SKIP() << "ucc_collective_init returned "                        \
                     << ucc_status_string(req.status);                         \
    }                                                                          \
    req.start();                                                               \
    req.wait();                                                                \
    EXPECT_EQ(true, data_validate());                                          \
    data_fini();

UCC_TEST_P(test_asymmetric_memory, single)
{
    TEST_ASYM_DECLARE
}

UCC_TEST_P(test_asymmetric_memory, persistent)
{
    const ucc_coll_type_t   coll_type    = std::get<0>(GetParam());
    const ucc_memory_type_t src_mem_type = std::get<1>(GetParam());
    const ucc_memory_type_t dst_mem_type = std::get<2>(GetParam());
    const int               n_procs      = std::get<3>(GetParam());
    int                     times        = 3;

    UccJob    job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL);
    UccTeam_h team = job.create_team(n_procs);

    data_init(coll_type, src_mem_type, dst_mem_type, team, /*persistent*/true);
    UccReq req(team, ctxs);
    if (req.status != UCC_OK) {
        data_fini();
        GTEST_SKIP() << "ucc_collective_init returned "
                     << ucc_status_string(req.status);
    }
    for (; times > 0; times--) {
        data_update(times); // Set each element in src to times
        req.start();
        req.wait();
        EXPECT_EQ(true, data_validate(times)); // Check that the dst was correct based on times
    }
    data_fini();
}

INSTANTIATE_TEST_CASE_P
(
    , test_asymmetric_memory,
        ::testing::Combine
        (
            ::testing::Values(UCC_COLL_TYPE_REDUCE, UCC_COLL_TYPE_GATHER, UCC_COLL_TYPE_SCATTER), // coll type (scatter may be skipped because tl/ucp does not support scatter)
            ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA),                        // src mem type
            ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA),                        // dst mem type
            ::testing::Values(8)                                                                  // n_procs
        )
);

UCC_TEST_P(test_asymmetric_memory_v, single_v)
{
    TEST_ASYM_DECLARE
}

INSTANTIATE_TEST_CASE_P
(
    , test_asymmetric_memory_v,
        ::testing::Combine
        (
            ::testing::Values(UCC_COLL_TYPE_GATHERV, UCC_COLL_TYPE_SCATTERV), // coll type
            ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA),    // src mem type
            ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA),    // dst mem type
            ::testing::Values(8)                                              // n_procs
        )
);



#endif
