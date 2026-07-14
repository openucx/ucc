/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

/*
 * Self-contained gtests for TL/CUDA alltoall and alltoallv push algorithms.
 *
 * The push algorithms require application-level destination buffer handles
 * pre-exchanged across all ranks (global_memh). This file handles that
 * exchange in-process by:
 *   1. Exporting each rank's CUDA dst buffer via ucc_mem_map(EXPORT)
 *   2. Copying the serialized handle bytes to each peer
 *   3. Importing via ucc_mem_map(IMPORT) from each peer's context
 *   4. Setting dst_memh.global_memh in coll_args before the collective
 *
 * Tests skip gracefully when CUDA IPC is unavailable (single-GPU single-
 * process environments where cudaIpcOpenMemHandle would fail).
 *
 * Algorithm selection:
 *   alltoall:  UCC_TL_CUDA_TUNE = "alltoall:cuda:@push:0-inf:inf"
 *   alltoallv: UCC_TL_CUDA_TUNE = "alltoallv:cuda:@push:0-inf:inf"
 */

#include "common/test_ucc.h"
#include "utils/ucc_math.h"

#ifdef HAVE_CUDA

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

/* Number of ranks used for push tests.  Must be >= 2. */
static const int PUSH_N_PROCS = 4;

/*
 * Exchange destination buffer IPC handles across all ranks.
 *
 * Each rank exports its dst buffer and every other rank imports it, building
 * an nprocs-element global_memh[] array per rank.
 *
 * dst_bufs[r]  - device address of rank r's dst buffer
 * dst_sizes[r] - byte length of rank r's dst buffer
 * ctx_handles  - UCC context handle per rank (team->procs[r].p->ctx_h)
 *
 * On success returns true and populates export_handles / global_memh.
 * On CUDA IPC failure returns false (caller should GTEST_SKIP).
 */
static bool exchange_dst_memh(
        int                               nprocs,
        const std::vector<void *>        &dst_bufs,
        const std::vector<size_t>        &dst_sizes,
        const std::vector<ucc_context_h> &ctx_handles,
        std::vector<ucc_mem_map_mem_h>   &export_handles,
        std::vector<ucc_mem_map_mem_h *> &global_memh)
{
    export_handles.assign(nprocs, nullptr);
    global_memh.assign(nprocs, nullptr);

    std::vector<size_t> exp_sizes(nprocs, 0);

    /* Step 1: export each rank's dst buffer */
    for (int r = 0; r < nprocs; r++) {
        ucc_mem_map_t        seg;
        ucc_mem_map_params_t params;

        seg.address       = dst_bufs[r];
        seg.len           = dst_sizes[r];
        params.segments   = &seg;
        params.n_segments = 1;

        if (ucc_mem_map(ctx_handles[r], UCC_MEM_MAP_MODE_EXPORT,
                        &params, &exp_sizes[r],
                        &export_handles[r]) != UCC_OK) {
            return false;
        }
    }

    size_t max_sz = *std::max_element(exp_sizes.begin(), exp_sizes.end());

    /* Step 2: for each rank build its global_memh[] by importing peers */
    for (int r = 0; r < nprocs; r++) {
        global_memh[r] = new ucc_mem_map_mem_h[nprocs];

        for (int i = 0; i < nprocs; i++) {
            if (i == r) {
                /* push_init skips self; slot can hold the exported handle */
                global_memh[r][i] = export_handles[i];
                continue;
            }

            /* Copy exported bytes and import from rank r's context */
            void *hbuf = malloc(max_sz);
            if (!hbuf) return false;
            memcpy(hbuf, export_handles[i], exp_sizes[i]);

            ucc_mem_map_mem_h    imp = (ucc_mem_map_mem_h)hbuf;
            ucc_mem_map_t        seg;
            ucc_mem_map_params_t params;

            seg.address       = dst_bufs[i];
            seg.len           = dst_sizes[i];
            params.segments   = &seg;
            params.n_segments = 1;

            if (ucc_mem_map(ctx_handles[r], UCC_MEM_MAP_MODE_IMPORT,
                            &params, &max_sz, &imp) != UCC_OK) {
                free(hbuf);
                return false;
            }
            global_memh[r][i] = imp;
        }
    }
    return true;
}

/*
 * Release handles allocated by exchange_dst_memh.
 * export_handles[] are always freed; global_memh[r][i] for i!=r are freed
 * (own-rank slot is the same pointer as export_handles[i] — freed above).
 */
static void release_dst_memh(
        int                               nprocs,
        std::vector<ucc_mem_map_mem_h>   &export_handles,
        std::vector<ucc_mem_map_mem_h *> &global_memh)
{
    for (int r = 0; r < nprocs; r++) {
        if (!global_memh[r]) continue;
        for (int i = 0; i < nprocs; i++) {
            if (i == r) continue; /* same pointer as export_handles[i] */
            if (global_memh[r][i])
                ucc_mem_unmap(&global_memh[r][i]);
        }
        delete[] global_memh[r];
        global_memh[r] = nullptr;
    }
    for (int r = 0; r < nprocs; r++) {
        if (export_handles[r])
            ucc_mem_unmap(&export_handles[r]);
    }
    export_handles.clear();
    global_memh.clear();
}

/* ================================================================== */
/* alltoall push                                                        */
/* ================================================================== */

class test_alltoall_push : public UccCollArgs, public ucc::test {
    std::vector<ucc_mem_map_mem_h>   m_export_handles;
    std::vector<ucc_mem_map_mem_h *> m_global_memh;

public:
    /*
     * Push-specific data_init that takes the team so it can reach every
     * rank's ctx_h for the mem_map export/import exchange.
     *
     * Returns false when CUDA IPC handle exchange fails — caller should
     * call GTEST_SKIP() in that case.
     */
    bool data_init(int nprocs, ucc_datatype_t dtype, size_t single_rank_count,
                   UccCollCtxVec &ctxs, UccTeam_h team)
    {
        ctxs.resize(nprocs);
        size_t buf_size = ucc_dt_size(dtype) * single_rank_count * nprocs;

        std::vector<void *>        dst_bufs(nprocs);
        std::vector<size_t>        dst_sizes(nprocs, buf_size);
        std::vector<ucc_context_h> ctx_handles(nprocs);

        for (int r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll =
                (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));
            ctxs[r] = (gtest_ucc_coll_ctx_t *)calloc(
                1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->coll_type         = UCC_COLL_TYPE_ALLTOALL;
            coll->src.info.mem_type = UCC_MEMORY_TYPE_CUDA;
            coll->src.info.count    = (ucc_count_t)(single_rank_count * nprocs);
            coll->src.info.datatype = dtype;
            coll->dst.info.mem_type = UCC_MEMORY_TYPE_CUDA;
            coll->dst.info.count    = (ucc_count_t)(single_rank_count * nprocs);
            coll->dst.info.datatype = dtype;

            /* Host-side reference buffer: data rank r sends to every peer */
            ctxs[r]->init_buf = ucc_malloc(buf_size, "push_init_buf");
            if (!ctxs[r]->init_buf) return false;
            for (int i = 0; i < nprocs; i++) {
                alltoallx_init_buf(
                    r, i,
                    (uint8_t *)ctxs[r]->init_buf +
                        i * single_rank_count * ucc_dt_size(dtype),
                    single_rank_count * ucc_dt_size(dtype));
            }

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header, buf_size,
                                   UCC_MEMORY_TYPE_CUDA));
            coll->src.info.buffer = ctxs[r]->src_mc_header->addr;
            UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer, ctxs[r]->init_buf,
                                    buf_size, UCC_MEMORY_TYPE_CUDA,
                                    UCC_MEMORY_TYPE_HOST));

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header, buf_size,
                                   UCC_MEMORY_TYPE_CUDA));
            coll->dst.info.buffer = ctxs[r]->dst_mc_header->addr;

            dst_bufs[r]    = coll->dst.info.buffer;
            ctx_handles[r] = team->procs[r].p->ctx_h;
        }

        if (!exchange_dst_memh(nprocs, dst_bufs, dst_sizes, ctx_handles,
                                m_export_handles, m_global_memh)) {
            /* CUDA IPC unavailable — clean up what we allocated */
            for (int r = 0; r < nprocs; r++) {
                if (ctxs[r]) {
                    if (ctxs[r]->src_mc_header) {
                        UCC_CHECK(ucc_mc_free(ctxs[r]->src_mc_header));
                    }
                    if (ctxs[r]->dst_mc_header) {
                        UCC_CHECK(ucc_mc_free(ctxs[r]->dst_mc_header));
                    }
                    if (ctxs[r]->init_buf)
                        ucc_free(ctxs[r]->init_buf);
                    free(ctxs[r]->args);
                    free(ctxs[r]);
                    ctxs[r] = nullptr;
                }
            }
            ctxs.clear();
            release_dst_memh(nprocs, m_export_handles, m_global_memh);
            return false;
        }

        for (int r = 0; r < nprocs; r++) {
            ctxs[r]->args->mask  |= UCC_COLL_ARGS_FIELD_FLAGS |
                                    UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;
            ctxs[r]->args->flags |= UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL;
            ctxs[r]->args->dst_memh.global_memh = m_global_memh[r];
        }
        return true;
    }

    /* Satisfy pure-virtual — not used for push tests */
    void data_init(int /*nprocs*/, ucc_datatype_t /*dtype*/, size_t /*count*/,
                   UccCollCtxVec & /*ctxs*/,
                   bool /*persistent*/ = false) override
    {
        ADD_FAILURE() << "test_alltoall_push: use the team-aware data_init overload";
    }

    void data_fini(UccCollCtxVec ctxs) override
    {
        release_dst_memh((int)ctxs.size(), m_export_handles, m_global_memh);
        for (gtest_ucc_coll_ctx_t *ctx : ctxs) {
            UCC_CHECK(ucc_mc_free(ctx->src_mc_header));
            UCC_CHECK(ucc_mc_free(ctx->dst_mc_header));
            ucc_free(ctx->init_buf);
            free(ctx->args);
            free(ctx);
        }
    }

    bool data_validate(UccCollCtxVec ctxs) override
    {
        int  nprocs = (int)ctxs.size();
        bool ret    = true;

        std::vector<uint8_t *> dsts(nprocs);
        for (int r = 0; r < nprocs; r++) {
            size_t buf_size = ucc_dt_size(ctxs[r]->args->dst.info.datatype) *
                              (size_t)ctxs[r]->args->dst.info.count;
            dsts[r] = (uint8_t *)ucc_malloc(buf_size, "push_val_buf");
            EXPECT_NE(dsts[r], nullptr);
            UCC_CHECK(ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info.buffer,
                                    buf_size, UCC_MEMORY_TYPE_HOST,
                                    UCC_MEMORY_TYPE_CUDA));
        }

        for (int r = 0; r < nprocs && ret; r++) {
            ucc_coll_args_t *coll = ctxs[r]->args;
            size_t per_rank = coll->dst.info.count / nprocs;
            for (int i = 0; i < nprocs; i++) {
                size_t rank_sz = ucc_dt_size(coll->dst.info.datatype) * per_rank;
                /* init(r,i) was used → validate(r,i) matches init(i,r) received */
                if (alltoallx_validate_buf(
                        r, i, (uint8_t *)dsts[r] + rank_sz * i, rank_sz) != 0) {
                    ret = false;
                    break;
                }
            }
        }

        for (int r = 0; r < nprocs; r++)
            ucc_free(dsts[r]);
        return ret;
    }
};

using Param_alltoall_push = std::tuple<ucc_datatype_t, int>;

class test_alltoall_push_0 : public test_alltoall_push,
      public ::testing::WithParamInterface<Param_alltoall_push> {};

UCC_TEST_P(test_alltoall_push_0, single)
{
    const ucc_datatype_t dtype = std::get<0>(GetParam());
    const int            count = std::get<1>(GetParam());
    const int            n     = PUSH_N_PROCS;

    SET_MEM_TYPE(UCC_MEMORY_TYPE_CUDA);

    ucc_job_env_t env = {
        {"UCC_TL_CUDA_TUNE", "alltoall:cuda:@push:0-inf:inf"},
        {"UCC_CL_BASIC_TUNE", "inf"}
    };
    UccJob    job(n, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h team = job.create_team(n);
    UccCollCtxVec ctxs;

    if (!data_init(n, dtype, count, ctxs, team)) {
        GTEST_SKIP() << "CUDA IPC handle exchange unavailable "
                        "(single-process/single-GPU environment)";
    }

    UccReq req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

UCC_TEST_P(test_alltoall_push_0, persistent)
{
    const ucc_datatype_t dtype   = std::get<0>(GetParam());
    const int            count   = std::get<1>(GetParam());
    const int            n       = PUSH_N_PROCS;
    const int            n_calls = 3;

    SET_MEM_TYPE(UCC_MEMORY_TYPE_CUDA);

    ucc_job_env_t env = {
        {"UCC_TL_CUDA_TUNE", "alltoall:cuda:@push:0-inf:inf"},
        {"UCC_CL_BASIC_TUNE", "inf"}
    };
    UccJob    job(n, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h team = job.create_team(n);
    UccCollCtxVec ctxs;

    if (!data_init(n, dtype, count, ctxs, team)) {
        GTEST_SKIP() << "CUDA IPC handle exchange unavailable "
                        "(single-process/single-GPU environment)";
    }

    /* Add PERSISTENT flag on top of what data_init already set */
    for (auto *ctx : ctxs) {
        ctx->args->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        ctx->args->flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
    }

    UccReq req(team, ctxs);
    for (int i = 0; i < n_calls; i++) {
        req.start();
        req.wait();
        EXPECT_EQ(true, data_validate(ctxs));
        /* Clear entire dst buffer between calls */
        for (auto *ctx : ctxs) {
            clear_buffer(ctx->dst_mc_header->addr,
                         ucc_dt_size(dtype) * (size_t)count * (size_t)n,
                         UCC_MEMORY_TYPE_CUDA, 0);
        }
    }
    data_fini(ctxs);
}

INSTANTIATE_TEST_CASE_P(
    , test_alltoall_push_0,
    ::testing::Combine(
        ::testing::Values(UCC_DT_UINT8, UCC_DT_FLOAT32),
        ::testing::Values(1, 64, 1024)));

/* ================================================================== */
/* alltoallv push                                                       */
/* ================================================================== */

template <typename T>
class test_alltoallv_push : public UccCollArgs, public ucc::test {
    std::vector<ucc_mem_map_mem_h>   m_export_handles;
    std::vector<ucc_mem_map_mem_h *> m_global_memh;

public:
    /*
     * data_init for alltoallv push.  Variable-length counts follow the same
     * pattern as test_alltoallv: rank r sends (nprocs + r - i) * count
     * elements to rank i, and expects to receive (nprocs - r + i) * count
     * elements from rank i.
     *
     * Returns false when CUDA IPC exchange fails — caller should GTEST_SKIP.
     */
    bool data_init(int nprocs, ucc_datatype_t dtype, size_t count,
                   UccCollCtxVec &ctxs, UccTeam_h team)
    {
        ctxs.resize(nprocs);

        std::vector<void *>        dst_bufs(nprocs);
        std::vector<size_t>        dst_sizes(nprocs);
        std::vector<ucc_context_h> ctx_handles(nprocs);

        for (int r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll =
                (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));
            ctxs[r] = (gtest_ucc_coll_ctx_t *)calloc(
                1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->coll_type = UCC_COLL_TYPE_ALLTOALLV;
            coll->mask      = UCC_COLL_ARGS_FIELD_FLAGS |
                              UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;
            coll->flags     = UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                              UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT |
                              UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                              UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER |
                              UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL;

            coll->src.info_v.mem_type     = UCC_MEMORY_TYPE_CUDA;
            coll->src.info_v.datatype     = dtype;
            coll->src.info_v.counts       =
                (ucc_count_t *)malloc(sizeof(T) * nprocs);
            coll->src.info_v.displacements =
                (ucc_aint_t *)malloc(sizeof(T) * nprocs);

            coll->dst.info_v.mem_type     = UCC_MEMORY_TYPE_CUDA;
            coll->dst.info_v.datatype     = dtype;
            coll->dst.info_v.counts       =
                (ucc_count_t *)malloc(sizeof(T) * nprocs);
            coll->dst.info_v.displacements =
                (ucc_aint_t *)malloc(sizeof(T) * nprocs);

            /* Build send counts / displacements */
            size_t sbuf_elems = 0;
            for (int i = 0; i < nprocs; i++) {
                T sc = (T)((nprocs + r - i) * count);
                ((T *)coll->src.info_v.counts)[i]       = sc;
                ((T *)coll->src.info_v.displacements)[i] = (T)sbuf_elems;
                sbuf_elems += sc;
            }
            /* Force one zero-count for corner-case coverage */
            ((T *)coll->src.info_v.counts)[(r + 1) % nprocs] = 0;

            size_t sbuf_bytes = sbuf_elems * ucc_dt_size(dtype);
            ctxs[r]->init_buf = ucc_malloc(sbuf_bytes, "alltoallv_push_ibuf");
            if (!ctxs[r]->init_buf) return false;
            for (int i = 0; i < nprocs; i++) {
                alltoallx_init_buf(
                    r, i,
                    (uint8_t *)ctxs[r]->init_buf +
                        ((T *)coll->src.info_v.displacements)[i] *
                            ucc_dt_size(dtype),
                    ((T *)coll->src.info_v.counts)[i] * ucc_dt_size(dtype));
            }
            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->src_mc_header, sbuf_bytes,
                                   UCC_MEMORY_TYPE_CUDA));
            coll->src.info_v.buffer = ctxs[r]->src_mc_header->addr;
            UCC_CHECK(ucc_mc_memcpy(coll->src.info_v.buffer, ctxs[r]->init_buf,
                                    sbuf_bytes, UCC_MEMORY_TYPE_CUDA,
                                    UCC_MEMORY_TYPE_HOST));

            /* Build recv counts / displacements */
            size_t rbuf_elems = 0;
            for (int i = 0; i < nprocs; i++) {
                T rc = (T)((nprocs - r + i) * count);
                ((T *)coll->dst.info_v.counts)[i]       = rc;
                ((T *)coll->dst.info_v.displacements)[i] = (T)rbuf_elems;
                rbuf_elems += rc;
            }
            ((T *)coll->dst.info_v.counts)[(r - 1 + nprocs) % nprocs] = 0;

            size_t rbuf_bytes      = rbuf_elems * ucc_dt_size(dtype);
            ctxs[r]->rbuf_size     = rbuf_bytes;
            dst_sizes[r]           = rbuf_bytes;
            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->dst_mc_header, rbuf_bytes,
                                   UCC_MEMORY_TYPE_CUDA));
            coll->dst.info_v.buffer = ctxs[r]->dst_mc_header->addr;

            dst_bufs[r]    = coll->dst.info_v.buffer;
            ctx_handles[r] = team->procs[r].p->ctx_h;
        }

        if (!exchange_dst_memh(nprocs, dst_bufs, dst_sizes, ctx_handles,
                                m_export_handles, m_global_memh)) {
            for (int r = 0; r < nprocs; r++) {
                if (ctxs[r]) {
                    ucc_coll_args_t *c = ctxs[r]->args;
                    if (ctxs[r]->src_mc_header) {
                        UCC_CHECK(ucc_mc_free(ctxs[r]->src_mc_header));
                    }
                    if (ctxs[r]->dst_mc_header) {
                        UCC_CHECK(ucc_mc_free(ctxs[r]->dst_mc_header));
                    }
                    if (ctxs[r]->init_buf)
                        ucc_free(ctxs[r]->init_buf);
                    free(c->src.info_v.counts);
                    free(c->src.info_v.displacements);
                    free(c->dst.info_v.counts);
                    free(c->dst.info_v.displacements);
                    free(c);
                    free(ctxs[r]);
                    ctxs[r] = nullptr;
                }
            }
            ctxs.clear();
            release_dst_memh(nprocs, m_export_handles, m_global_memh);
            return false;
        }

        for (int r = 0; r < nprocs; r++)
            ctxs[r]->args->dst_memh.global_memh = m_global_memh[r];

        return true;
    }

    /* Satisfy pure-virtual */
    void data_init(int /*nprocs*/, ucc_datatype_t /*dtype*/, size_t /*count*/,
                   UccCollCtxVec & /*ctxs*/,
                   bool /*persistent*/ = false) override
    {
        ADD_FAILURE() << "test_alltoallv_push: use the team-aware data_init overload";
    }

    void data_fini(UccCollCtxVec ctxs) override
    {
        release_dst_memh((int)ctxs.size(), m_export_handles, m_global_memh);
        for (gtest_ucc_coll_ctx_t *ctx : ctxs) {
            ucc_coll_args_t *coll = ctx->args;
            UCC_CHECK(ucc_mc_free(ctx->src_mc_header));
            UCC_CHECK(ucc_mc_free(ctx->dst_mc_header));
            free(coll->src.info_v.counts);
            free(coll->src.info_v.displacements);
            free(coll->dst.info_v.counts);
            free(coll->dst.info_v.displacements);
            ucc_free(ctx->init_buf);
            free(coll);
            free(ctx);
        }
    }

    bool data_validate(UccCollCtxVec ctxs) override
    {
        int  nprocs = (int)ctxs.size();
        bool ret    = true;

        std::vector<uint8_t *> dsts(nprocs);
        for (int r = 0; r < nprocs; r++) {
            dsts[r] = (uint8_t *)ucc_malloc(ctxs[r]->rbuf_size,
                                            "alltoallv_push_val");
            EXPECT_NE(dsts[r], nullptr);
            UCC_CHECK(ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info_v.buffer,
                                    ctxs[r]->rbuf_size, UCC_MEMORY_TYPE_HOST,
                                    UCC_MEMORY_TYPE_CUDA));
        }

        for (int r = 0; r < nprocs && ret; r++) {
            ucc_coll_args_t *coll = ctxs[r]->args;
            for (int i = 0; i < nprocs; i++) {
                T      rc        = ((T *)coll->dst.info_v.counts)[i];
                T      rd        = ((T *)coll->dst.info_v.displacements)[i];
                size_t rank_sz   = ucc_dt_size(coll->dst.info_v.datatype) * rc;
                size_t rank_off  = ucc_dt_size(coll->dst.info_v.datatype) * rd;
                if (alltoallx_validate_buf(
                        r, i,
                        (uint8_t *)dsts[r] + rank_off,
                        rank_sz) != 0) {
                    ret = false;
                    break;
                }
            }
        }

        for (int r = 0; r < nprocs; r++)
            ucc_free(dsts[r]);
        return ret;
    }
};

using Param_alltoallv_push = std::tuple<ucc_datatype_t, int>;

class test_alltoallv_push_0 : public test_alltoallv_push<uint64_t>,
      public ::testing::WithParamInterface<Param_alltoallv_push> {};

class test_alltoallv_push_1 : public test_alltoallv_push<uint32_t>,
      public ::testing::WithParamInterface<Param_alltoallv_push> {};

UCC_TEST_P(test_alltoallv_push_0, single)
{
    const ucc_datatype_t dtype = std::get<0>(GetParam());
    const int            count = std::get<1>(GetParam());
    const int            n     = PUSH_N_PROCS;

    SET_MEM_TYPE(UCC_MEMORY_TYPE_CUDA);

    ucc_job_env_t env = {
        {"UCC_TL_CUDA_TUNE", "alltoallv:cuda:@push:0-inf:inf"},
        {"UCC_CL_BASIC_TUNE", "inf"}
    };
    UccJob    job(n, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h team = job.create_team(n);
    UccCollCtxVec ctxs;

    if (!data_init(n, dtype, count, ctxs, team)) {
        GTEST_SKIP() << "CUDA IPC handle exchange unavailable "
                        "(single-process/single-GPU environment)";
    }

    UccReq req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

UCC_TEST_P(test_alltoallv_push_1, single)
{
    const ucc_datatype_t dtype = std::get<0>(GetParam());
    const int            count = std::get<1>(GetParam());
    const int            n     = PUSH_N_PROCS;

    SET_MEM_TYPE(UCC_MEMORY_TYPE_CUDA);

    ucc_job_env_t env = {
        {"UCC_TL_CUDA_TUNE", "alltoallv:cuda:@push:0-inf:inf"},
        {"UCC_CL_BASIC_TUNE", "inf"}
    };
    UccJob    job(n, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h team = job.create_team(n);
    UccCollCtxVec ctxs;

    if (!data_init(n, dtype, count, ctxs, team)) {
        GTEST_SKIP() << "CUDA IPC handle exchange unavailable "
                        "(single-process/single-GPU environment)";
    }

    UccReq req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

INSTANTIATE_TEST_CASE_P(
    64bit, test_alltoallv_push_0,
    ::testing::Combine(
        ::testing::Values(UCC_DT_UINT8, UCC_DT_FLOAT32),
        ::testing::Values(1, 64, 1024)));

INSTANTIATE_TEST_CASE_P(
    32bit, test_alltoallv_push_1,
    ::testing::Combine(
        ::testing::Values(UCC_DT_UINT8, UCC_DT_FLOAT32),
        ::testing::Values(1, 64, 1024)));

#endif /* HAVE_CUDA */
