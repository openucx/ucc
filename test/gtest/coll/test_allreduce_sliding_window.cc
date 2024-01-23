/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

/*
   This file is for setting up the global work buffer for sliding window 
   allreduce. This entails allocating ucp workers, registering memory,
   exchanging rkeys, and allocating the pipeline datastructure the 
   algorithm uses.
*/

#include "common/test_ucc.h"

#ifdef HAVE_UCX

#include "core/test_mc_reduce.h"
#include "utils/ucc_math.h"

#include <array>

#include "test_allreduce_sliding_window.h"

void test_init_ucp(ucp_context_h *ucp_ctx, ucp_config_t **ucp_config_p)
{
    ucs_status_t  ucs_status;
    ucp_config_t *ucp_config;
    ucp_params_t  ucp_params;
    ucp_context_h ucp_context;

    ucs_status = ucp_config_read(NULL, NULL, &ucp_config);
    EXPECT_EQ(UCS_OK, ucs_status) << "ucp_config_read() returned error: "
                                  << ucs_status_string(ucs_status);

    ucp_params.field_mask = UCP_PARAM_FIELD_FEATURES;
    ucp_params.features   = UCP_FEATURE_TAG | UCP_FEATURE_RMA |
                          UCP_FEATURE_AMO64 | UCP_FEATURE_EXPORTED_MEMH;

    ucs_status = ucp_init(&ucp_params, ucp_config, &ucp_context);
    EXPECT_EQ(UCS_OK, ucs_status) << "ucp_init() returned error: "
                                  << ucs_status_string(ucs_status);

    *ucp_ctx = ucp_context;
    *ucp_config_p = ucp_config;
}

ucs_status_t buffer_export_ucc(ucp_context_h ucp_context, void *buf, size_t len,
                               struct export_buf *ebuf)
{
    ucs_status_t           ucs_status = UCS_OK;
    ucp_mem_map_params_t   params;
    ucp_memh_pack_params_t pack_params;

    ebuf->ucp_context = ucp_context;

    params.field_mask =
        UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
    params.address = buf;
    params.length  = len;

    ucs_status = ucp_mem_map(ucp_context, &params, &ebuf->memh);
    if (ucs_status != UCS_OK) {
         printf("ucp_mem_map() returned error: %s\n", ucs_status_string(ucs_status));
         return ucs_status;
    }

    pack_params.field_mask = UCP_MEMH_PACK_PARAM_FIELD_FLAGS;
    pack_params.flags      = UCP_MEMH_PACK_FLAG_EXPORT;

    ucs_status = ucp_memh_pack(ebuf->memh, &pack_params, &ebuf->packed_memh,
                               &ebuf->packed_memh_len);
    if (ucs_status != UCS_OK) {
        printf("ucp_memh_pack() returned error: %s\n", ucs_status_string(ucs_status));
        return ucs_status;
    }

    return ucs_status;
}

ucs_status_t setup_gwbi(int n_procs, UccCollCtxVec &ctxs,
                        test_ucp_info_t **ucp_infos_p /* out */, bool inplace)
{
    int i;
    ucs_status_t ucs_status = UCS_OK;

    test_ucp_info_t *ucp_infos =
        (test_ucp_info_t *)ucc_malloc(sizeof(test_ucp_info_t) * n_procs);
    EXPECT_NE(ucp_infos, nullptr);
    *ucp_infos_p = ucp_infos;

    // allocate gwbi
    for (auto ctx : ctxs) {
        global_work_buf_info *gwbi =
            (global_work_buf_info *)ucc_malloc(
                sizeof(global_work_buf_info),
                "global work buf info");

        EXPECT_NE(gwbi, nullptr);

        ctx->args->global_work_buffer = gwbi;
    }

    // setup ucp contexts and workers
    for (i = 0; i < n_procs; i++) {
        test_init_ucp(&ucp_infos[i].ucp_ctx, &ucp_infos[i].ucp_config);
        ucp_infos[i].src_ebuf = {0};
        ucp_infos[i].dst_ebuf = {0};
    }

    // set up packed src/dst memh
    for (i = 0; i < n_procs; i++) {
        // my proc's gwbi
        global_work_buf_info *gwbi =
            (global_work_buf_info *)ctxs[i]
                ->args->global_work_buffer;
        // my proc's ucp_info
        test_ucp_info_t *       ucp_info = &ucp_infos[i];
        struct export_buf *dst_ebuf = &ucp_info->dst_ebuf;
        size_t             dst_len  = ctxs[i]->args->dst.info.count *
                         ucc_dt_size(ctxs[i]->args->dst.info.datatype);

        ucs_status = buffer_export_ucc(
                          ucp_info->ucp_ctx, ctxs[i]->args->dst.info.buffer,
                          dst_len, dst_ebuf);
        if (ucs_status != UCS_OK) return ucs_status;

        gwbi->packed_dst_memh = dst_ebuf->packed_memh;

        if (!inplace) {
            size_t src_len = ctxs[i]->args->src.info.count *
                             ucc_dt_size(ctxs[i]->args->src.info.datatype);
            struct export_buf *src_ebuf = &ucp_info->src_ebuf;
            ucs_status = buffer_export_ucc(
                              ucp_info->ucp_ctx, ctxs[i]->args->src.info.buffer,
                              src_len, src_ebuf);
            if (ucs_status != UCS_OK) return ucs_status;

            gwbi->packed_src_memh = src_ebuf->packed_memh;
        }
    }

    // set the flag that indicates the global work buffer was passed
    for (auto ctx : ctxs) {
        ctx->args->mask |=
            UCC_COLL_ARGS_FIELD_FLAGS | UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;
        ctx->args->flags |= UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS;
    }

    return ucs_status;
}

void free_gwbi(int n_procs, UccCollCtxVec &ctxs, test_ucp_info_t *ucp_infos,
               bool inplace)
{
    int i, k;
    ucs_status_t ucs_status;

    // free sbufs, rbufs, src_rkeys, and dst_rkeys
    for (i = 0; i < n_procs; i++) {
        // my proc's ucp_info
        test_ucp_info_t *ucp_info = &ucp_infos[i];

        if (!inplace) {
            struct export_buf *src_ebuf = &ucp_info->src_ebuf;
            if (src_ebuf->memh != 0) {
                ucs_status = ucp_mem_unmap(ucp_info->ucp_ctx, src_ebuf->memh);
                ASSERT_EQ(UCS_OK, ucs_status) << "ucp_mem_unmap() returned error: "
                                            << ucs_status_string(ucs_status);
            }
        }

        struct export_buf *dst_ebuf = &ucp_info->dst_ebuf;
        if (dst_ebuf->memh != 0) {
            ucs_status = ucp_mem_unmap(ucp_info->ucp_ctx, dst_ebuf->memh);
            ASSERT_EQ(UCS_OK, ucs_status) << "ucp_mem_unmap() returned error: "
                                        << ucs_status_string(ucs_status);
        }
    }

    // free ucp contexts
    for (i = 0; i < n_procs; i++) {
        ucp_config_release(ucp_infos[i].ucp_config);
        ucp_cleanup(ucp_infos[i].ucp_ctx);
    }

    // free gwbi and each gwbi's set of pipes
    for (k = 0; k < n_procs; k++) {
        global_work_buf_info *gwbi =
            (global_work_buf_info *) ctxs[k]->args->global_work_buffer;

        ucc_free(gwbi);
    }

    ucc_free(ucp_infos);
}

#endif
