/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp_team_mem.h"
#include "tl_ucp_ep.h"
#include "tl_ucp_coll.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_assert.h"
#include <string.h>

ucc_status_t ucc_tl_ucp_team_mem_map_size_exch(ucc_tl_ucp_team_t *team)
{
    ucc_tl_ucp_context_t         *ctx        = UCC_TL_UCP_TEAM_CTX(team);
    const ucc_base_team_params_t *bparams    = &team->super.super.params;
    ucc_rank_t                    local_rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t                    size       = UCC_TL_TEAM_SIZE(team);
    uint64_t                     *sbuf       = NULL;
    uint64_t                     *rbuf       = NULL;
    ucc_mem_map_params_t         *mp;
    uint64_t                      n_segs;
    ucc_subset_t                  subset;
    ucc_status_t                  status;
    ucp_mem_map_params_t          mmap_params;
    ucp_mem_h                     mh;
    void                         *packed_key;
    size_t                        packed_key_len;
    ucs_status_t                  ucs_status;
    uint64_t                      i;

    mp     = (ucc_mem_map_params_t *)&bparams->params.mem_params;
    n_segs = mp->n_segments;

    team->mem_segs = ucc_calloc(n_segs, sizeof(ucc_tl_ucp_remote_info_t),
                                "team_mem_segs");
    if (!team->mem_segs) {
        goto err_no_memory;
    }

    team->team_remote_va = ucc_calloc((size_t)size * n_segs, sizeof(uint64_t),
                                      "team_remote_va");
    if (!team->team_remote_va) {
        goto err_no_memory;
    }

    team->team_remote_len = ucc_calloc((size_t)size * n_segs, sizeof(size_t),
                                       "team_remote_len");
    if (!team->team_remote_len) {
        goto err_no_memory;
    }

    team->team_rkeys = ucc_calloc((size_t)size * n_segs, sizeof(ucp_rkey_h),
                                  "team_rkeys");
    if (!team->team_rkeys) {
        goto err_no_memory;
    }

    /* Size exchange: allgather packed_key_len per segment */
    sbuf = ucc_calloc(n_segs, sizeof(uint64_t), "mem_map_sbuf_p1");
    if (!sbuf) {
        goto err_no_memory;
    }

    rbuf = ucc_malloc((size_t)size * n_segs * sizeof(uint64_t),
                      "mem_map_rbuf_p1");
    if (!rbuf) {
        goto err_no_memory;
    }

    /* Register user segments, dup packed keys, fill phase 1 sbuf */
    for (i = 0; i < n_segs; i++) {
        mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_ADDRESS |
                                 UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        mmap_params.address    = mp->segments[i].address;
        mmap_params.length     = mp->segments[i].len;

        ucs_status = ucp_mem_map(ctx->worker.ucp_context, &mmap_params, &mh);
        if (UCS_OK != ucs_status) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "ucp_mem_map failed for user segment %lu: %s",
                     i, ucs_status_string(ucs_status));
            status = ucs_status_to_ucc_status(ucs_status);
            goto err;
        }

        ucs_status = ucp_rkey_pack(ctx->worker.ucp_context, mh,
                                   &packed_key, &packed_key_len);
        if (UCS_OK != ucs_status) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "ucp_rkey_pack failed for user segment %lu: %s",
                     i, ucs_status_string(ucs_status));
            ucp_mem_unmap(ctx->worker.ucp_context, mh);
            status = ucs_status_to_ucc_status(ucs_status);
            goto err;
        }

        team->mem_segs[i].va_base        = mp->segments[i].address;
        team->mem_segs[i].len            = mp->segments[i].len;
        team->mem_segs[i].mem_h          = (void *)mh;
        team->mem_segs[i].packed_key_len = packed_key_len;
        team->mem_segs[i].packed_key     = ucc_malloc(packed_key_len,
                                                      "packed_key_dup");
        if (!team->mem_segs[i].packed_key) {
            ucp_rkey_buffer_release(packed_key);
            ucp_mem_unmap(ctx->worker.ucp_context, mh);
            team->mem_segs[i].mem_h = NULL;
            goto err_no_memory;
        }
        memcpy(team->mem_segs[i].packed_key, packed_key, packed_key_len);
        ucp_rkey_buffer_release(packed_key);

        sbuf[i] = (uint64_t)packed_key_len;
    }

    team->n_mem_segs             = n_segs;
    team->mem_map_allgather_rbuf = rbuf;

    subset.map.type   = UCC_EP_MAP_FULL;
    subset.map.ep_num = (uint64_t)size;
    subset.myrank     = local_rank;

    status = ucc_tl_ucp_service_allgather(
        (ucc_base_team_t *)team, sbuf, rbuf,
        n_segs * sizeof(uint64_t), subset, &team->mem_map_task);
    if (status != UCC_OK && status != UCC_INPROGRESS) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "phase 1 service allgather failed: %s",
                 ucc_status_string(status));
        team->mem_map_allgather_rbuf = NULL;
        goto err;
    }

    /* sbuf is consumed synchronously by allgather_ring_start */
    ucc_free(sbuf);
    team->mem_map_phase = 1;
    return UCC_INPROGRESS;

err_no_memory:
    status = UCC_ERR_NO_MEMORY;
err:
    ucc_free(sbuf);
    ucc_free(rbuf);
    team->mem_map_allgather_rbuf = NULL;
    if (team->mem_segs) {
        for (i = 0; i < n_segs; i++) {
            if (team->mem_segs[i].mem_h) {
                ucp_mem_unmap(ctx->worker.ucp_context,
                              (ucp_mem_h)team->mem_segs[i].mem_h);
            }
            ucc_free(team->mem_segs[i].packed_key);
        }
        ucc_free(team->mem_segs);
        team->mem_segs = NULL;
    }
    ucc_free(team->team_remote_va);
    team->team_remote_va = NULL;
    ucc_free(team->team_remote_len);
    team->team_remote_len = NULL;
    ucc_free(team->team_rkeys);
    team->team_rkeys = NULL;
    team->n_mem_segs = 0;
    return status;
}

ucc_status_t ucc_tl_ucp_team_mem_map_data_exch(ucc_tl_ucp_team_t *team)
{
    uint64_t    *rbuf_p1     = (uint64_t *)team->mem_map_allgather_rbuf;
    ucc_rank_t   size        = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t   local_rank  = UCC_TL_TEAM_RANK(team);
    uint64_t     n_segs      = team->n_mem_segs;
    uint64_t     max_key_len = 0;
    uint8_t     *sbuf        = NULL;
    uint8_t     *rbuf        = NULL;
    uint8_t     *elem;
    uint64_t     klen;
    size_t       stride;
    ucc_subset_t subset;
    ucc_status_t status;
    ucc_rank_t   r;
    uint64_t     i;

    /* Caller (tl_ucp_team.c) only enters the mem-map path when both are > 0. */
    ucc_assert_always(n_segs > 0);
    ucc_assert_always(size > 0);

    /* Determine the max packed_key_len across all ranks and segments */
    for (r = 0; r < size; r++) {
        for (i = 0; i < n_segs; i++) {
            klen = rbuf_p1[(uint64_t)r * n_segs + i];
            if (klen > max_key_len) {
                max_key_len = klen;
            }
        }
    }

    ucc_tl_ucp_coll_finalize(team->mem_map_task);
    team->mem_map_task = NULL;
    ucc_free(rbuf_p1);
    team->mem_map_allgather_rbuf = NULL;

    team->mem_map_max_key_len = max_key_len;

    /*
     * Data exchange element layout (per segment per rank):
     *   [va: u64][len: u64][packed_key_len: u64][packed_key: max_key_len bytes]
     *
     * Pad stride to a multiple of 8 so uint64_t fields in subsequent elements
     * remain naturally aligned.
     */
    stride = (3 * sizeof(uint64_t) + (size_t)max_key_len + 7) & ~(size_t)7;

    sbuf = ucc_calloc(n_segs, stride, "mem_map_sbuf_p2");
    if (!sbuf) {
        return UCC_ERR_NO_MEMORY;
    }

    rbuf = ucc_malloc((size_t)size * n_segs * stride, "mem_map_rbuf_p2");
    if (!rbuf) {
        ucc_free(sbuf);
        return UCC_ERR_NO_MEMORY;
    }

    for (i = 0; i < n_segs; i++) {
        elem = sbuf + i * stride;
        *(uint64_t *)(elem + 0)  = (uint64_t)team->mem_segs[i].va_base;
        *(uint64_t *)(elem + 8)  = (uint64_t)team->mem_segs[i].len;
        *(uint64_t *)(elem + 16) = (uint64_t)team->mem_segs[i].packed_key_len;
        memcpy(elem + 24, team->mem_segs[i].packed_key,
               team->mem_segs[i].packed_key_len);
    }

    team->mem_map_allgather_rbuf = rbuf;

    subset.map.type   = UCC_EP_MAP_FULL;
    subset.map.ep_num = (uint64_t)size;
    subset.myrank     = local_rank;

    status = ucc_tl_ucp_service_allgather(
        (ucc_base_team_t *)team, sbuf, rbuf,
        n_segs * stride, subset, &team->mem_map_task);
    if (status != UCC_OK && status != UCC_INPROGRESS) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "phase 2 service allgather failed: %s",
                 ucc_status_string(status));
        ucc_free(sbuf);
        ucc_free(rbuf);
        team->mem_map_allgather_rbuf = NULL;
        return status;
    }

    ucc_free(sbuf);
    team->mem_map_phase = 2;
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_ucp_team_mem_map_finalize(ucc_tl_ucp_team_t *team)
{
    uint8_t     *rbuf   = (uint8_t *)team->mem_map_allgather_rbuf;
    ucc_rank_t   size   = UCC_TL_TEAM_SIZE(team);
    uint64_t     n_segs = team->n_mem_segs;
    uint8_t     *elem;
    size_t       stride;
    ucs_status_t ucs_status;
    ucc_status_t status;
    ucp_ep_h     ep;
    ucc_rank_t   r;
    uint64_t     s;

    stride = (3 * sizeof(uint64_t) +
              (size_t)team->mem_map_max_key_len + 7) & ~(size_t)7;

    for (r = 0; r < size; r++) {
        status = ucc_tl_ucp_get_ep(team, r, &ep);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "failed to get ep for rank %u: %s",
                     r, ucc_status_string(status));
            goto err;
        }

        for (s = 0; s < n_segs; s++) {
            elem = rbuf + ((uint64_t)r * n_segs + s) * stride;

            UCC_TL_UCP_TEAM_REMOTE_VA(team, r, s)  = *(uint64_t *)(elem + 0);
            UCC_TL_UCP_TEAM_REMOTE_LEN(team, r, s) =
                (size_t)(*(uint64_t *)(elem + 8));

            ucs_status = ucp_ep_rkey_unpack(
                ep, elem + 24, &UCC_TL_UCP_TEAM_RKEY(team, r, s));
            if (UCS_OK != ucs_status) {
                tl_error(UCC_TL_TEAM_LIB(team),
                         "ucp_ep_rkey_unpack failed for rank %u seg %lu: %s",
                         r, s, ucs_status_string(ucs_status));
                status = ucs_status_to_ucc_status(ucs_status);
                goto err;
            }
        }
    }

    status = UCC_OK;
err:
    /* Free scratch and finalize the allgather task on both success and error.
     * Already-unpacked rkeys remain in team_rkeys[] and are released by
     * ucc_tl_ucp_team_mem_map_destroy during team teardown. */
    ucc_free(rbuf);
    team->mem_map_allgather_rbuf = NULL;
    ucc_tl_ucp_coll_finalize(team->mem_map_task);
    team->mem_map_task  = NULL;
    team->mem_map_phase = 0;
    return status;
}

void ucc_tl_ucp_team_mem_map_destroy(ucc_tl_ucp_team_t *team)
{
    ucc_tl_ucp_context_t *ctx    = UCC_TL_UCP_TEAM_CTX(team);
    ucc_rank_t            size   = UCC_TL_TEAM_SIZE(team);
    uint64_t              n_segs = team->n_mem_segs;
    ucc_rank_t            r;
    uint64_t              s;

    /* Clean up any in-flight allgather task on error path.
     * If the task is still queued (UCC_INPROGRESS), remove it from the
     * progress queue's linked list before freeing to prevent the next
     * ucc_context_progress() from walking freed memory. */
    if (team->mem_map_task != NULL) {
        if (team->mem_map_task->status == UCC_INPROGRESS) {
            ucc_list_del(&team->mem_map_task->list_elem);
        }
        ucc_tl_ucp_coll_finalize(team->mem_map_task);
        team->mem_map_task = NULL;
    }
    if (team->mem_map_allgather_rbuf != NULL) {
        ucc_free(team->mem_map_allgather_rbuf);
        team->mem_map_allgather_rbuf = NULL;
    }

    /* Destroy unpacked rkeys */
    if (team->team_rkeys != NULL) {
        for (r = 0; r < size; r++) {
            for (s = 0; s < n_segs; s++) {
                if (UCC_TL_UCP_TEAM_RKEY(team, r, s) != NULL) {
                    ucp_rkey_destroy(UCC_TL_UCP_TEAM_RKEY(team, r, s));
                }
            }
        }
        ucc_free(team->team_rkeys);
        team->team_rkeys = NULL;
    }

    /* Unmap memory handles and free packed key copies */
    if (team->mem_segs != NULL) {
        for (s = 0; s < n_segs; s++) {
            if (team->mem_segs[s].mem_h) {
                ucp_mem_unmap(ctx->worker.ucp_context,
                              (ucp_mem_h)team->mem_segs[s].mem_h);
            }
            ucc_free(team->mem_segs[s].packed_key);
        }
        ucc_free(team->mem_segs);
        team->mem_segs = NULL;
    }

    if (team->team_remote_va != NULL) {
        ucc_free(team->team_remote_va);
        team->team_remote_va = NULL;
    }

    if (team->team_remote_len != NULL) {
        ucc_free(team->team_remote_len);
        team->team_remote_len = NULL;
    }

    team->n_mem_segs = 0;
}
