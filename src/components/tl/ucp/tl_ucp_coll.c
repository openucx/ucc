/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_team.h"
#include "utils/ucc_math.h"
#include "barrier/barrier.h"
#include "alltoall/alltoall.h"
#include "alltoallv/alltoallv.h"
#include "allreduce/allreduce.h"
#include "allgather/allgather.h"
#include "allgatherv/allgatherv.h"
#include "reduce_scatter/reduce_scatter.h"
#include "reduce_scatterv/reduce_scatterv.h"
#include "bcast/bcast.h"
#include "reduce/reduce.h"
#include "gather/gather.h"
#include "gatherv/gatherv.h"
#include "fanin/fanin.h"
#include "fanout/fanout.h"
#include "scatterv/scatterv.h"

const ucc_tl_ucp_default_alg_desc_t
    ucc_tl_ucp_default_alg_descs[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR] = {
        {
            .select_str = NULL,
            .str_get_fn = ucc_tl_ucp_allgather_score_str_get
        },
        {
            .select_str = UCC_TL_UCP_ALLGATHERV_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = NULL,
            .str_get_fn = ucc_tl_ucp_alltoall_score_str_get
        },
        {
            .select_str = UCC_TL_UCP_ALLREDUCE_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_BCAST_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_REDUCE_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_REDUCE_SCATTERV_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        },
        {
            .select_str = UCC_TL_UCP_ALLTOALLV_DEFAULT_ALG_SELECT_STR,
            .str_get_fn = NULL
        }
};

ucc_status_t ucc_tl_ucp_team_default_score_str_alloc(ucc_tl_ucp_team_t *team,
    char *default_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR])
{
    ucc_status_t st = UCC_OK;
    int i;

    for (i = 0; i < UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR; i++) {
        if (ucc_tl_ucp_default_alg_descs[i].select_str) {
            default_select_str[i] = strdup(ucc_tl_ucp_default_alg_descs[i].select_str);
        } else {
            default_select_str[i] = ucc_tl_ucp_default_alg_descs[i].str_get_fn(team);
        }
        if (!default_select_str[i]) {
            st = UCC_ERR_NO_MEMORY;
            goto exit;
        }

    }

exit:
    return st;
}

void ucc_tl_ucp_team_default_score_str_free(
    char *default_select_str[UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR])
{
    int i;

    for (i = 0; i < UCC_TL_UCP_N_DEFAULT_ALG_SELECT_STR; i++) {
        ucc_free(default_select_str[i]);
    }
}

void ucc_tl_ucp_send_completion_cb_st(void *request, ucs_status_t status,
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in send completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ++task->tagged.send_completed;
    ucp_request_free(request);
}

void ucc_tl_ucp_send_completion_cb_mt(void *request, ucs_status_t status,
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in send completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ucc_atomic_add32(&task->tagged.send_completed, 1);
    ucp_request_free(request);
}

void ucc_tl_ucp_put_completion_cb(void *request, ucs_status_t status,
                                  void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in put completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->onesided.put_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_get_completion_cb(void *request, ucs_status_t status,
                                  void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in get completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    task->onesided.get_completed++;
    ucp_request_free(request);
}

void ucc_tl_ucp_recv_completion_cb_mt(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info, /* NOLINT */
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ucc_atomic_add32(&task->tagged.recv_completed, 1);
    ucp_request_free(request);
}

void ucc_tl_ucp_recv_completion_cb_st(void *request, ucs_status_t status,
                                      const ucp_tag_recv_info_t *info, /* NOLINT */
                                      void *user_data)
{
    ucc_tl_ucp_task_t *task = (ucc_tl_ucp_task_t *)user_data;
    if (ucc_unlikely(UCS_OK != status)) {
        tl_error(UCC_TASK_LIB(task), "failure in recv completion %s",
                 ucs_status_string(status));
        task->super.status = ucs_status_to_ucc_status(status);
    }
    ++task->tagged.recv_completed;
    ucp_request_free(request);
}

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_ucp_put_task(task);
    return UCC_OK;
}

static inline ucc_status_t dynamic_segment_map_memh(ucc_mem_map_memh_t **memh,
                                                    ucc_coll_buffer_info_t *info,
                                                    ucc_tl_ucp_task_t   *task)
{
    ucc_tl_ucp_team_t    *tl_team = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx     = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_status_t          status  = UCC_OK;
    ucc_mem_map_memh_t   *lmemh   = NULL;

    lmemh = ucc_calloc(1, sizeof(ucc_mem_map_memh_t), "dyn_memh");
    if (lmemh == NULL) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate memh");
        status = UCC_ERR_NO_MEMORY;
        goto out;
    }
    lmemh->tl_h = ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "dyn_tlh");
    if (!lmemh->tl_h) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate memh");
        ucc_free(lmemh);
        status = UCC_ERR_NO_MEMORY;
        goto out;
    }
    lmemh->address = info->buffer;
    lmemh->len     = info->count * ucc_dt_size(info->datatype);
    lmemh->num_tls = 1;  /* Only one transport layer (UCP) */
    strncpy(lmemh->tl_h->tl_name, "ucp", UCC_MEM_MAP_TL_NAME_LEN - 1);
    status = ucc_tl_ucp_mem_map(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                lmemh, lmemh->tl_h);
    if (UCC_OK != status) {
        tl_error(UCC_TASK_LIB(task), "failed to map memory for memh");
        ucc_free(lmemh->tl_h);
        ucc_free(lmemh);
        goto out;
    }
    *memh = lmemh;
out:
    return status;
}

/*
 * This function is called when a onesided algorithm is used without mapped
 * memory by the user. A new memory handle should be created and returned. */
UCC_TL_UCP_PROFILE_FUNC(ucc_status_t, ucc_tl_ucp_coll_dynamic_segment_init, (coll_args, task),
                        ucc_coll_args_t *coll_args, ucc_tl_ucp_task_t *task)
{
    ucc_status_t          status = UCC_OK;
    ucc_mem_map_memh_t   *src_memh;
    ucc_mem_map_memh_t   *dst_memh;

    /* TODO check if this is needed */
    if ((coll_args->mask & UCC_COLL_ARGS_FIELD_FLAGS)) {
        if ((coll_args->flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            return UCC_OK;
        }
    } else if ((coll_args->mask & UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH) &&
               (coll_args->mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH)) {
        return UCC_OK;
    }

    /* TODO: add support for info_v */
    status = dynamic_segment_map_memh(&src_memh, &coll_args->src.info, task);
    if (UCC_OK != status) {
        return status;
    }
    status = dynamic_segment_map_memh(&dst_memh, &coll_args->dst.info, task);
    if (UCC_OK != status) {
        ucc_free(src_memh->tl_h);
        ucc_free(src_memh);
        return status;
    }
    /* TODO: add gwb */

    task->dynamic_segments.src_local = src_memh;
    task->dynamic_segments.dst_local = dst_memh;
    task->flags |= UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG;
    return status;
}

static inline ucc_status_t dynamic_segment_alloc_packed_seg(ucc_mem_map_memh_t **memh,
                                                            size_t packed_size)
{
    ucc_mem_map_memh_t *lmemh;

    lmemh = ucc_calloc(1, sizeof(ucc_mem_map_memh_t) + packed_size + sizeof(size_t) * 2, "packed_memh");
    if (!lmemh) {
        return UCC_ERR_NO_MEMORY;
    }

    lmemh->tl_h = ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "packed_tl_h");
    if (!lmemh->tl_h) {
        ucc_free(lmemh);
        return UCC_ERR_NO_MEMORY;
    }

    *memh = lmemh;
    return UCC_OK;
}

static inline void dynamic_segment_memh_pack(ucc_mem_map_memh_t *memh,
                                             void               *pack_buffer,
                                             size_t              pack_size,
                                             ucc_context_h       ctx,
                                             void               *address,
                                             size_t              len)
{
    /* Pack local data into exchange buffer */
    strncpy(memh->pack_buffer, "ucp", UCC_MEM_MAP_TL_NAME_LEN - 1);
    memcpy(PTR_OFFSET(memh->pack_buffer, UCC_MEM_MAP_TL_NAME_LEN), &pack_size,
           sizeof(size_t));
    memcpy(PTR_OFFSET(memh->pack_buffer, UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t)),
           pack_buffer, pack_size);
/*    memcpy(PTR_OFFSET(memh->pack_buffer,
                      UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t) + pack_size),
           memh->tl_h, sizeof(ucc_mem_map_tl_t));*/
    memh->mode    = UCC_MEM_MAP_MODE_EXPORT;
    memh->context = ctx;
    memh->address = address;
    memh->len     = len;
    memh->num_tls = 1;
}

UCC_TL_UCP_PROFILE_FUNC(ucc_status_t, ucc_tl_ucp_coll_dynamic_segment_exchange, (task),
                        ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t      *tl_team         = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t   *ctx             = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_team_t             *core_team       = UCC_TL_CORE_TEAM(tl_team);
    ucc_subset_t            subset          = {.map    = tl_team->ctx_map,
                                               .myrank = core_team->rank};
    ucc_status_t            status          = UCC_OK;
    ucc_mem_map_memh_t     *src_memh_local  = task->dynamic_segments.src_local;
    ucc_mem_map_memh_t     *dst_memh_local  = task->dynamic_segments.dst_local;
    ucc_mem_map_memh_t     *src_memh_pack   = task->dynamic_segments.src_local;
    ucc_mem_map_memh_t     *dst_memh_pack   = task->dynamic_segments.dst_local;
    void                   *src_pack_buffer = NULL;
    void                   *dst_pack_buffer = NULL;
    size_t                  src_pack_size   = 0;
    size_t                  dst_pack_size   = 0;
    size_t                  max_individual_pack_size   = 0;
    size_t                  exchange_size   = 0;
    void                   *exchange_buffer = NULL;
    void                   *global_buffer   = NULL;
    int                     i               = 0;
    size_t                  local_pack_size = 0;
    size_t                 *global_sizes    = NULL;
    ucc_service_coll_req_t *scoll_req;

    if (core_team->size == 0) {
        tl_error(UCC_TASK_LIB(task), "unable to exchange segments with team size of 0");
        return UCC_ERR_INVALID_PARAM;
    }
    status =
        ucc_tl_ucp_memh_pack(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                             src_memh_local->tl_h, &src_pack_buffer);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task), "failed to pack src memory handle");
        goto err_cleanup;
    }
    src_pack_size = src_memh_local->tl_h->packed_size;

    status =
        ucc_tl_ucp_memh_pack(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                             dst_memh_local->tl_h, &dst_pack_buffer);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task), "failed to pack dst memory handle");
        goto err_cleanup;
    }
    dst_pack_size = dst_memh_local->tl_h->packed_size;

    /* Calculate total pack size for this rank */
    local_pack_size = ucc_max(src_pack_size, dst_pack_size) + sizeof(size_t) * 2;

    /* Allgather to find the maximum pack size across all ranks */
    global_sizes =
        ucc_calloc(core_team->size, sizeof(size_t), "global sizes");
    if (!global_sizes) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate global sizes buffer");
        status = UCC_ERR_NO_MEMORY;
        goto err_cleanup_sizes;
    }

    status = ucc_service_allgather(core_team, &local_pack_size, global_sizes,
                                   sizeof(size_t), subset, &scoll_req);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to start service allgather for sizes");
        goto err_cleanup_sizes;
    }

    /* Wait for the allgather to complete */
    while (UCC_INPROGRESS == (status = ucc_service_coll_test(scoll_req))) {
        /* Progress the service worker if available */
        if (ctx->cfg.service_worker) {
            ucp_worker_progress(ctx->service_worker.ucp_worker);
        }
    }
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task),
                 "failed during service allgather for sizes");
        ucc_service_coll_finalize(scoll_req);
        goto err_cleanup_sizes;
    }
    ucc_service_coll_finalize(scoll_req);

    /* Find the maximum pack size */
    for (i = 0; i < core_team->size; i++) {
        if (global_sizes[i] > max_individual_pack_size) {
            max_individual_pack_size = global_sizes[i];
        }
    }
    exchange_size =
        2 * (sizeof(ucc_mem_map_memh_t) + max_individual_pack_size + sizeof(size_t) * 2);
    if (exchange_size == 0) {
        status = UCC_ERR_NO_MESSAGE;
        goto err_cleanup_sizes;
    }

    status = dynamic_segment_alloc_packed_seg(&src_memh_pack, max_individual_pack_size);
    if (UCC_OK != status) {
       tl_error(UCC_TASK_LIB(task), "failed to allocate src_memh_pack");
       goto err_cleanup_sizes;
    }

    status = dynamic_segment_alloc_packed_seg(&dst_memh_pack, max_individual_pack_size);
    if (UCC_OK != status) {
       tl_error(UCC_TASK_LIB(task), "failed to allocate dst_memh_pack");
       goto err_cleanup_sizes;
    }

    /* Allocate exchange buffer */
    exchange_buffer = ucc_malloc(
        exchange_size,
        "exchange buffer");
    if (!exchange_buffer) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate exchange buffer");
        status = UCC_ERR_NO_MEMORY;
        goto err_cleanup_sizes;
    }

    /* Pack local data into exchange buffer */
    dynamic_segment_memh_pack(src_memh_pack, src_pack_buffer, src_pack_size,
                              (ucc_context_h)&ctx->super.super, src_memh_local->address,
                              src_memh_local->len);
    dynamic_segment_memh_pack(dst_memh_pack, dst_pack_buffer, dst_pack_size,
                              (ucc_context_h)&ctx->super.super, dst_memh_local->address,
                              dst_memh_local->len);
    ucc_free(src_pack_buffer);
    src_pack_buffer = NULL;
    ucc_free(dst_pack_buffer);
    dst_pack_buffer = NULL;

    memcpy(exchange_buffer, src_memh_pack, src_pack_size + 2 * sizeof(size_t) + sizeof(ucc_mem_map_memh_t));
    memcpy(PTR_OFFSET(exchange_buffer, sizeof(ucc_mem_map_memh_t) + max_individual_pack_size + sizeof(size_t) * 2), dst_memh_pack,
           dst_pack_size + 2 * sizeof(size_t) + sizeof(ucc_mem_map_memh_t));

    ucc_free(src_memh_pack->tl_h);
    ucc_free(src_memh_pack);
    ucc_free(dst_memh_pack->tl_h);
    ucc_free(dst_memh_pack);
    src_memh_pack = NULL;
    dst_memh_pack = NULL;
    if (core_team->size == 0) {
        tl_error(UCC_TASK_LIB(task), "unable to exchange segments with team size of 0");
        status = UCC_ERR_INVALID_PARAM;
        goto err_cleanup_exchange;
    }

    /* Allocate global buffer for allgather */
    global_buffer =
        ucc_malloc(exchange_size * core_team->size, "global buffer");
    if (!global_buffer) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate global buffer");
        status = UCC_ERR_NO_MEMORY;
        goto err_cleanup_exchange;
    }

    /* Allgather the packed memory handles */
    status = ucc_service_allgather(core_team, exchange_buffer, global_buffer,
                                   exchange_size, subset, &scoll_req);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to start service allgather for memory handles");
        goto err_cleanup_global;
    }

    /* Wait for the allgather to complete */
    while (UCC_INPROGRESS == (status = ucc_service_coll_test(scoll_req))) {
        /* Progress the service worker if available */
        if (ctx->cfg.service_worker) {
            ucp_worker_progress(ctx->service_worker.ucp_worker);
        }
    }
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task),
                 "failed during service allgather for memory handles");
        ucc_service_coll_finalize(scoll_req);
        goto err_cleanup_global;
    }
    ucc_service_coll_finalize(scoll_req);

    task->dynamic_segments.src_global = ucc_calloc(core_team->size, sizeof(ucc_mem_map_memh_t *),
                                                    "src_global");
    if (!task->dynamic_segments.src_global) {
         tl_error(UCC_TASK_LIB(task),
                 "failed to allocate global src memory handles");
        status = UCC_ERR_NO_MEMORY;
        goto err_cleanup_global;
    }
    task->dynamic_segments.dst_global = ucc_calloc(core_team->size, sizeof(ucc_mem_map_memh_t *),
                                                    "dst_global");
    if (!task->dynamic_segments.dst_global) {
         tl_error(UCC_TASK_LIB(task),
                 "failed to allocate global dst memory handles");
        status = UCC_ERR_NO_MEMORY;
        goto err_cleanup_global;
    }

    /* Import memory handles for each rank using ucc_tl_ucp_mem_map */
    for (i = 0; i < core_team->size; i++) {
        /* Each rank's data in global buffer contains:
           - src_memh_pack: sizeof(ucc_mem_map_memh_t) + max_individual_pack_size + sizeof(size_t) * 2
           - dst_memh_pack: sizeof(ucc_mem_map_memh_t) + max_individual_pack_size + sizeof(size_t) * 2
        */
        size_t src_offset = i * exchange_size;
        size_t dst_offset = i * exchange_size + exchange_size / 2;

        task->dynamic_segments.src_global[i] = (ucc_mem_map_memh_t *)PTR_OFFSET(global_buffer, src_offset);
        task->dynamic_segments.dst_global[i] = (ucc_mem_map_memh_t *)PTR_OFFSET(global_buffer, dst_offset);

        task->dynamic_segments.src_global[i]->tl_h = ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "global tl_h");
        task->dynamic_segments.dst_global[i]->tl_h = ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "global tl_h");

        status =
            ucc_tl_ucp_mem_map(&ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                               task->dynamic_segments.src_global[i],
                               task->dynamic_segments.src_global[i]->tl_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to import src memory handle for rank %d", i);
            goto err_cleanup_global;
        }

        status =
            ucc_tl_ucp_mem_map(&ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                               task->dynamic_segments.dst_global[i],
                               task->dynamic_segments.dst_global[i]->tl_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to import dst memory handle for rank %d", i);
            goto err_cleanup_global;
        }
    }

    if (exchange_buffer) {
        ucc_free(exchange_buffer);
    }
    if (src_memh_local) {
        if (src_memh_local->tl_h) {
            ucc_free(src_memh_local->tl_h);
        }
        ucc_free(src_memh_local);
    }
    if (dst_memh_local) {
        if (dst_memh_local->tl_h) {
            ucc_free(dst_memh_local->tl_h);
        }
        ucc_free(dst_memh_local);
    }

    if (global_sizes) {
        ucc_free(global_sizes);
    }
    if (src_pack_buffer) {
        ucc_free(src_pack_buffer);
    }
    if (dst_pack_buffer) {
        ucc_free(dst_pack_buffer);
    }
    return status;

err_cleanup_global:
    for (--i; i > 0; i--) {
        if (task->dynamic_segments.src_global && task->dynamic_segments.src_global[i]) {
            ucc_tl_ucp_mem_unmap(&ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                                task->dynamic_segments.src_global[i]->tl_h);
            ucc_free(task->dynamic_segments.src_global[i]->tl_h->tl_data);
        }
        if (task->dynamic_segments.dst_global && task->dynamic_segments.dst_global[i]) {
            ucc_tl_ucp_mem_unmap(&ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                            task->dynamic_segments.dst_global[i]->tl_h);
            ucc_free(task->dynamic_segments.dst_global[i]->tl_h->tl_data);
        }
    }
    if (task->dynamic_segments.src_global) {
        ucc_free(task->dynamic_segments.src_global);
    }
    if (task->dynamic_segments.dst_global) {
        ucc_free(task->dynamic_segments.dst_global);
    }
    if (global_buffer) {
        ucc_free(global_buffer);
    }
err_cleanup_exchange:
    if (exchange_buffer) {
        ucc_free(exchange_buffer);
    }
err_cleanup_sizes:
    if (src_memh_pack) {
        if (src_memh_pack->tl_h) {
            ucc_free(src_memh_pack->tl_h);
        }
        ucc_free(src_memh_pack);
    }
    if (dst_memh_pack) {
        if (dst_memh_pack->tl_h) {
            ucc_free(dst_memh_pack->tl_h);
        }
        ucc_free(dst_memh_pack);
    }
    if (global_sizes) {
        ucc_free(global_sizes);
    }
err_cleanup:
    if (src_pack_buffer) {
        ucc_free(src_pack_buffer);
    }
    if (dst_pack_buffer) {
        ucc_free(dst_pack_buffer);
    }
    return status;
}

UCC_TL_UCP_PROFILE_FUNC(ucc_status_t, ucc_tl_ucp_coll_dynamic_segment_finalize, (task),
                        ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t    *team      = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(team);
    ucc_status_t          status    = UCC_OK;
    size_t                team_size = UCC_TL_TEAM_SIZE(team);
    int                   i;

    /* Unmap global_src_memh array if it exists */
    if (task->dynamic_segments.src_global) {
        for (i = 0; i < team_size; i++) {
            if (task->dynamic_segments.src_global[i]) {
                status = ucc_tl_ucp_mem_unmap(&ctx->super.super,
                                              UCC_MEM_MAP_MODE_IMPORT,
                                              task->dynamic_segments.src_global[i]->tl_h);
                if (status != UCC_OK) {
                    tl_error(
                        UCC_TASK_LIB(task),
                        "failed to unmap global src memory handle for rank %d",
                        i);
                }
                task->dynamic_segments.src_global[i] = NULL;
            }
        }
    }
    /* Unmap global_dst_memh array if it exists */
    if (task->dynamic_segments.dst_global) {
        for (i = 0; i < team_size; i++) {
            if (task->dynamic_segments.dst_global[i]) {
                status = ucc_tl_ucp_mem_unmap(&ctx->super.super,
                                              UCC_MEM_MAP_MODE_IMPORT,
                                              task->dynamic_segments.dst_global[i]->tl_h);
                if (status != UCC_OK) {
                    tl_error(
                        UCC_TASK_LIB(task),
                        "failed to unmap global dst memory handle for rank %d",
                        i);
                }
                task->dynamic_segments.dst_global[i] = NULL;
            }
        }
    }
    return status;
}

ucc_status_t ucc_tl_ucp_coll_init(ucc_base_coll_args_t *coll_args,
                                  ucc_base_team_t *team,
                                  ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_task_t    *task = ucc_tl_ucp_init_task(coll_args, team);
    ucc_status_t          status;

    switch (coll_args->args.coll_type) {
    case UCC_COLL_TYPE_BARRIER:
        status = ucc_tl_ucp_barrier_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        status = ucc_tl_ucp_alltoall_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        status = ucc_tl_ucp_alltoallv_init(task);
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        status = ucc_tl_ucp_allreduce_init(task);
        break;
    case UCC_COLL_TYPE_ALLGATHER:
        status = ucc_tl_ucp_allgather_init(task);
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
        status = ucc_tl_ucp_allgatherv_init(task);
        break;
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_ucp_bcast_init(task);
        break;
    case UCC_COLL_TYPE_REDUCE:
        status = ucc_tl_ucp_reduce_init(task);
        break;
    case UCC_COLL_TYPE_GATHER:
        status = ucc_tl_ucp_gather_init(task);
        break;
    case UCC_COLL_TYPE_FANIN:
        status = ucc_tl_ucp_fanin_init(task);
        break;
    case UCC_COLL_TYPE_FANOUT:
        status = ucc_tl_ucp_fanout_init(task);
        break;
    case UCC_COLL_TYPE_SCATTERV:
        status = ucc_tl_ucp_scatterv_init(task);
        break;
    case UCC_COLL_TYPE_GATHERV:
        status = ucc_tl_ucp_gatherv_init(task);
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    tl_trace(team->context->lib, "init coll req %p", task);
    *task_h = &task->super;
    return status;
}

static inline int alg_id_from_str(ucc_coll_type_t coll_type, const char *str)
{
    switch (coll_type) {
    case UCC_COLL_TYPE_ALLGATHER:
        return ucc_tl_ucp_allgather_alg_from_str(str);
    case UCC_COLL_TYPE_ALLGATHERV:
        return ucc_tl_ucp_allgatherv_alg_from_str(str);
    case UCC_COLL_TYPE_ALLREDUCE:
        return ucc_tl_ucp_allreduce_alg_from_str(str);
    case UCC_COLL_TYPE_ALLTOALL:
        return ucc_tl_ucp_alltoall_alg_from_str(str);
    case UCC_COLL_TYPE_ALLTOALLV:
        return ucc_tl_ucp_alltoallv_alg_from_str(str);
    case UCC_COLL_TYPE_BCAST:
        return ucc_tl_ucp_bcast_alg_from_str(str);
    case UCC_COLL_TYPE_REDUCE:
        return ucc_tl_ucp_reduce_alg_from_str(str);
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        return ucc_tl_ucp_reduce_scatter_alg_from_str(str);
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        return ucc_tl_ucp_reduce_scatterv_alg_from_str(str);
    default:
        break;
    }
    return -1;
}

ucc_status_t ucc_tl_ucp_alg_id_to_init(int alg_id, const char *alg_id_str,
                                       ucc_coll_type_t   coll_type,
                                       ucc_memory_type_t mem_type, //NOLINT
                                       ucc_base_coll_init_fn_t *init)
{
    ucc_status_t status = UCC_OK;

    if (alg_id_str) {
        alg_id = alg_id_from_str(coll_type, alg_id_str);
    }

    switch (coll_type) {
    case UCC_COLL_TYPE_ALLGATHER:
        switch (alg_id) {
        case UCC_TL_UCP_ALLGATHER_ALG_KNOMIAL:
            *init = ucc_tl_ucp_allgather_knomial_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_RING:
            *init = ucc_tl_ucp_allgather_ring_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_NEIGHBOR:
            *init = ucc_tl_ucp_allgather_neighbor_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_BRUCK:
            *init = ucc_tl_ucp_allgather_bruck_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_SPARBIT:
            *init = ucc_tl_ucp_allgather_sparbit_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_LINEAR:
            *init = ucc_tl_ucp_allgather_linear_init;
            break;
        case UCC_TL_UCP_ALLGATHER_ALG_LINEAR_BATCHED:
            *init = ucc_tl_ucp_allgather_linear_batched_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
        switch (alg_id) {
        case UCC_TL_UCP_ALLGATHERV_ALG_KNOMIAL:
            *init = ucc_tl_ucp_allgatherv_knomial_init;
            break;
        case UCC_TL_UCP_ALLGATHERV_ALG_RING:
            *init = ucc_tl_ucp_allgatherv_ring_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        switch (alg_id) {
        case UCC_TL_UCP_ALLREDUCE_ALG_KNOMIAL:
            *init = ucc_tl_ucp_allreduce_knomial_init;
            break;
        case UCC_TL_UCP_ALLREDUCE_ALG_SRA_KNOMIAL:
            *init = ucc_tl_ucp_allreduce_sra_knomial_init;
            break;
        case UCC_TL_UCP_ALLREDUCE_ALG_DBT:
            *init = ucc_tl_ucp_allreduce_dbt_init;
            break;
        case UCC_TL_UCP_ALLREDUCE_ALG_SLIDING_WINDOW:
            *init = ucc_tl_ucp_allreduce_sliding_window_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_BCAST:
        switch (alg_id) {
        case UCC_TL_UCP_BCAST_ALG_KNOMIAL:
            *init = ucc_tl_ucp_bcast_knomial_init;
            break;
        case UCC_TL_UCP_BCAST_ALG_SAG_KNOMIAL:
            *init = ucc_tl_ucp_bcast_sag_knomial_init;
            break;
        case UCC_TL_UCP_BCAST_ALG_DBT:
            *init = ucc_tl_ucp_bcast_dbt_init;
            break;
        default:
           status = UCC_ERR_INVALID_PARAM;
           break;
        };
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        switch (alg_id) {
        case UCC_TL_UCP_ALLTOALL_ALG_PAIRWISE:
            *init = ucc_tl_ucp_alltoall_pairwise_init;
            break;
        case UCC_TL_UCP_ALLTOALL_ALG_BRUCK:
            *init = ucc_tl_ucp_alltoall_bruck_init;
            break;
        case UCC_TL_UCP_ALLTOALL_ALG_ONESIDED:
            *init = ucc_tl_ucp_alltoall_onesided_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        switch (alg_id) {
        case UCC_TL_UCP_ALLTOALLV_ALG_PAIRWISE:
            *init = ucc_tl_ucp_alltoallv_pairwise_init;
            break;
        case UCC_TL_UCP_ALLTOALLV_ALG_HYBRID:
            *init = ucc_tl_ucp_alltoallv_hybrid_init;
            break;
        case UCC_TL_UCP_ALLTOALLV_ALG_ONESIDED:
            *init = ucc_tl_ucp_alltoallv_onesided_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_REDUCE:
        switch (alg_id) {
        case UCC_TL_UCP_REDUCE_ALG_KNOMIAL:
            *init = ucc_tl_ucp_reduce_knomial_init;
            break;
        case UCC_TL_UCP_REDUCE_ALG_DBT:
            *init = ucc_tl_ucp_reduce_dbt_init;
            break;
        case UCC_TL_UCP_REDUCE_ALG_SRG:
            *init = ucc_tl_ucp_reduce_srg_knomial_init;
            break;
        default:
           status = UCC_ERR_INVALID_PARAM;
           break;
        };
        break;
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        switch (alg_id) {
        case UCC_TL_UCP_REDUCE_SCATTER_ALG_RING:
            *init = ucc_tl_ucp_reduce_scatter_ring_init;
            break;
        case UCC_TL_UCP_REDUCE_SCATTER_ALG_KNOMIAL:
            *init = ucc_tl_ucp_reduce_scatter_knomial_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    case UCC_COLL_TYPE_REDUCE_SCATTERV:
        switch (alg_id) {
        case UCC_TL_UCP_REDUCE_SCATTERV_ALG_RING:
            *init = ucc_tl_ucp_reduce_scatterv_ring_init;
            break;
        default:
            status = UCC_ERR_INVALID_PARAM;
            break;
        };
        break;
    default:
        status = UCC_ERR_NOT_SUPPORTED;
        break;
    }
    return status;
}
