/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_team.h"
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

/*
 * This function is called when a onesided algorithm is used without mapped
 * memory by the user. A new memory handle should be created and returned. */
ucc_status_t ucc_tl_ucp_coll_dynamic_segment_init(ucc_coll_args_t     *coll_args,
                                                  ucc_mem_map_memh_t **src_memh,
                                                  ucc_mem_map_memh_t **dst_memh,
                                                  ucc_tl_ucp_task_t   *task)
{
    ucc_tl_ucp_team_t    *tl_team        = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx            = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_status_t          status         = UCC_OK;
    ucc_mem_map_memh_t   *src_memh_local = NULL;
    ucc_mem_map_memh_t   *dst_memh_local = NULL;

    /* if src_memh is NULL, create a new memory handle */
    if (*src_memh == NULL) {
        src_memh_local = ucc_calloc(1, sizeof(ucc_mem_map_memh_t), "src_memh");
        if (src_memh_local == NULL) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate src_memh");
            return UCC_ERR_NO_MEMORY;
        }
        src_memh_local->address = coll_args->src.info.buffer;
        src_memh_local->len =
            coll_args->src.info.count * sizeof(coll_args->src.info.datatype);
        src_memh_local->tl_h =
            ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "src_tl_h");
        if (src_memh_local->tl_h == NULL) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate src_tl_h");
            return UCC_ERR_NO_MEMORY;
        }
        strncpy(src_memh_local->tl_h->tl_name, "ucp",
                UCC_MEM_MAP_TL_NAME_LEN - 1);

        status = ucc_tl_ucp_mem_map(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                    src_memh_local, src_memh_local->tl_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to map memory");
            return status;
        }
        *src_memh = src_memh_local;
    }
    /* if dst_memh is NULL, create a new memory handle */
    if (*dst_memh == NULL) {
        dst_memh_local = ucc_calloc(1, sizeof(ucc_mem_map_memh_t), "dst_memh");
        if (dst_memh_local == NULL) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate dst_memh");
            return UCC_ERR_NO_MEMORY;
        }
        dst_memh_local->address = coll_args->dst.info.buffer;
        dst_memh_local->len =
            coll_args->dst.info.count * sizeof(coll_args->dst.info.datatype);
        dst_memh_local->tl_h =
            ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "dst_tl_h");
        if (dst_memh_local->tl_h == NULL) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate dst_tl_h");
            return UCC_ERR_NO_MEMORY;
        }
        strncpy(dst_memh_local->tl_h->tl_name, "ucp",
                UCC_MEM_MAP_TL_NAME_LEN - 1);

        status = ucc_tl_ucp_mem_map(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                    dst_memh_local, dst_memh_local->tl_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to map memory");
            return status;
        }
        *dst_memh = dst_memh_local;
    }
    /* update coll_args with new src and destination memory handles if the
     * src and dst memory handles are NULL */
    if (coll_args->src_memh.local_memh == NULL) {
        coll_args->src_memh.local_memh = *src_memh;
    }
    if (coll_args->dst_memh.local_memh == NULL) {
        coll_args->dst_memh.local_memh = *dst_memh;
    }
    return status;
}

ucc_status_t ucc_tl_ucp_coll_dynamic_segment_exchange(
    ucc_tl_ucp_task_t *task, ucc_mem_map_memh_t *src_memh,
    ucc_mem_map_memh_t *dst_memh, ucc_mem_map_memh_t **global_src_memh,
    ucc_mem_map_memh_t **global_dst_memh)
{
    ucc_tl_ucp_team_t    *tl_team   = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_team_t           *core_team = UCC_TL_CORE_TEAM(tl_team);
    ucc_subset_t subset = {.map = tl_team->ctx_map, .myrank = core_team->rank};
    ucc_service_coll_req_t *scoll_req;
    ucc_status_t            status          = UCC_OK;
    ucc_mem_map_memh_t     *src_memh_local  = NULL;
    ucc_mem_map_memh_t     *dst_memh_local  = NULL;
    void                   *src_pack_buffer = NULL;
    void                   *dst_pack_buffer = NULL;
    size_t                  src_pack_size   = 0;
    size_t                  dst_pack_size   = 0;
    size_t                  max_pack_size   = 0;
    void                   *exchange_buffer = NULL;
    void                   *global_buffer   = NULL;
    int                     i;

    if (src_memh && dst_memh) {
        return UCC_OK;
    }

    /* Pack src_memh if it exists */
    if (src_memh) {
        status =
            ucc_tl_ucp_memh_pack(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                 src_memh->tl_h, &src_pack_buffer);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to pack src memory handle");
            goto cleanup;
        }
        src_pack_size = src_memh->tl_h->packed_size;
    }

    /* Pack dst_memh if it exists */
    if (dst_memh) {
        status =
            ucc_tl_ucp_memh_pack(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                 dst_memh->tl_h, &dst_pack_buffer);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to pack dst memory handle");
            goto cleanup;
        }
        dst_pack_size = dst_memh->tl_h->packed_size;
    }

    /* Calculate total pack size for this rank */
    size_t local_pack_size = sizeof(size_t) * 2 + src_pack_size + dst_pack_size;

    /* Allgather to find the maximum pack size across all ranks */
    size_t *global_sizes =
        ucc_calloc(core_team->size, sizeof(size_t), "global sizes");
    if (!global_sizes) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate global sizes buffer");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_sizes;
    }

    status = ucc_service_allgather(core_team, &local_pack_size, global_sizes,
                                   sizeof(size_t), subset, &scoll_req);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to start service allgather for sizes");
        goto cleanup_sizes;
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
        goto cleanup_sizes;
    }
    ucc_service_coll_finalize(scoll_req);

    /* Find the maximum pack size */
    for (i = 0; i < core_team->size; i++) {
        if (global_sizes[i] > max_pack_size) {
            max_pack_size = global_sizes[i];
        }
    }
    src_memh_local = ucc_calloc(
        1, sizeof(ucc_mem_map_memh_t) + max_pack_size + sizeof(size_t) * 2,
        "src_memh_local");
    if (!src_memh_local) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate src_memh_local");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_sizes;
    }
    dst_memh_local = ucc_calloc(
        1, sizeof(ucc_mem_map_memh_t) + max_pack_size + sizeof(size_t) * 2,
        "dst_memh_local");
    if (!dst_memh_local) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate dst_memh_local");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_sizes;
    }
    src_memh_local->tl_h = ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "src_tl_h");
    if (!src_memh_local->tl_h) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate src_tl_h");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_sizes;
    }
    dst_memh_local->tl_h = ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "dst_tl_h");
    if (!dst_memh_local->tl_h) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate dst_tl_h");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_sizes;
    }
    /* Allocate exchange buffer */
    exchange_buffer = ucc_malloc(
        2 * (sizeof(ucc_mem_map_memh_t) + max_pack_size + sizeof(size_t) * 2),
        "exchange buffer");
    if (!exchange_buffer) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate exchange buffer");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_sizes;
    }

    /* Pack local data into exchange buffer */
    strncpy(src_memh_local->pack_buffer, "ucp", UCC_MEM_MAP_TL_NAME_LEN - 1);
    memcpy(PTR_OFFSET(src_memh_local, UCC_MEM_MAP_TL_NAME_LEN), &src_pack_size,
           sizeof(size_t));
    memcpy(PTR_OFFSET(src_memh_local, UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t)),
           src_pack_buffer, src_pack_size);
    ucc_free(src_pack_buffer);
    memcpy(PTR_OFFSET(src_memh_local,
                      UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t) + src_pack_size),
           src_memh->tl_h, sizeof(ucc_mem_map_tl_t));

    strncpy(dst_memh_local->pack_buffer, "ucp", UCC_MEM_MAP_TL_NAME_LEN - 1);
    memcpy(PTR_OFFSET(dst_memh_local, UCC_MEM_MAP_TL_NAME_LEN), &dst_pack_size,
           sizeof(size_t));
    memcpy(PTR_OFFSET(dst_memh_local, UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t)),
           dst_pack_buffer, dst_pack_size);
    ucc_free(dst_pack_buffer);
    memcpy(PTR_OFFSET(dst_memh_local,
                      UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t) + dst_pack_size),
           dst_memh->tl_h, sizeof(ucc_mem_map_tl_t));

    src_memh_local->mode    = UCC_MEM_MAP_MODE_EXPORT;
    dst_memh_local->mode    = UCC_MEM_MAP_MODE_EXPORT;
    src_memh_local->context = (ucc_context_h)&ctx->super.super;
    dst_memh_local->context = (ucc_context_h)&ctx->super.super;
    src_memh_local->address = src_memh->address;
    dst_memh_local->address = dst_memh->address;
    src_memh_local->len     = src_memh->len;
    dst_memh_local->len     = dst_memh->len;
    src_memh_local->num_tls = 1;
    dst_memh_local->num_tls = 1;

    if (src_pack_size > 0) {
        memcpy(exchange_buffer, src_memh_local, src_pack_size);
    }

    if (dst_pack_size > 0) {
        memcpy(PTR_OFFSET(exchange_buffer, max_pack_size), dst_memh_local,
               dst_pack_size);
    }

    /* Allocate global buffer for allgather */
    global_buffer =
        ucc_malloc((2 * max_pack_size) * core_team->size, "global buffer");
    if (!global_buffer) {
        tl_error(UCC_TASK_LIB(task), "failed to allocate global buffer");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_exchange;
    }

    /* Allgather the packed memory handles */
    status = ucc_service_allgather(core_team, exchange_buffer, global_buffer,
                                   2 * max_pack_size, subset, &scoll_req);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to start service allgather for memory handles");
        goto cleanup_global;
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
        goto cleanup_global;
    }
    ucc_service_coll_finalize(scoll_req);

    global_src_memh = ucc_calloc(core_team->size, sizeof(ucc_mem_map_memh_t),
                                 "global_src_memh");
    if (!global_src_memh) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to allocate global src memory handles");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_global;
    }
    global_dst_memh = ucc_calloc(core_team->size, sizeof(ucc_mem_map_memh_t),
                                 "global_dst_memh");
    if (!global_dst_memh) {
        tl_error(UCC_TASK_LIB(task),
                 "failed to allocate global memory handles");
        status = UCC_ERR_NO_MEMORY;
        goto cleanup_global;
    }
    /* Import memory handles for each rank using ucc_tl_ucp_mem_map */
    for (i = 0; i < core_team->size; i++) {
        //ucc_mem_map_memh_t *s_ptr = (ucc_mem_map_memh_t *)PTR_OFFSET(global_buffer, i * 2 * max_pack_size);
        //ucc_mem_map_memh_t *d_ptr = (ucc_mem_map_memh_t *)PTR_OFFSET(global_buffer, i * 2 * max_pack_size + max_pack_size);

        status =
            ucc_tl_ucp_mem_map(&ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                               global_src_memh[i], global_src_memh[i]->tl_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to import src memory handle for rank %d", i);
            goto cleanup_global;
        }

        status =
            ucc_tl_ucp_mem_map(&ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                               global_dst_memh[i], global_dst_memh[i]->tl_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to import dst memory handle for rank %d", i);
            goto cleanup_global;
        }
    }
    TASK_ARGS(task).src_memh.global_memh = (ucc_mem_map_mem_h *)global_src_memh;
    TASK_ARGS(task).dst_memh.global_memh = (ucc_mem_map_mem_h *)global_dst_memh;

cleanup_global:
    if (global_buffer) {
        ucc_free(global_buffer);
    }
cleanup_exchange:
    if (exchange_buffer) {
        ucc_free(exchange_buffer);
    }
cleanup_sizes:
    if (global_sizes) {
        ucc_free(global_sizes);
    }
cleanup:
    if (src_pack_buffer) {
        ucc_free(src_pack_buffer);
    }
    if (dst_pack_buffer) {
        ucc_free(dst_pack_buffer);
    }
    return status;
}

ucc_status_t ucc_tl_ucp_coll_dynamic_segment_finalize(
    ucc_tl_ucp_task_t *task, ucc_mem_map_memh_t **global_src_memh,
    ucc_mem_map_memh_t **global_dst_memh, int team_size)
{
    ucc_tl_ucp_team_t    *team   = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx    = UCC_TL_UCP_TEAM_CTX(team);
    ucc_status_t          status = UCC_OK;
    int                   i;

    /* Unmap global_src_memh array if it exists */
    if (global_src_memh) {
        for (i = 0; i < team_size; i++) {
            if (global_src_memh[i]) {
                status = ucc_tl_ucp_mem_unmap(&ctx->super.super,
                                              UCC_MEM_MAP_MODE_IMPORT,
                                              global_src_memh[i]->tl_h);
                if (status != UCC_OK) {
                    tl_error(
                        UCC_TASK_LIB(task),
                        "failed to unmap global src memory handle for rank %d",
                        i);
                }
                global_src_memh[i] = NULL;
            }
        }
    }
    /* Unmap global_dst_memh array if it exists */
    if (global_dst_memh) {
        for (i = 0; i < team_size; i++) {
            if (global_dst_memh[i]) {
                status = ucc_tl_ucp_mem_unmap(&ctx->super.super,
                                              UCC_MEM_MAP_MODE_IMPORT,
                                              global_dst_memh[i]->tl_h);
                if (status != UCC_OK) {
                    tl_error(
                        UCC_TASK_LIB(task),
                        "failed to unmap global dst memory handle for rank %d",
                        i);
                }
                global_dst_memh[i] = NULL;
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
