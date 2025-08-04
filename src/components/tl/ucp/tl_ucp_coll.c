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
#include "utils/ucc_coll_utils.h"
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

/* Constants for dynamic segment memory handle packing */
#define IS_SRC 1
#define IS_DST 0

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

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_ucp_put_task(task);
    return UCC_OK;
}

static inline ucc_status_t dynamic_segment_map_memh(ucc_mem_map_memh_t **memh,
                                                    ucc_coll_args_t     *coll_args,
                                                    int                  is_src,
                                                    ucc_tl_ucp_task_t   *task)
{
    ucc_tl_ucp_team_t        *tl_team = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t     *ctx     = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_status_t              status  = UCC_OK;
    ucc_mem_map_memh_t       *lmemh   = NULL;
    void                     *buffer;
    ucc_count_t               total_count;
    ucc_datatype_t            datatype;

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

    if (is_src) {
        total_count = coll_args->src.info.count;
        buffer      = coll_args->src.info.buffer;
        datatype    = coll_args->src.info.datatype;
    } else {
        total_count = coll_args->dst.info.count;
        buffer      = coll_args->dst.info.buffer;
        datatype    = coll_args->dst.info.datatype;
    }

    lmemh->address = buffer;
    lmemh->len     = total_count * ucc_dt_size(datatype);
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
 * This function initializes dynamic memory segments for onesided collectives.
 * It checks if user-provided memory handles are available, and if not,
 * creates and maps local memory handles for source and destination buffers.
 * These handles will be exchanged across ranks for remote memory access. */
UCC_TL_UCP_PROFILE_FUNC(ucc_status_t, ucc_tl_ucp_coll_dynamic_segment_init,
                        (coll_args, task), ucc_coll_args_t *coll_args,
                        ucc_tl_ucp_onesided_alg_type alg,
                        ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t    *tl_team = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx     = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_status_t          status  = UCC_OK;
    ucc_mem_map_memh_t   *src_memh;
    ucc_mem_map_memh_t   *dst_memh;

    if ((coll_args->coll_type == UCC_COLL_TYPE_ALLTOALLV) ||
        (coll_args->coll_type == UCC_COLL_TYPE_ALLGATHERV) ||
        (coll_args->coll_type == UCC_COLL_TYPE_GATHERV) ||
        (coll_args->coll_type == UCC_COLL_TYPE_REDUCE_SCATTERV) ||
        (coll_args->coll_type == UCC_COLL_TYPE_SCATTERV)) {
            tl_debug(UCC_TASK_LIB(task), "dynamic segments are not supported for %s",
                     ucc_coll_type_str(coll_args->coll_type));
        return UCC_ERR_NOT_SUPPORTED;
    }
    if ((coll_args->mask & UCC_COLL_ARGS_FIELD_FLAGS)) {
        if ((coll_args->flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            return UCC_OK;
        }
    } else if ((coll_args->mask & UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH) &&
               (coll_args->mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH)) {
        return UCC_OK;
    }
    status = dynamic_segment_map_memh(&src_memh, coll_args, IS_SRC, task);
    if (UCC_OK != status) {
        return status;
    }
    status = dynamic_segment_map_memh(&dst_memh, coll_args, IS_DST, task);
    if (UCC_OK != status) {
        status = ucc_tl_ucp_mem_unmap(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                      src_memh->tl_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to unmap src memory handle");
        }
        ucc_free(src_memh->tl_h);
        ucc_free(src_memh);
        return status;
    }
    memset(&task->dynamic_segments, 0, sizeof(task->dynamic_segments));
    task->dynamic_segments.src_local  = src_memh;
    task->dynamic_segments.dst_local  = dst_memh;
    task->dynamic_segments.src_global = NULL;
    task->dynamic_segments.dst_global = NULL;
    task->dynamic_segments.alg        = alg;
    task->flags                      |= UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG;
    task->dynamic_segments.exchange_step = UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_INIT;
    return status;
}

static inline ucc_status_t dynamic_segment_alloc_seg(ucc_mem_map_memh_t **memh,
                                                     size_t packed_size)
{
    ucc_mem_map_memh_t *lmemh;

    lmemh = ucc_calloc(1, sizeof(ucc_mem_map_memh_t) + packed_size, "packed_memh");
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

static inline void dynamic_segment_memh_pack(ucc_context_h              ctx,
                                             ucc_tl_ucp_dyn_seg_args_t *args,
                                             int                        is_src)
{
    ucc_mem_map_memh_t *memh;
    void               *pack_buffer;
    size_t              pack_size;
    void               *address;
    size_t              len;

    if (is_src) {
        pack_buffer = args->src_pack_buffer;
        pack_size   = args->src_pack_size;
        address     = args->src_memh_local->address;
        len         = args->src_memh_local->len;
        memh        = args->src_memh_pack;
    } else {
        pack_buffer = args->dst_pack_buffer;
        pack_size   = args->dst_pack_size;
        address     = args->dst_memh_local->address;
        len         = args->dst_memh_local->len;
        memh        = args->dst_memh_pack;
    }
    /* Pack local data into exchange buffer */
    strncpy(memh->pack_buffer, "ucp", UCC_MEM_MAP_TL_NAME_LEN - 1);
    memcpy(PTR_OFFSET(memh->pack_buffer, UCC_MEM_MAP_TL_NAME_LEN), &pack_size,
           sizeof(size_t));
    memcpy(
        PTR_OFFSET(memh->pack_buffer, UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t)),
        pack_buffer, pack_size);
    if (is_src == IS_DST) {
        memcpy(memh->tl_h, args->dst_memh_local->tl_h, sizeof(ucc_mem_map_tl_t));
    } else {
        memcpy(memh->tl_h, args->src_memh_local->tl_h, sizeof(ucc_mem_map_tl_t));
    }
    memh->mode    = UCC_MEM_MAP_MODE_EXPORT;
    memh->context = ctx;
    memh->address = address;
    memh->len     = len;
    memh->num_tls = 1;
}

static ucc_status_t
dynamic_segment_pack_memory_handles(ucc_tl_ucp_dyn_seg_args_t *args)
{
    ucc_tl_ucp_team_t    *tl_team = UCC_TL_UCP_TASK_TEAM(args->task);
    ucc_tl_ucp_context_t *ctx     = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_status_t          status;

    status = ucc_tl_ucp_memh_pack(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                  args->src_memh_local->tl_h,
                                  &args->src_pack_buffer);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(args->task), "failed to pack src memory handle");
        return status;
    }
    args->src_pack_size = args->src_memh_local->tl_h->packed_size;

    status = ucc_tl_ucp_memh_pack(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                  args->dst_memh_local->tl_h,
                                  &args->dst_pack_buffer);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(args->task), "failed to pack dst memory handle");
        ucc_free(args->src_pack_buffer);
        args->src_pack_buffer = NULL;
        return status;
    }
    args->dst_pack_size = args->dst_memh_local->tl_h->packed_size;
    return UCC_OK;
}

static ucc_status_t
dynamic_segment_calculate_sizes_start(ucc_tl_ucp_dyn_seg_args_t *args,
                                      ucc_service_coll_req_t   **scoll_req)
{
    ucc_tl_ucp_team_t *tl_team   = UCC_TL_UCP_TASK_TEAM(args->task);
    ucc_team_t        *core_team = UCC_TL_CORE_TEAM(tl_team);
    ucc_subset_t       subset;
    size_t            *global_sizes;
    ucc_status_t       status;
    size_t             local_pack_size;

    /* Use the TL team's context map */
    subset.map.type           = UCC_EP_MAP_STRIDED;
    subset.map.ep_num         = UCC_TL_TEAM_SIZE(tl_team);
    subset.map.strided.start  = 0;
    subset.map.strided.stride = 1;
    subset.myrank             = UCC_TL_TEAM_RANK(tl_team);

    /* Calculate total pack size for this rank - only destination handles
     * Use inner packed TL size; exchange_size will add outer TL header. */
    if (args->task->dynamic_segments.alg == UCC_TL_UCP_ALLTOALL_ONESIDED_GET) {
        local_pack_size = args->src_pack_size;
    } else {
        local_pack_size = args->dst_pack_size;
    }

    global_sizes = ucc_calloc(core_team->size, sizeof(size_t), "global sizes");
    if (!global_sizes) {
        tl_error(UCC_TASK_LIB(args->task),
                 "failed to allocate global sizes buffer");
        return UCC_ERR_NO_MEMORY;
    }
    args->global_sizes = global_sizes;

    status = ucc_service_allgather(core_team, &local_pack_size, global_sizes,
                                   sizeof(size_t), subset, scoll_req);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(args->task),
                 "failed to start service allgather for sizes");
        ucc_free(global_sizes);
        args->global_sizes = NULL;
        return status;
    }
    return UCC_OK;
}

static ucc_status_t
dynamic_segment_calculate_sizes_test(ucc_tl_ucp_dyn_seg_args_t *args,
                                     ucc_service_coll_req_t   **scoll_req)
{
    ucc_tl_ucp_team_t    *tl_team   = UCC_TL_UCP_TASK_TEAM(args->task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_team_t           *core_team = UCC_TL_CORE_TEAM(tl_team);
    ucc_status_t          status;
    int                   i;

    status = ucc_collective_test(&(*scoll_req)->task->super);
    if (status == UCC_INPROGRESS) {
        if (ctx->cfg.service_worker) {
            ucp_worker_progress(ctx->service_worker.ucp_worker);
        }
        return UCC_INPROGRESS;
    } else if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(args->task),
                 "failed during service allgather for sizes %s",
                 ucc_status_string(status));
        ucc_service_coll_finalize(*scoll_req);
        ucc_free(args->global_sizes);
        args->global_sizes = NULL;
        return status;
    }
    args->max_individual_pack_size = 0;
    for (i = 0; i < core_team->size; i++) {
        if (args->global_sizes[i] > args->max_individual_pack_size) {
            args->max_individual_pack_size = args->global_sizes[i];
        }
    }
    /* max_individual_pack_size already includes sizeof(size_t) * 2 (inner packed TL header)
     * Add outer TL header (TL name + size) to compute the total exchange size per rank. */
    args->exchange_size = sizeof(ucc_mem_map_memh_t) + UCC_MEM_MAP_TL_NAME_LEN +
                          sizeof(size_t) + args->max_individual_pack_size;
    if (args->exchange_size == 0) {
        ucc_service_coll_finalize(*scoll_req);
        ucc_free(args->global_sizes);
        args->global_sizes = NULL;
        return UCC_ERR_NO_MESSAGE;
    }
    ucc_service_coll_finalize(*scoll_req);
    ucc_free(args->global_sizes);
    args->global_sizes = NULL;
    *scoll_req         = NULL;
    return UCC_OK;
}

static ucc_status_t
dynamic_segment_allocate_buffers(ucc_tl_ucp_dyn_seg_args_t *args)
{
    ucc_tl_ucp_team_t *tl_team   = UCC_TL_UCP_TASK_TEAM(args->task);
    ucc_team_t        *core_team = UCC_TL_CORE_TEAM(tl_team);
    ucc_status_t       status;

    status = dynamic_segment_alloc_seg(
        &args->src_memh_pack, args->max_individual_pack_size +
                                  UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t));
    if (UCC_OK != status) {
        tl_error(UCC_TASK_LIB(args->task), "failed to allocate src_memh_pack");
        return status;
    }
    status = dynamic_segment_alloc_seg(
        &args->dst_memh_pack, args->max_individual_pack_size +
                                  UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t));
    if (UCC_OK != status) {
        tl_error(UCC_TASK_LIB(args->task), "failed to allocate dst_memh_pack");
        ucc_free(args->src_memh_pack);
        args->src_memh_pack = NULL;
        return status;
    }

    args->exchange_buffer =
        ucc_calloc(1, args->exchange_size, "exchange buffer");
    if (!args->exchange_buffer) {
        tl_error(UCC_TASK_LIB(args->task),
                 "failed to allocate exchange buffer");
        ucc_free(args->src_memh_pack);
        args->src_memh_pack = NULL;
        ucc_free(args->dst_memh_pack);
        args->dst_memh_pack = NULL;
        return UCC_ERR_NO_MEMORY;
    }

    args->task->dynamic_segments.global_buffer =
        ucc_calloc(core_team->size, args->exchange_size, "global buffer");
    if (!args->task->dynamic_segments.global_buffer) {
        tl_error(UCC_TASK_LIB(args->task), "failed to allocate global buffer");
        ucc_free(args->exchange_buffer);
        args->exchange_buffer = NULL;
        ucc_free(args->src_memh_pack);
        args->src_memh_pack = NULL;
        ucc_free(args->dst_memh_pack);
        args->dst_memh_pack = NULL;
        return UCC_ERR_NO_MEMORY;
    }
    return UCC_OK;
}

static ucc_status_t
dynamic_segment_pack_and_exchange_data_start(ucc_tl_ucp_dyn_seg_args_t *args,
                                             ucc_service_coll_req_t **scoll_req)
{
    ucc_tl_ucp_team_t    *tl_team   = UCC_TL_UCP_TASK_TEAM(args->task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_team_t           *core_team = UCC_TL_CORE_TEAM(tl_team);
    ucc_subset_t          subset;
    ucc_status_t          status;
    size_t                copy_size;

    /* Use the TL team's context map */
    subset.map.type           = UCC_EP_MAP_STRIDED;
    subset.map.ep_num         = UCC_TL_TEAM_SIZE(tl_team);
    subset.map.strided.start  = 0;
    subset.map.strided.stride = 1;
    subset.myrank             = UCC_TL_TEAM_RANK(tl_team);

    dynamic_segment_memh_pack((ucc_context_h)&ctx->super.super, args, IS_DST);
    /* Include outer TL header (name + size) in the copy size in addition to memh struct */
    copy_size = sizeof(ucc_mem_map_memh_t) + UCC_MEM_MAP_TL_NAME_LEN +
                sizeof(size_t);
    if (args->task->dynamic_segments.alg == UCC_TL_UCP_ALLTOALL_ONESIDED_GET) {
        /* Only pack destination handles for one-sided operations */
        dynamic_segment_memh_pack((ucc_context_h)&ctx->super.super, args, IS_SRC);
        copy_size += args->src_pack_size;
        memcpy(args->exchange_buffer, args->src_memh_pack, copy_size);
    } else {
        copy_size += args->dst_pack_size;
        memcpy(args->exchange_buffer, args->dst_memh_pack, copy_size);
    }
    /* Allgather the packed memory handles */
    status = ucc_service_allgather(core_team, args->exchange_buffer,
                                   args->task->dynamic_segments.global_buffer,
                                   args->exchange_size, subset, scoll_req);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(args->task),
                 "failed to start service allgather for memory handles");
        return status;
    }
    return UCC_OK;
}

static ucc_status_t
dynamic_segment_pack_and_exchange_data_test(ucc_tl_ucp_dyn_seg_args_t *args,
                                            ucc_service_coll_req_t **scoll_req)
{
    ucc_tl_ucp_team_t    *tl_team = UCC_TL_UCP_TASK_TEAM(args->task);
    ucc_tl_ucp_context_t *ctx     = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_status_t          status;

    status = ucc_collective_test(&(*scoll_req)->task->super);
    if (status < UCC_OK) {
        tl_error(UCC_TASK_LIB(args->task),
                 "failed during service allgather for memory handles %s",
                 ucc_status_string(status));
        ucc_service_coll_finalize(*scoll_req);
        return status;
    } else if (status == UCC_INPROGRESS) {
        if (ctx->cfg.service_worker) {
            ucp_worker_progress(ctx->service_worker.ucp_worker);
        }
        return UCC_INPROGRESS;
    }
    ucc_service_coll_finalize(*scoll_req);
    *scoll_req = NULL;
    return UCC_OK;
}

static ucc_status_t
dynamic_segment_import_memory_handles(ucc_tl_ucp_dyn_seg_args_t *args)
{
    ucc_tl_ucp_team_t    *tl_team   = UCC_TL_UCP_TASK_TEAM(args->task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucc_team_t           *core_team = UCC_TL_CORE_TEAM(tl_team);
    ucc_mem_map_memh_t  **global;
    size_t                offset;
    ucc_status_t          status;
    int                   i;
    int                   j;

    /* Only allocate destination handles for one-sided operations */
    global =
        ucc_calloc(core_team->size, sizeof(ucc_mem_map_memh_t *), "global");
    if (!global) {
        tl_error(UCC_TASK_LIB(args->task),
                 "failed to allocate global memory handles");
        return UCC_ERR_NO_MEMORY;
    }

    /* Import memory handles for each rank using ucc_tl_ucp_mem_map */
    for (i = 0; i < UCC_TL_TEAM_SIZE(tl_team); i++) {
        /* Each rank's data in global buffer contains only:
           - dst_memh_pack: sizeof(ucc_mem_map_memh_t) + max_individual_pack_size + sizeof(size_t) * 2
        */
        offset    = i * args->exchange_size;
        global[i] =
        (ucc_mem_map_memh_t *)PTR_OFFSET(
                args->task->dynamic_segments.global_buffer, offset);
        global[i]->tl_h =
            ucc_calloc(1, sizeof(ucc_mem_map_tl_t), "global tl_h");
        if (!global[i]->tl_h) {
            tl_error(UCC_TASK_LIB(args->task),
                    "failed to allocate global tl handles");
            status = UCC_ERR_NO_MEMORY;
            goto out;
        }

        status = ucc_tl_ucp_mem_map(
            &ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
            global[i],
            global[i]->tl_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(args->task),
                     "failed to import dst memory handle for rank %d ", i);
            goto out;
        }
    }

    if (args->task->dynamic_segments.alg == UCC_TL_UCP_ALLTOALL_ONESIDED_GET) {
        args->task->dynamic_segments.src_global = global;
        args->task->dynamic_segments.dst_global = NULL;
    } else {
        args->task->dynamic_segments.src_global = NULL;
        args->task->dynamic_segments.dst_global = global;
    }

    return UCC_OK;
out:
    for (j = 0; j < i; j++) {
        if (global[j]->tl_h) {
            /* we need to unmap */
            ucc_tl_ucp_mem_unmap(&ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                                 global[j]->tl_h);
        }
        ucc_free(global[j]->tl_h);
    }
    ucc_free(global);
    return status;
}

static void dynamic_segment_cleanup_buffers(ucc_tl_ucp_dyn_seg_args_t *args)
{
    if (!args) {
        return;
    }

    if (args->src_pack_buffer) {
        ucc_free(args->src_pack_buffer);
        args->src_pack_buffer = NULL;
    }
    if (args->dst_pack_buffer) {
        ucc_free(args->dst_pack_buffer);
        args->dst_pack_buffer = NULL;
    }
    if (args->src_memh_pack) {
        if (args->src_memh_pack->tl_h) {
            ucc_free(args->src_memh_pack->tl_h);
            args->src_memh_pack->tl_h = NULL;
        }
        ucc_free(args->src_memh_pack);
        args->src_memh_pack = NULL;
    }
    if (args->dst_memh_pack) {
        if (args->dst_memh_pack->tl_h) {
            ucc_free(args->dst_memh_pack->tl_h);
            args->dst_memh_pack->tl_h = NULL;
        }
        ucc_free(args->dst_memh_pack);
        args->dst_memh_pack = NULL;
    }
    if (args->exchange_buffer) {
        ucc_free(args->exchange_buffer);
        args->exchange_buffer = NULL;
    }
    if (args->global_sizes) {
        ucc_free(args->global_sizes);
        args->global_sizes = NULL;
    }
}

UCC_TL_UCP_PROFILE_FUNC(ucc_status_t,
                        ucc_tl_ucp_coll_dynamic_segment_exchange_nb, (task),
                        ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *tl_team   = UCC_TL_UCP_TASK_TEAM(task);
    ucc_team_t        *core_team = UCC_TL_CORE_TEAM(tl_team);
    ucc_status_t       status    = UCC_OK;

    if (core_team->size == 0) {
        tl_error(UCC_TASK_LIB(task),
                 "unable to exchange segments with team size of 0");
        return UCC_ERR_INVALID_PARAM;
    }

    /* Initialize on first call */
    if (task->dynamic_segments.exchange_args == NULL) {
        task->dynamic_segments.exchange_args =
            ucc_calloc(1, sizeof(ucc_tl_ucp_dyn_seg_args_t), "exchange_args");
        if (!task->dynamic_segments.exchange_args) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate exchange_args");
            return UCC_ERR_NO_MEMORY;
        }
        task->dynamic_segments.exchange_args->task = task;
        task->dynamic_segments.exchange_args->src_memh_local =
            task->dynamic_segments.src_local;
        task->dynamic_segments.exchange_args->dst_memh_local =
            task->dynamic_segments.dst_local;
        task->dynamic_segments.exchange_step   = 0;
        task->dynamic_segments.exchange_status = UCC_OK;
        task->dynamic_segments.scoll_req_sizes = NULL;
        task->dynamic_segments.scoll_req_data  = NULL;
    }
    switch (task->dynamic_segments.exchange_step) {
    case UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_INIT:
        status = dynamic_segment_pack_memory_handles(
            task->dynamic_segments.exchange_args);
        if (status != UCC_OK) {
            goto err_cleanup;
        }
        task->dynamic_segments.exchange_step = UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_SIZE_TEST;
        return UCC_INPROGRESS;

    case UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_SIZE_TEST:
        if (task->dynamic_segments.exchange_args->global_sizes == NULL) {

            /* First call - start the allgather */
            status = dynamic_segment_calculate_sizes_start(
                task->dynamic_segments.exchange_args,
                &task->dynamic_segments.scoll_req_sizes);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to start allgather %s",
                         ucc_status_string(status));
                goto err_cleanup;
            }
            return UCC_INPROGRESS;
        } else {
            /* Subsequent calls - test for completion */
            status = dynamic_segment_calculate_sizes_test(
                task->dynamic_segments.exchange_args,
                &task->dynamic_segments.scoll_req_sizes);
            if (status == UCC_INPROGRESS) {
                return UCC_INPROGRESS;
            }
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to test allgather");
                goto err_cleanup;
            }
            task->dynamic_segments.exchange_step =
                UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_DATA_ALLOC;
            return UCC_INPROGRESS;
        }

    case UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_DATA_ALLOC:
        status = dynamic_segment_allocate_buffers(
            task->dynamic_segments.exchange_args);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate buffers");
            goto err_cleanup;
        }
        task->dynamic_segments.exchange_step = UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_DATA_START;
        return UCC_INPROGRESS;

    case UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_DATA_START:
        if (task->dynamic_segments.scoll_req_data == NULL) {
            /* First call - start the allgather */
            status = dynamic_segment_pack_and_exchange_data_start(
                task->dynamic_segments.exchange_args,
                &task->dynamic_segments.scoll_req_data);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to start data exchange %s",
                         ucc_status_string(status));
                goto err_cleanup_global;
            }
            return UCC_INPROGRESS;
        } else {
            /* Subsequent calls - test for completion */
            status = dynamic_segment_pack_and_exchange_data_test(
                task->dynamic_segments.exchange_args,
                &task->dynamic_segments.scoll_req_data);
            if (status == UCC_INPROGRESS) {
                return UCC_INPROGRESS;
            }
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to test data exchange");
                goto err_cleanup_global;
            }
        }
    }
    status = dynamic_segment_import_memory_handles(
        task->dynamic_segments.exchange_args);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task), "failed to import memory handles");
        goto err_cleanup_global;
    }

    /* Cleanup and complete */
    dynamic_segment_cleanup_buffers(task->dynamic_segments.exchange_args);
    ucc_free(task->dynamic_segments.exchange_args);
    task->dynamic_segments.exchange_args = NULL;
    task->dynamic_segments.exchange_step = UCC_TL_UCP_DYN_SEG_EXCHANGE_STEP_COMPLETE;
    return UCC_OK;

err_cleanup_global:
    if (task->dynamic_segments.global_buffer) {
        ucc_free(task->dynamic_segments.global_buffer);
        task->dynamic_segments.global_buffer = NULL;
    }
err_cleanup:
    if (task->dynamic_segments.exchange_args) {
        dynamic_segment_cleanup_buffers(task->dynamic_segments.exchange_args);
        ucc_free(task->dynamic_segments.exchange_args);
        task->dynamic_segments.exchange_args = NULL;
    }
    return status;
}

UCC_TL_UCP_PROFILE_FUNC(ucc_status_t, ucc_tl_ucp_coll_dynamic_segment_finalize,
                        (task), ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t    *team      = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx       = UCC_TL_UCP_TEAM_CTX(team);
    ucc_status_t          status    = UCC_OK;
    size_t                team_size = UCC_TL_TEAM_SIZE(team);
    int                   i;

    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_USE_DYN_SEG)) {
        return UCC_OK;
    }
    if (task->dynamic_segments.src_global) {
        for (i = 0; i < team_size; i++) {
            if (task->dynamic_segments.src_global[i] &&
                task->dynamic_segments.src_global[i]->tl_h) {
                status = ucc_tl_ucp_mem_unmap(
                    &ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                    task->dynamic_segments.src_global[i]->tl_h);
                if (status != UCC_OK) {
                    tl_error(UCC_TASK_LIB(task),
                             "failed to unmap src global memory handle for "
                             "rank %d",
                             i);
                }
                ucc_free(task->dynamic_segments.src_global[i]->tl_h);
                task->dynamic_segments.src_global[i]->tl_h = NULL;
                task->dynamic_segments.src_global[i] = NULL;
            }
        }
        ucc_free(task->dynamic_segments.src_global);
        task->dynamic_segments.src_global = NULL;
    }
    if (task->dynamic_segments.dst_global) {
        for (i = 0; i < team_size; i++) {
            if (task->dynamic_segments.dst_global[i] &&
                task->dynamic_segments.dst_global[i]->tl_h) {
                status = ucc_tl_ucp_mem_unmap(
                    &ctx->super.super, UCC_MEM_MAP_MODE_IMPORT,
                    task->dynamic_segments.dst_global[i]->tl_h);
                if (status != UCC_OK) {
                    tl_error(UCC_TASK_LIB(task),
                             "failed to unmap dst global memory handle for "
                             "rank %d",
                             i);
                }
                ucc_free(task->dynamic_segments.dst_global[i]->tl_h);
                task->dynamic_segments.dst_global[i]->tl_h = NULL;
                task->dynamic_segments.dst_global[i] = NULL;
            }
        }
        ucc_free(task->dynamic_segments.dst_global);
        task->dynamic_segments.dst_global = NULL;
    }
    /* Free global buffer */
    if (task->dynamic_segments.global_buffer) {
        ucc_free(task->dynamic_segments.global_buffer);
        task->dynamic_segments.global_buffer = NULL;
    }
    if (task->dynamic_segments.src_local) {
        if (task->dynamic_segments.src_local->tl_h) {
            status =
                ucc_tl_ucp_mem_unmap(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                     task->dynamic_segments.src_local->tl_h);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task),
                         "failed to unmap src local memory handle");
            }
            ucc_free(task->dynamic_segments.src_local->tl_h);
            task->dynamic_segments.src_local->tl_h = NULL;
        }
        ucc_free(task->dynamic_segments.src_local);
        task->dynamic_segments.src_local = NULL;
    }
    if (task->dynamic_segments.dst_local) {
        if (task->dynamic_segments.dst_local->tl_h) {
            status =
                ucc_tl_ucp_mem_unmap(&ctx->super.super, UCC_MEM_MAP_MODE_EXPORT,
                                     task->dynamic_segments.dst_local->tl_h);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task),
                         "failed to unmap dst local memory handle");
            }
            ucc_free(task->dynamic_segments.dst_local->tl_h);
            task->dynamic_segments.dst_local->tl_h = NULL;
        }
        ucc_free(task->dynamic_segments.dst_local);
        task->dynamic_segments.dst_local = NULL;
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
