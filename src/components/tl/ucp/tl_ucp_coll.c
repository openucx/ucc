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

void ucc_tl_ucp_send_completion_cb(void *request, ucs_status_t status,
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

void ucc_tl_ucp_recv_completion_cb(void *request, ucs_status_t status,
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

ucc_status_t ucc_tl_ucp_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_ucp_put_task(task);
    return UCC_OK;
}

static void ucc_tl_ucp_pack_data(ucc_tl_ucp_context_t *ctx, void *pack)
{
    uint64_t  nsegs          = ctx->n_dynrinfo_segs;
    uint64_t  offset         = 0;
    size_t    section_offset = sizeof(uint64_t) * nsegs;
    void     *keys;
    uint64_t *rvas;
    uint64_t *lens;
    uint64_t *key_sizes;
    int       i;

    /* pack into one data object in following order: */
    /* rva, len, pack sizes, packed keys */
    rvas      = pack;
    lens      = PTR_OFFSET(pack, section_offset);
    key_sizes = PTR_OFFSET(pack, (section_offset * 2));
    keys      = PTR_OFFSET(pack, (section_offset * 3));

    for (i = 0; i < nsegs; i++) {
        rvas[i]      = (uint64_t)ctx->dynamic_remote_info[i].va_base;
        lens[i]      = ctx->dynamic_remote_info[i].len;
        key_sizes[i] = ctx->dynamic_remote_info[i].packed_key_len;
        memcpy(PTR_OFFSET(keys, offset), ctx->dynamic_remote_info[i].packed_key,
               ctx->dynamic_remote_info[i].packed_key_len);
        offset += ctx->dynamic_remote_info[i].packed_key_len;
    }
}

ucc_status_t ucc_tl_ucp_memmap_segment(ucc_tl_ucp_task_t *task,
                                       ucc_mem_map_t *map, int segid)
{
    ucc_tl_ucp_team_t    *tl_team = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *tl_ctx  = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucs_status_t          ucs_status;
    ucp_mem_map_params_t  mmap_params;
    ucp_mem_h             mh;

    /* map the memory */
    if (map->resource != NULL) {
        mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
        mmap_params.exported_memh_buffer               = map->resource;
        tl_ctx->dynamic_remote_info[segid].packed_memh = map->resource;
    } else {
        mmap_params.field_mask =
            UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        mmap_params.address                            = map->address;
        mmap_params.length                             = map->len;
        tl_ctx->dynamic_remote_info[segid].packed_memh = NULL;
    }
    /* map exported memory handle */
    ucs_status = ucp_mem_map(tl_ctx->worker.ucp_context, &mmap_params, &mh);
    if (ucs_status == UCS_ERR_UNREACHABLE) {
        tl_error(tl_ctx->super.super.lib, "exported memh is unsupported");
        return UCC_ERR_MEM_MAP_FAILURE;
    } else if (ucs_status < UCS_OK) {
        tl_error(tl_ctx->super.super.lib,
                 "ucp_mem_map failed with error code: %d", ucs_status);
        return UCC_ERR_MEM_MAP_FAILURE;
    }
    /* generate rkeys / packed keys */
    tl_ctx->dynamic_remote_info[segid].va_base = map->address;
    tl_ctx->dynamic_remote_info[segid].len     = map->len;
    tl_ctx->dynamic_remote_info[segid].mem_h   = mh;
    ucs_status =
        ucp_rkey_pack(tl_ctx->worker.ucp_context, mh,
                      &tl_ctx->dynamic_remote_info[segid].packed_key,
                      &tl_ctx->dynamic_remote_info[segid].packed_key_len);
    if (UCS_OK != ucs_status) {
        tl_error(tl_ctx->super.super.lib,
                 "failed to pack UCP key with error code: %d", ucs_status);
        return ucs_status_to_ucc_status(ucs_status);
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_coll_dynamic_segment_init(ucc_coll_args_t   *coll_args,
                                                  ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t    *tl_team    = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx        = UCC_TL_UCP_TEAM_CTX(tl_team);
    int                   i          = 0;
    uint64_t              need_map   = 0x7;
    ucc_mem_map_t        *maps       = coll_args->mem_map.segments;
    ucc_mem_map_t        *seg_maps   = NULL;
    size_t                n_segments = 3;
    ucc_status_t          status;


    /* check if src, dst, global work in ctx mapped segments */
    for (i = 0; i < ctx->n_rinfo_segs && n_segments > 0; i++) {
        uint64_t base = (uint64_t)ctx->remote_info[i].va_base;
        uint64_t end = (uint64_t)(base + ctx->remote_info[i].len);
        if ((uint64_t)coll_args->src.info.buffer >= base &&
            (uint64_t)coll_args->src.info.buffer < end) {
            // found it
            need_map ^= 1;
            --n_segments;
        }
        if ((uint64_t)coll_args->dst.info.buffer >= base &&
            (uint64_t)coll_args->dst.info.buffer < end) {
            // found it
            need_map ^= 2;
            --n_segments;
        }

        if ((uint64_t)coll_args->global_work_buffer >= base &&
            (uint64_t)coll_args->global_work_buffer < end) {
            // found it
            need_map ^= 4;
            --n_segments;
        }

        if (n_segments == 0) {
            break;
        }
    }

    /* add any valid segments */
    if (n_segments > 0) {
        int index = 0;
        seg_maps = ucc_calloc(n_segments, sizeof(ucc_mem_map_t));
        if (!seg_maps) {
            return UCC_ERR_NO_MEMORY;
        }

        if (need_map & 0x1) {
            seg_maps[index].address = coll_args->src.info.buffer;
            seg_maps[index].len = (coll_args->src.info.count) *
                          ucc_dt_size(coll_args->src.info.datatype);
            seg_maps[index++].resource = NULL;
        }
        if (need_map & 0x2) {
            seg_maps[index].address = coll_args->dst.info.buffer;
            seg_maps[index].len = (coll_args->dst.info.count) *
                          ucc_dt_size(coll_args->dst.info.datatype);
            seg_maps[index++].resource = NULL;
        }
        if (need_map & 0x4) {
            seg_maps[index].address = coll_args->global_work_buffer;
            seg_maps[index].len = (ONESIDED_SYNC_SIZE + ONESIDED_REDUCE_SIZE) * sizeof(long);
            seg_maps[index++].resource = NULL;
        }
    }

    if (n_segments > 0) {
        ctx->dynamic_remote_info =
            ucc_calloc(n_segments, sizeof(ucc_tl_ucp_remote_info_t), "dynamic remote info");
        /* map memory and fill in local segment information */
        for (i = 0; i < n_segments; i++) {
            status = ucc_tl_ucp_memmap_segment(task, &seg_maps[i], i);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to memory map a segment");
                goto failed_memory_map;
            }
            ++ctx->n_dynrinfo_segs;
        }
        for (i = 0; i < coll_args->mem_map.n_segments; i++) {
            status = ucc_tl_ucp_memmap_segment(task, &maps[i], i + n_segments);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to memory map a segment");
                goto failed_memory_map;
            }
            ++ctx->n_dynrinfo_segs;
        }
        if (n_segments) {
            free(seg_maps);
        }
    }
    return UCC_OK;
failed_memory_map:
    for (i = 0; i < ctx->n_dynrinfo_segs; i++) {
        if (ctx->dynamic_remote_info[i].mem_h) {
            ucp_mem_unmap(ctx->worker.ucp_context,
                          ctx->dynamic_remote_info[i].mem_h);
        }
        if (ctx->dynamic_remote_info[i].packed_key) {
            ucp_rkey_buffer_release(ctx->dynamic_remote_info[i].packed_key);
        }
        if (ctx->dynamic_remote_info[i].packed_memh) {
            ucp_rkey_buffer_release(ctx->dynamic_remote_info[i].packed_memh);
        }
    }
    ctx->n_dynrinfo_segs = 0;
    if (n_segments) {
        ucc_free(seg_maps);
    }
    return status;
}

ucc_status_t ucc_tl_ucp_coll_dynamic_segment_exchange(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t    *tl_team       = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx           = UCC_TL_UCP_TEAM_CTX(tl_team);
    int                   i             = 0;
    size_t                seg_pack_size = 0;
    uint64_t             *global_size   = NULL;
    void                 *ex_buffer     = NULL;
    ucc_status_t          status;

    if (ctx->n_dynrinfo_segs) {
        size_t       team_size = UCC_TL_TEAM_SIZE(tl_team);
        ucc_team_t  *core_team = UCC_TL_CORE_TEAM(UCC_TL_UCP_TASK_TEAM(task));
        ucc_subset_t subset    = {.map    = tl_team->ctx_map,
                                  .myrank = core_team->rank};
        ucc_service_coll_req_t *scoll_req;

        for (i = 0; i < ctx->n_dynrinfo_segs; i++) {
            seg_pack_size += sizeof(uint64_t) * 3 +
                             ctx->dynamic_remote_info[i].packed_key_len;
        }

        global_size = ucc_calloc(core_team->size, sizeof(uint64_t));
        if (!global_size) {
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            return UCC_ERR_NO_MEMORY;
        }

        /* allgather on the new segments size */
        status = ucc_service_allgather(core_team, &seg_pack_size, global_size,
                                       sizeof(uint64_t), subset, &scoll_req);
        if (status < UCC_OK) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to perform a service allgather");
            goto failed_size_exch;
        }
        while (UCC_INPROGRESS == (status = ucc_service_coll_test(scoll_req))) {
        }
        if (status < UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed on the allgather");
            ucc_service_coll_finalize(scoll_req);
            goto failed_size_exch;
        }
        ucc_service_coll_finalize(scoll_req);
        for (i = 0; i < core_team->size; i++) {
            if (global_size[i] > seg_pack_size) {
                seg_pack_size = global_size[i];
            }
        }
        ucc_free(global_size);
        global_size = NULL;

        /* pack the dynamic_remote_info segments */
        ex_buffer = ucc_malloc(seg_pack_size, "ex pack size");
        if (!ex_buffer) {
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            status = UCC_ERR_NO_MEMORY;
            goto failed_data_exch;
        }
        ucc_tl_ucp_pack_data(ctx, ex_buffer);

        ctx->dyn_seg_buf = ucc_calloc(1, team_size * seg_pack_size, "dyn buff");
        if (!ctx->dyn_seg_buf) {
            status = UCC_ERR_NO_MEMORY;
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            goto failed_data_exch;
        }

        /* allgather on the new segments (packed) */
        status = ucc_service_allgather(core_team, ex_buffer, ctx->dyn_seg_buf,
                                       seg_pack_size, subset, &scoll_req);
        if (status < UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed on the allgather");
            goto failed_data_exch;
        }
        while (UCC_INPROGRESS == (status = ucc_service_coll_test(scoll_req))) {
        }
        if (status < UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed on the allgather");
            ucc_service_coll_finalize(scoll_req);
            goto failed_data_exch;
        }
        /* done with allgather */
        ucc_service_coll_finalize(scoll_req);
        ctx->dyn_rkeys =
            ucc_calloc(1, team_size * sizeof(ucp_rkey_h) * ctx->n_dynrinfo_segs,
                       "dyn rkeys");
        if (!ctx->dyn_rkeys) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate space for keys");
            status = UCC_ERR_NO_MEMORY;
            goto failed_data_exch;
        }
        ctx->dyn_seg_size = seg_pack_size;
        ucc_free(ex_buffer);
    }
    return UCC_OK;
failed_data_exch:
    if (ctx->dyn_seg_buf) {
        ucc_free(ctx->dyn_seg_buf);
        ctx->dyn_seg_buf = NULL;
    }
    if (ex_buffer) {
        ucc_free(ex_buffer);
    }
failed_size_exch:
    if (!global_size) {
        ucc_free(global_size);
    }
    return status;
}

void ucc_tl_ucp_coll_dynamic_segment_finalize(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t    *tl_team = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *ctx     = UCC_TL_UCP_TEAM_CTX(tl_team);
    int                   i       = 0;
    int                   j       = 0;
    /* free library resources, unmap user resources */
    if (ctx->dyn_seg_buf) {
        /* unmap and release packed buffers */
        for (i = 0; i < ctx->n_dynrinfo_segs; i++) {
            if (ctx->dynamic_remote_info[i].mem_h) {
                ucp_mem_unmap(ctx->worker.ucp_context,
                              ctx->dynamic_remote_info[i].mem_h);
            }
            if (ctx->dynamic_remote_info[i].packed_key) {
                ucp_rkey_buffer_release(ctx->dynamic_remote_info[i].packed_key);
            }
            if (ctx->dynamic_remote_info[i].packed_memh) {
                ucp_rkey_buffer_release(
                    ctx->dynamic_remote_info[i].packed_memh);
            }
        }
        /* destroy rkeys */
        for (i = 0; i < UCC_TL_TEAM_SIZE(tl_team); i++) {
            for (j = 0; j < ctx->n_dynrinfo_segs; j++) {
                if (UCC_TL_UCP_DYN_REMOTE_RKEY(ctx, i, j)) {
                    ucp_rkey_destroy(UCC_TL_UCP_DYN_REMOTE_RKEY(ctx, i, j));
                }
            }
        }
        ucc_free(ctx->dynamic_remote_info);
        ucc_free(ctx->dyn_rkeys);
        ucc_free(ctx->dyn_seg_buf);

        ctx->dynamic_remote_info = NULL;
        ctx->dyn_rkeys           = NULL;
        ctx->dyn_seg_buf         = NULL;
        ctx->dyn_seg_size        = 0;
        ctx->n_dynrinfo_segs     = 0;
    }
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
