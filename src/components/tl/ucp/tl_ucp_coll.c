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

static void ucc_tl_ucp_pack_data(ucc_tl_ucp_context_t *ctx, int starting_index,
                                 void *pack)
{
    uint64_t  nsegs          = ctx->n_dynrinfo_segs - starting_index;
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
        int index    = i + starting_index;
        rvas[i]      = (uint64_t)ctx->dynamic_remote_info[index].va_base;
        lens[i]      = ctx->dynamic_remote_info[index].len;
        key_sizes[i] = ctx->dynamic_remote_info[index].packed_key_len;
        memcpy(PTR_OFFSET(keys, offset),
               ctx->dynamic_remote_info[index].packed_key,
               ctx->dynamic_remote_info[index].packed_key_len);
        offset += ctx->dynamic_remote_info[index].packed_key_len;
    }
}

ucc_status_t ucc_tl_ucp_memmap_append_segment(ucc_tl_ucp_task_t *task,
                                              ucc_mem_map_t *map, int segid)
{
    ucc_tl_ucp_team_t    *tl_team = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_context_t *tl_ctx  = UCC_TL_UCP_TEAM_CTX(tl_team);
    ucs_status_t          ucs_status;
    ucp_mem_map_params_t  mmap_params;
    ucp_mem_h             mh;

    // map the memory
    if (map->resource != NULL) {
        mmap_params.field_mask = UCP_MEM_MAP_PARAM_FIELD_EXPORTED_MEMH_BUFFER;
        mmap_params.exported_memh_buffer = map->resource;

        ucs_status = ucp_mem_map(tl_ctx->worker.ucp_context, &mmap_params, &mh);
        if (ucs_status == UCS_ERR_UNREACHABLE) {
            tl_error(tl_ctx->super.super.lib, "exported memh is unsupported");
            return ucs_status_to_ucc_status(ucs_status);
        } else if (ucs_status < UCS_OK) {
            tl_error(tl_ctx->super.super.lib,
                     "ucp_mem_map failed with error code: %d", ucs_status);
            return ucs_status_to_ucc_status(ucs_status);
        }
        /* generate rkeys / packed keys */

        tl_ctx->dynamic_remote_info[segid].va_base     = map->address;
        tl_ctx->dynamic_remote_info[segid].len         = map->len;
        tl_ctx->dynamic_remote_info[segid].mem_h       = mh;
        tl_ctx->dynamic_remote_info[segid].packed_memh = map->resource;
        ucs_status =
            ucp_rkey_pack(tl_ctx->worker.ucp_context, mh,
                          &tl_ctx->dynamic_remote_info[segid].packed_key,
                          &tl_ctx->dynamic_remote_info[segid].packed_key_len);
        if (UCS_OK != ucs_status) {
            tl_error(tl_ctx->super.super.lib,
                     "failed to pack UCP key with error code: %d", ucs_status);
            return ucs_status_to_ucc_status(ucs_status);
        }
    } else {
        mmap_params.field_mask =
            UCP_MEM_MAP_PARAM_FIELD_ADDRESS | UCP_MEM_MAP_PARAM_FIELD_LENGTH;
        mmap_params.address = map->address;
        mmap_params.length  = map->len;

        ucs_status = ucp_mem_map(tl_ctx->worker.ucp_context, &mmap_params, &mh);
        if (ucs_status != UCS_OK) {
            tl_error(UCC_TASK_LIB(task), "failure in ucp_mem_map %s",
                     ucs_status_string(ucs_status));
            return ucs_status_to_ucc_status(ucs_status);
        }
        tl_ctx->dynamic_remote_info[segid].va_base     = map->address;
        tl_ctx->dynamic_remote_info[segid].len         = map->len;
        tl_ctx->dynamic_remote_info[segid].mem_h       = mh;
        tl_ctx->dynamic_remote_info[segid].packed_memh = NULL;
        ucs_status =
            ucp_rkey_pack(tl_ctx->worker.ucp_context, mh,
                          &tl_ctx->dynamic_remote_info[segid].packed_key,
                          &tl_ctx->dynamic_remote_info[segid].packed_key_len);
        if (UCS_OK != ucs_status) {
            tl_error(tl_ctx->super.super.lib,
                     "failed to pack UCP key with error code: %d", ucs_status);
            return ucs_status_to_ucc_status(ucs_status);
        }
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_coll_dynamic_segments(ucc_coll_args_t   *coll_args,
                                              ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t    *tl_team        = UCC_TL_UCP_TASK_TEAM(task);
    ucc_tl_ucp_lib_t     *tl_lib         = UCC_TL_UCP_TEAM_LIB(tl_team);
    ucc_tl_ucp_context_t *ctx            = UCC_TL_UCP_TEAM_CTX(tl_team);
    int                   i              = 0;
    ucc_status_t          status;

    if (tl_lib->cfg.use_dynamic_segments && coll_args->mem_map.n_segments > 0) {
        int                   starting_index = ctx->n_dynrinfo_segs;
        size_t                seg_pack_size  = 0;
        size_t               *global_size    = NULL;
        size_t                team_size      = UCC_TL_TEAM_SIZE(tl_team);
        ucc_team_t  *core_team = UCC_TL_CORE_TEAM(UCC_TL_UCP_TASK_TEAM(task));
        ucc_subset_t subset    = {.map = tl_team->ctx_map,
                                  .myrank     = core_team->rank};
        ucc_service_coll_req_t *scoll_req;
        void                   *ex_buffer;
        ptrdiff_t               old_offset;

        /* increase dynamic remote info size */
        ctx->dynamic_remote_info = ucc_realloc(
            ctx->dynamic_remote_info,
            sizeof(ucc_tl_ucp_remote_info_t) *
                (ctx->n_dynrinfo_segs + coll_args->mem_map.n_segments),
            "dyn remote info");
        if (!ctx->dynamic_remote_info) {
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            return UCC_ERR_NO_MEMORY;
        }

        for (i = 0; i < coll_args->mem_map.n_segments; i++) {
            /* map the buffer and populate the dynamic_remote_info segments */
            status = ucc_tl_ucp_memmap_append_segment(
                task, &coll_args->mem_map.segments[i], starting_index + i);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed to memory map a segment");
                goto failed_memory_map;
            }
            seg_pack_size +=
                sizeof(uint64_t) * 3 +
                ctx->dynamic_remote_info[starting_index + i].packed_key_len;
        }

        global_size = ucc_calloc(core_team->size, sizeof(size_t));
        if (!global_size) {
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            goto failed_memory_map;
        }

        /* allgather on the new segments size */
        status = ucc_service_allgather(core_team, &seg_pack_size, global_size,
                                       sizeof(uint64_t), subset, &scoll_req);
        if (status < UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed to perform a service allgather");
            ucc_free(global_size);
            goto failed_memory_map;
        }
        while (UCC_INPROGRESS == (status = ucc_service_coll_test(scoll_req))) {
        }
        if (status < UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed on the allgather");
            ucc_service_coll_finalize(scoll_req);
            ucc_free(global_size);
            goto failed_memory_map;
        }
        ucc_service_coll_finalize(scoll_req);
        for (i = 0; i < core_team->size; i++) {
            if (global_size[i] > seg_pack_size) {
                seg_pack_size = global_size[i];
            }
        }
        ucc_free(global_size);

        /* pack the dynamic_remote_info segments */
        ctx->n_dynrinfo_segs += coll_args->mem_map.n_segments;
        ex_buffer = ucc_malloc(seg_pack_size, "ex pack size");
        if (!ex_buffer) {
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            status = UCC_ERR_NO_MEMORY;
            goto failed_memory_map;
        }
        ucc_tl_ucp_pack_data(ctx, starting_index, ex_buffer);

        old_offset = ctx->dyn_seg.buff_size;
        ctx->dyn_seg.buff_size += seg_pack_size * core_team->size;
        ctx->dyn_seg.dyn_buff = ucc_realloc(ctx->dyn_seg.dyn_buff,
                                            ctx->dyn_seg.buff_size, "dyn buff");
        if (!ctx->dyn_seg.dyn_buff) {
            status = UCC_ERR_NO_MEMORY;
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            goto failed_memory_map;
        }
        ctx->dyn_seg.seg_groups = ucc_realloc(
            ctx->dyn_seg.seg_groups, sizeof(uint64_t) * ctx->n_dynrinfo_segs,
            "n_dynrinfo_segs");
        if (!ctx->dyn_seg.seg_groups) {
            status = UCC_ERR_NO_MEMORY;
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            goto failed_memory_map;
        }
        ctx->dyn_seg.seg_group_start = ucc_realloc(
            ctx->dyn_seg.seg_group_start,
            sizeof(uint64_t) * ctx->n_dynrinfo_segs, "n_dynrinfo_segs");
        if (!ctx->dyn_seg.seg_group_start) {
            status = UCC_ERR_NO_MEMORY;
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            goto failed_memory_map;
        }
        ctx->dyn_seg.seg_group_size = ucc_realloc(
            ctx->dyn_seg.seg_group_size,
            sizeof(uint64_t) * ctx->dyn_seg.num_groups + 1, "n_dynrinfo_segs");
        if (!ctx->dyn_seg.seg_group_size) {
            status = UCC_ERR_NO_MEMORY;
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            goto failed_memory_map;
        }

        ctx->dyn_seg.starting_seg = ucc_realloc(
            ctx->dyn_seg.starting_seg, sizeof(uint64_t) * ctx->n_dynrinfo_segs,
            "n_dynrinfo_segs");
        if (!ctx->dyn_seg.starting_seg) {
            status = UCC_ERR_NO_MEMORY;
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            goto failed_memory_map;
        }
        ctx->dyn_seg.num_seg_per_group = ucc_realloc(
            ctx->dyn_seg.num_seg_per_group,
            sizeof(uint64_t) * ctx->dyn_seg.num_groups + 1, "n_dynrinfo_segs");
        if (!ctx->dyn_seg.num_seg_per_group) {
            status = UCC_ERR_NO_MEMORY;
            tl_error(UCC_TASK_LIB(task), "Out of Memory");
            goto failed_memory_map;
        }

        ctx->dyn_seg.num_groups += 1;
        ctx->dyn_seg.num_seg_per_group[ctx->dyn_seg.num_groups - 1] =
            coll_args->mem_map.n_segments;
        ctx->dyn_seg.seg_group_size[ctx->dyn_seg.num_groups - 1] =
            seg_pack_size;
        if (starting_index == 0) {
            for (i = starting_index; i < ctx->n_dynrinfo_segs; i++) {
                ctx->dyn_seg.seg_groups[i]      = 0;
                ctx->dyn_seg.seg_group_start[i] = 0;
                ctx->dyn_seg.starting_seg[i]    = starting_index;
            }
        } else {
            for (i = starting_index; i < ctx->n_dynrinfo_segs; i++) {
                ctx->dyn_seg.seg_groups[i] =
                    ctx->dyn_seg.seg_groups[starting_index - 1] + 1;
                ctx->dyn_seg.seg_group_start[i] = old_offset;
                ctx->dyn_seg.starting_seg[i]    = starting_index;
            }
        }

        /* allgather on the new segments (packed) */
        status = ucc_service_allgather(
            core_team, ex_buffer, PTR_OFFSET(ctx->dyn_seg.dyn_buff, old_offset),
            seg_pack_size, subset, &scoll_req);
        if (status < UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed on the allgather");
            goto failed_memory_map;
        }
        while (UCC_INPROGRESS == (status = ucc_service_coll_test(scoll_req))) {
        }
        if (status < UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "failed on the allgather");
            ucc_service_coll_finalize(scoll_req);
            goto failed_memory_map;
        }
        /* done with allgather */
        ucc_service_coll_finalize(scoll_req);
        ctx->rkeys = ucc_realloc(ctx->rkeys,
                                 team_size * sizeof(ucp_rkey_h) *
                                     (ctx->n_rinfo_segs + ctx->n_dynrinfo_segs),
                                 "rkeys");
        memset(PTR_OFFSET(ctx->rkeys, team_size * sizeof(ucp_rkey_h) *
                                          (ctx->n_rinfo_segs + starting_index)),
               0,
               team_size * sizeof(ucp_rkey_h) * coll_args->mem_map.n_segments);
        ucc_free(ex_buffer);
    }
    return UCC_OK;
failed_memory_map:
    for (i = 0; i < coll_args->mem_map.n_segments; i++) {
        if (ctx->dynamic_remote_info[ctx->n_dynrinfo_segs + i].mem_h) {
            ucp_mem_unmap(ctx->worker.ucp_context, ctx->dynamic_remote_info[ctx->n_dynrinfo_segs + i].mem_h);
        }
        if (ctx->dynamic_remote_info[ctx->n_dynrinfo_segs + i].packed_key) {
            ucp_rkey_buffer_release(ctx->dynamic_remote_info[ctx->n_dynrinfo_segs + i].packed_key);
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
