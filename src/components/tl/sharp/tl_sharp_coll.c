/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_sharp_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include <sharp/api/version.h>
#include <sharp/api/sharp_coll.h>

/* Disable allgather and reduce_scatter for message sizes 0-16k since SHARP SAT is only enabled by default for messages larger than 16k.
 * TODO: update with precise threshold once SHARP query can return SAT threshold.
 */
#define UCC_TL_SHARP_ALLGATHER_DEFAULT_ALG_SELECT_STR                          \
    "allgather:0-16k:0"

#define UCC_TL_SHARP_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR                     \
    "reduce_scatter:0-16k:0"

const char
    *ucc_tl_sharp_default_alg_select_str[UCC_TL_SHARP_N_DEFAULT_ALG_SELECT_STR] = {
        UCC_TL_SHARP_ALLGATHER_DEFAULT_ALG_SELECT_STR,
        UCC_TL_SHARP_REDUCE_SCATTER_DEFAULT_ALG_SELECT_STR};

enum sharp_datatype ucc_to_sharp_dtype[] = {
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT16)]            = SHARP_DTYPE_SHORT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT32)]            = SHARP_DTYPE_INT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT64)]            = SHARP_DTYPE_LONG,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT128)]           = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT16)]           = SHARP_DTYPE_UNSIGNED_SHORT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT32)]           = SHARP_DTYPE_UNSIGNED,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT64)]           = SHARP_DTYPE_UNSIGNED_LONG,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT128)]          = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT16)]          = SHARP_DTYPE_FLOAT_SHORT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32)]          = SHARP_DTYPE_FLOAT,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64)]          = SHARP_DTYPE_DOUBLE,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT128)]         = SHARP_DTYPE_NULL,
#if SHARP_API > SHARP_VERSION(3, 0)
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)]             = (enum sharp_datatype)SHARP_DTYPE_UNKNOWN,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)]            = (enum sharp_datatype)SHARP_DTYPE_UNKNOWN,
    [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)]         = (enum sharp_datatype)SHARP_DTYPE_UNKNOWN,
#else
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)]             = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)]            = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)]         = SHARP_DTYPE_NULL,
#endif
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32_COMPLEX)]  = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64_COMPLEX)]  = SHARP_DTYPE_NULL,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT128_COMPLEX)] = SHARP_DTYPE_NULL,
};

enum sharp_reduce_op ucc_to_sharp_reduce_op[] = {
    [UCC_OP_SUM]         = SHARP_OP_SUM,
    [UCC_OP_PROD]        = SHARP_OP_NULL,
    [UCC_OP_MAX]         = SHARP_OP_MAX,
    [UCC_OP_MIN]         = SHARP_OP_MIN,
    [UCC_OP_LAND]        = SHARP_OP_LAND,
    [UCC_OP_LOR]         = SHARP_OP_LOR,
    [UCC_OP_LXOR]        = SHARP_OP_LXOR,
    [UCC_OP_BAND]        = SHARP_OP_BAND,
    [UCC_OP_BOR]         = SHARP_OP_BOR,
    [UCC_OP_BXOR]        = SHARP_OP_BXOR,
    [UCC_OP_MAXLOC]      = SHARP_OP_MAXLOC,
    [UCC_OP_MINLOC]      = SHARP_OP_MINLOC,
    [UCC_OP_AVG]         = SHARP_OP_NULL,
};

enum sharp_data_memory_type ucc_to_sharp_memtype[] = {
    [UCC_MEMORY_TYPE_HOST]         = SHARP_MEM_TYPE_HOST,
    [UCC_MEMORY_TYPE_CUDA]         = SHARP_MEM_TYPE_CUDA,
    [UCC_MEMORY_TYPE_CUDA_MANAGED] = SHARP_MEM_TYPE_LAST,
    [UCC_MEMORY_TYPE_ROCM]         = SHARP_MEM_TYPE_LAST,
    [UCC_MEMORY_TYPE_ROCM_MANAGED] = SHARP_MEM_TYPE_LAST,
    [UCC_MEMORY_TYPE_LAST]         = SHARP_MEM_TYPE_LAST,
};

static inline ucc_status_t ucc_tl_sharp_status_to_ucc(int status)
{
    switch (status) {
    case SHARP_COLL_SUCCESS:
        return UCC_OK;
    case SHARP_COLL_ENOMEM:
        return UCC_ERR_NO_MEMORY;
    case SHARP_COLL_ENOT_SUPP:
        return UCC_ERR_NOT_SUPPORTED;
    case SHARP_COLL_EINVAL:
        return UCC_ERR_INVALID_PARAM;
    case SHARP_COLL_ENO_RESOURCE:
        return UCC_ERR_NO_RESOURCE;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}

static ucc_tl_sharp_reg_t ucc_tl_sharp_reg_null = { .mr = NULL };

static ucc_status_t
ucc_tl_sharp_mem_register(ucc_tl_sharp_context_t *ctx, ucc_tl_sharp_team_t *team,
                          void *addr, size_t length, ucc_tl_sharp_reg_t **reg)
{
    ucc_rcache_region_t          *rregion;
    ucc_tl_sharp_rcache_region_t *region;
    ucc_status_t                  status;
    ucc_tl_sharp_reg_t           *r;
    ucc_rcache_t                 *rcache;
    struct sharp_coll_context    *sharp_ctx;

    if (length < ctx->cfg.reg_threshold) {
        *reg = &ucc_tl_sharp_reg_null;
        return UCC_OK;
    }

    sharp_ctx = team->sharp_context;
    rcache    = team->rcache;

    if (rcache) {
        status = ucc_rcache_get(rcache, (void *)addr, length, NULL,
                                &rregion);
        if (status != UCC_OK) {
            tl_error(ctx->super.super.lib, "ucc_rcache_get failed");
            return UCC_ERR_INVALID_PARAM;
        }
        region = ucc_derived_of(rregion, ucc_tl_sharp_rcache_region_t);
        *reg   = &region->reg;
    } else {
        r = ucc_malloc(sizeof(ucc_tl_sharp_reg_t), "sharp reg");
        if (!r) {
            tl_error(ctx->super.super.lib, "failed to allocate reg data");
            return UCC_ERR_NO_MEMORY;
        }

        sharp_coll_reg_mr(sharp_ctx, addr, length, &r->mr);
        *reg = r;
    }

    return UCC_OK;
}

static ucc_status_t
ucc_tl_sharp_mem_deregister(ucc_tl_sharp_team_t *team, ucc_tl_sharp_reg_t *reg)
{
    ucc_tl_sharp_rcache_region_t *region;
    ucc_rcache_t *rcache;
    struct       sharp_coll_context *sharp_ctx;

    if (reg == &ucc_tl_sharp_reg_null) {
        return UCC_OK;
    }

    sharp_ctx = team->sharp_context;
    rcache    = team->rcache;

    if (rcache) {
        region = ucc_container_of(reg, ucc_tl_sharp_rcache_region_t, reg);
        ucc_rcache_region_put(rcache, &region->super);
    } else {
        sharp_coll_dereg_mr(sharp_ctx, reg->mr);
        ucc_free(reg);
    }

    return UCC_OK;
}

void ucc_tl_sharp_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    int completed;
    ucc_status_t st1, st2;

    if (task->req_handle != NULL) {
        completed = sharp_coll_req_test(task->req_handle);
        if (completed) {
            st1 = UCC_OK; st2 = UCC_OK;
            switch(TASK_ARGS(task).coll_type){
            case UCC_COLL_TYPE_ALLREDUCE:
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    st1 = ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                      task->allreduce.s_mem_h);
                }
                st2 = ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                  task->allreduce.r_mem_h);
                break;
            case UCC_COLL_TYPE_BCAST:
                st1 = ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                            task->bcast.mem_h);
                break;
            case UCC_COLL_TYPE_ALLGATHER:
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    st1 = ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                      task->allgather.s_mem_h);
                }
                st2 = ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                  task->allgather.r_mem_h);
                break;
            case UCC_COLL_TYPE_REDUCE_SCATTER:
                if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
                    st1 = ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                      task->reduce_scatter.s_mem_h);
                }
                st2 = ucc_tl_sharp_mem_deregister(TASK_TEAM(task),
                                                  task->reduce_scatter.r_mem_h);
                break;
            default:
                break;
            }
            if (ucc_unlikely(st1 != UCC_OK || st2 != UCC_OK)) {
                tl_warn(UCC_TASK_LIB(task), "ucc_tl_sharp_mem_deregister failed");
            }
            sharp_coll_req_free(task->req_handle);
            coll_task->status = UCC_OK;
            UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task,
                                               "sharp_collective_done", 0);
        }
    }
}

ucc_status_t ucc_tl_sharp_barrier_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t *team  = TASK_TEAM(task);
    int                  ret;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_barrier_start", 0);

    ret = sharp_coll_do_barrier_nb(team->sharp_comm, &task->req_handle);
    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        tl_error(UCC_TASK_LIB(task), "sharp_coll_do_barrier_nb failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_sharp_allreduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t          *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t          *team  = TASK_TEAM(task);
    ucc_coll_args_t              *args  = &TASK_ARGS(task);
    size_t                        count = args->dst.info.count;
    ucc_datatype_t                dt    = args->dst.info.datatype;
    struct sharp_coll_reduce_spec reduce_spec;
    enum sharp_datatype           sharp_type;
    enum sharp_reduce_op          op_type;
    size_t                        data_size;
    int                           ret;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_allreduce_start", 0);

    sharp_type = ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(dt)];
    op_type    = ucc_to_sharp_reduce_op[args->op];
    data_size  = ucc_dt_size(dt) * count;

    if (!UCC_IS_INPLACE(*args)) {
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer,data_size,
                                  &task->allreduce.s_mem_h);
    }
    ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->dst.info.buffer, data_size,
                              &task->allreduce.r_mem_h);

    if (!UCC_IS_INPLACE(*args)) {
        reduce_spec.sbuf_desc.buffer.ptr        = args->src.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->allreduce.s_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->src.info.mem_type];
    } else {
        reduce_spec.sbuf_desc.buffer.ptr        = args->dst.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle = task->allreduce.r_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    }

    reduce_spec.sbuf_desc.buffer.length     = data_size;
    reduce_spec.sbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.buffer.ptr        = args->dst.info.buffer;
    reduce_spec.rbuf_desc.buffer.length     = data_size;
    reduce_spec.rbuf_desc.buffer.mem_handle = task->allreduce.r_mem_h->mr;
    reduce_spec.rbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.mem_type          = ucc_to_sharp_memtype[args->dst.info.mem_type];
    reduce_spec.aggr_mode                   = SHARP_AGGREGATION_NONE;
    reduce_spec.length                      = count;
    reduce_spec.dtype                       = sharp_type;
    reduce_spec.op                          = op_type;

    ret = sharp_coll_do_allreduce_nb(team->sharp_comm, &reduce_spec, &task->req_handle);
    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        tl_error(UCC_TASK_LIB(task), "sharp_coll_do_allreduce_nb failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_sharp_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t *    args  = &TASK_ARGS(task);
    ucc_datatype_t       dt    = args->src.info.datatype;
    size_t               count = args->src.info.count;
    ucc_rank_t           root  = args->root;
    size_t               data_size;
    struct sharp_coll_bcast_spec bcast_spec;
    int                          ret;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_bcast_start", 0);

    data_size = ucc_dt_size(dt) * count;

    ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer,
                              data_size, &task->bcast.mem_h);

    bcast_spec.size                       = data_size;
    bcast_spec.root                       = root;
    bcast_spec.buf_desc.type              = SHARP_DATA_BUFFER;
    bcast_spec.buf_desc.buffer.ptr        = args->src.info.buffer;
    bcast_spec.buf_desc.buffer.length     = data_size;
    bcast_spec.buf_desc.buffer.mem_handle = task->bcast.mem_h->mr;
    bcast_spec.buf_desc.mem_type =
        ucc_to_sharp_memtype[args->src.info.mem_type];

    ret = sharp_coll_do_bcast_nb(team->sharp_comm, &bcast_spec,
                                 &task->req_handle);

    if (ucc_unlikely(ret != SHARP_COLL_SUCCESS)) {
        if (ret == SHARP_COLL_ENOT_SUPP) {
            tl_debug(UCC_TASK_LIB(task),
                     "sharp_coll_do_bcast_nb not supported, msgsize %zd",
                     data_size);
        } else {
            tl_error(UCC_TASK_LIB(task), "sharp_coll_do_bcast_nb failed:%s",
                     sharp_coll_strerror(ret));
        }
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

#if HAVE_DECL_SHARP_COLL_DO_REDUCE_SCATTER
ucc_status_t ucc_tl_sharp_reduce_scatter_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t     *args  = &TASK_ARGS(task);
    size_t              count = args->dst.info.count;
    ucc_datatype_t      dt    = args->dst.info.datatype;
    struct sharp_coll_reduce_spec reduce_spec;
    enum sharp_datatype           sharp_type;
    enum sharp_reduce_op          op_type;
    size_t                        src_data_size, dst_data_size;
    int                           ret;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_reduce_scatter_start",
                                       0);

    sharp_type    = ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(dt)];
    op_type       = ucc_to_sharp_reduce_op[args->op];
    src_data_size = ucc_dt_size(dt) * count * UCC_TL_TEAM_SIZE(team);
    dst_data_size = ucc_dt_size(dt) * count;

    if (!UCC_IS_INPLACE(*args)) {
        ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer,
                                  src_data_size, &task->reduce_scatter.s_mem_h);
    }
    ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->dst.info.buffer,
                              dst_data_size, &task->reduce_scatter.r_mem_h);

    if (!UCC_IS_INPLACE(*args)) {
        reduce_spec.sbuf_desc.buffer.ptr = args->src.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle =
            task->reduce_scatter.s_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type =
            ucc_to_sharp_memtype[args->src.info.mem_type];
    } else {
        reduce_spec.sbuf_desc.buffer.ptr = args->dst.info.buffer;
        reduce_spec.sbuf_desc.buffer.mem_handle =
            task->reduce_scatter.r_mem_h->mr;
        reduce_spec.sbuf_desc.mem_type =
            ucc_to_sharp_memtype[args->dst.info.mem_type];
    }

    reduce_spec.sbuf_desc.buffer.length     = src_data_size;
    reduce_spec.sbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.buffer.ptr        = args->dst.info.buffer;
    reduce_spec.rbuf_desc.buffer.length     = dst_data_size;
    reduce_spec.rbuf_desc.buffer.mem_handle = task->reduce_scatter.r_mem_h->mr;
    reduce_spec.rbuf_desc.type              = SHARP_DATA_BUFFER;
    reduce_spec.rbuf_desc.mem_type =
        ucc_to_sharp_memtype[args->dst.info.mem_type];
    reduce_spec.aggr_mode = SHARP_AGGREGATION_NONE;
    reduce_spec.length    = count;
    reduce_spec.dtype     = sharp_type;
    reduce_spec.op        = op_type;
    reduce_spec.offset    = 0;

    ret = sharp_coll_do_reduce_scatter_nb(team->sharp_comm, &reduce_spec,
                                          &task->req_handle);
    if (ret != SHARP_COLL_SUCCESS) {
        tl_error(UCC_TASK_LIB(task),
                 "sharp_coll_do_reduce_scatter_nb failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_sharp_reduce_scatter_init(ucc_tl_sharp_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);

    if (!(TASK_CTX(task)->sharp_caps.support_mask.feature_mask & SHARP_FEATURE_SAT)) {
       return UCC_ERR_NOT_SUPPORTED;
    }

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if ((!UCC_IS_INPLACE(*args) &&
        ucc_to_sharp_memtype[args->src.info.mem_type] == SHARP_MEM_TYPE_LAST) ||
        ucc_to_sharp_memtype[args->dst.info.mem_type] == SHARP_MEM_TYPE_LAST ||
        ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)] ==
            SHARP_DTYPE_NULL ||
        ucc_to_sharp_reduce_op[args->op] == SHARP_OP_NULL) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_sharp_reduce_scatter_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
};
#endif

ucc_status_t ucc_tl_sharp_allreduce_init(ucc_tl_sharp_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if ((!UCC_IS_INPLACE(*args) &&
         ucc_to_sharp_memtype[args->src.info.mem_type] == SHARP_MEM_TYPE_LAST) ||
        ucc_to_sharp_memtype[args->dst.info.mem_type] == SHARP_MEM_TYPE_LAST ||
        ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)] == SHARP_DTYPE_NULL ||
        ucc_to_sharp_reduce_op[args->op] == SHARP_OP_NULL) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_sharp_allreduce_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_sharp_bcast_init(ucc_tl_sharp_task_t *task)
{
    ucc_coll_args_t *args  = &TASK_ARGS(task);
    size_t           data_size;

    data_size = ucc_dt_size(args->src.info.datatype) * args->src.info.count;

    /* check SHARP supports memory type, dataype is contig and
       data size is even in case of older sharp versions */
    if ((ucc_to_sharp_memtype[args->src.info.mem_type] == SHARP_MEM_TYPE_LAST) ||
        !ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID) ||
        ((data_size % 2 != 0) &&
        ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(UCC_DT_INT8)] == SHARP_DTYPE_NULL)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_sharp_bcast_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
}

ucc_status_t ucc_tl_sharp_barrier_init(ucc_tl_sharp_task_t *task)
{
    task->super.post     = ucc_tl_sharp_barrier_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
};

#if HAVE_DECL_SHARP_COLL_DO_ALLGATHER
ucc_status_t ucc_tl_sharp_allgather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_sharp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_sharp_task_t);
    ucc_tl_sharp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t     *args  = &TASK_ARGS(task);
    size_t               count = args->dst.info.count;
    ucc_datatype_t       dt    = args->dst.info.datatype;
    struct sharp_coll_gather_spec gather_spec;
    size_t                        src_data_size, dst_data_size;
    int                           ret;
    ucc_status_t                  status;

    UCC_TL_SHARP_PROFILE_REQUEST_EVENT(coll_task, "sharp_allgather_start", 0);

    src_data_size = ucc_dt_size(dt) * count / UCC_TL_TEAM_SIZE(team);
    dst_data_size = ucc_dt_size(dt) * count;

    status = ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->dst.info.buffer,
                              dst_data_size, &task->allgather.r_mem_h);
    if (status != UCC_OK) {
        tl_error(UCC_TASK_LIB(task), "ucc_tl_sharp_mem_register failed "
                 "for dst buffer: %s",
                 ucc_status_string(status));
        return status;
    }
    if (!UCC_IS_INPLACE(*args)) {
        status = ucc_tl_sharp_mem_register(TASK_CTX(task), team, args->src.info.buffer,
                                  src_data_size, &task->allgather.s_mem_h);
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "ucc_tl_sharp_mem_register failed "
                     "for src buffer: %s",
                     ucc_status_string(status));
            return status;
        }
    }

    if (!UCC_IS_INPLACE(*args)) {
        gather_spec.sbuf_desc.buffer.ptr = args->src.info.buffer;
        gather_spec.sbuf_desc.buffer.mem_handle = task->allgather.s_mem_h->mr;
        gather_spec.sbuf_desc.mem_type =
            ucc_to_sharp_memtype[args->src.info.mem_type];
    } else {
        gather_spec.sbuf_desc.buffer.ptr = PTR_OFFSET(args->dst.info.buffer,
                                                      UCC_TL_TEAM_RANK(team) *
                                                      src_data_size);
        gather_spec.sbuf_desc.buffer.mem_handle = task->allgather.r_mem_h->mr;
        gather_spec.sbuf_desc.mem_type =
            ucc_to_sharp_memtype[args->dst.info.mem_type];
    }
    gather_spec.sbuf_desc.buffer.length = src_data_size;
    gather_spec.sbuf_desc.type = SHARP_DATA_BUFFER;

    gather_spec.rbuf_desc.buffer.ptr        = args->dst.info.buffer;
    gather_spec.rbuf_desc.buffer.length     = dst_data_size;
    gather_spec.rbuf_desc.buffer.mem_handle = task->allgather.r_mem_h->mr;
    gather_spec.rbuf_desc.type              = SHARP_DATA_BUFFER;
    gather_spec.rbuf_desc.mem_type =
        ucc_to_sharp_memtype[args->dst.info.mem_type];
    gather_spec.offset    = 0;
    gather_spec.dtype = SHARP_DTYPE_INT8;

    ret = sharp_coll_do_allgather_nb(team->sharp_comm, &gather_spec,
                                     &task->req_handle);
    if (ret != SHARP_COLL_SUCCESS) {
        tl_error(UCC_TASK_LIB(task),
                 "sharp_coll_do_allgather_nb failed:%s",
                 sharp_coll_strerror(ret));
        coll_task->status = ucc_tl_sharp_status_to_ucc(ret);
        return ucc_task_complete(coll_task);
    }
    coll_task->status = UCC_INPROGRESS;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_sharp_allgather_init(ucc_tl_sharp_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);

    if (!(TASK_CTX(task)->sharp_caps.support_mask.feature_mask & SHARP_FEATURE_SAT)) {
       return UCC_ERR_NOT_SUPPORTED;
    }

    if (!ucc_coll_args_is_predefined_dt(args, UCC_RANK_INVALID)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if ((!UCC_IS_INPLACE(*args) &&
        ucc_to_sharp_memtype[args->src.info.mem_type] == SHARP_MEM_TYPE_LAST) ||
        ucc_to_sharp_memtype[args->dst.info.mem_type] == SHARP_MEM_TYPE_LAST ||
        ucc_to_sharp_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)] ==
            SHARP_DTYPE_NULL) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->allgather.s_mem_h = NULL;
    task->allgather.r_mem_h = NULL;
    task->super.post     = ucc_tl_sharp_allgather_start;
    task->super.progress = ucc_tl_sharp_collective_progress;
    return UCC_OK;
};
#endif
