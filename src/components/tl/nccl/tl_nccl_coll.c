/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl_coll.h"
#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"
#include "core/ucc_ee.h"
#include "core/ucc_context.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "utils/arch/cuda_def.h"
#include "allgatherv/allgatherv.h"

#define ncclOpUnsupported (ncclNumOps + 1)
#define ncclDataTypeUnsupported (ncclNumTypes + 1)

ncclDataType_t ucc_to_nccl_dtype[] = {
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)] = (ncclDataType_t)ncclInt8,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT16)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT32)] = (ncclDataType_t)ncclInt32,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT64)] = (ncclDataType_t)ncclInt64,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT128)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)] = (ncclDataType_t)ncclUint8,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT16)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT32)] = (ncclDataType_t)ncclUint32,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT64)] = (ncclDataType_t)ncclUint64,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT128)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT16)] = (ncclDataType_t)ncclFloat16,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32)] = (ncclDataType_t)ncclFloat32,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64)] = (ncclDataType_t)ncclFloat64,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT128)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32_COMPLEX)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64_COMPLEX)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT128_COMPLEX)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
#if (CUDART_VERSION >= 11000) && (NCCL_VERSION_CODE >= NCCL_VERSION(2,10,3))
    [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)] = (ncclDataType_t)ncclBfloat16,
#else
    [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)] =
        (ncclDataType_t)ncclDataTypeUnsupported,
#endif
};

ncclRedOp_t ucc_to_nccl_reduce_op[] = {
    [UCC_OP_SUM]         = (ncclRedOp_t)ncclSum,
    [UCC_OP_PROD]        = (ncclRedOp_t)ncclProd,
    [UCC_OP_MAX]         = (ncclRedOp_t)ncclMax,
    [UCC_OP_MIN]         = (ncclRedOp_t)ncclMin,
#if NCCL_VERSION_CODE < NCCL_VERSION(2,10,3)
    [UCC_OP_AVG]         = (ncclRedOp_t)ncclOpUnsupported,
#else
    [UCC_OP_AVG]         = (ncclRedOp_t)ncclAvg,
#endif
    [UCC_OP_LAND]        = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_LOR]         = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_LXOR]        = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_BAND]        = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_BOR]         = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_BXOR]        = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_MAXLOC]      = (ncclRedOp_t)ncclOpUnsupported,
    [UCC_OP_MINLOC]      = (ncclRedOp_t)ncclOpUnsupported,
};

const char
    *ucc_tl_nccl_default_alg_select_str[UCC_TL_NCCL_N_DEFAULT_ALG_SELECT_STR] = {
        UCC_TL_NCCL_ALLGATHERV_DEFAULT_ALG_SELECT_STR};

static inline void
ucc_tl_nccl_check_and_convert_buffer(ucc_coll_buffer_info_t *buffer_info,
                                     ucc_datatype_t          new_datatype)
{
    if (ucc_to_nccl_dtype[UCC_DT_PREDEFINED_ID(buffer_info->datatype)] ==
        ncclDataTypeUnsupported) {
        ucc_assert(ucc_dt_size(buffer_info->datatype) %
                       ucc_dt_size(new_datatype) ==
                   0);
        buffer_info->count *=
            ucc_dt_size(buffer_info->datatype) / ucc_dt_size(new_datatype);
        buffer_info->datatype = new_datatype;
    }
}

static inline ucc_status_t ucc_tl_nccl_check_and_convert_buffer_reduction(
    ucc_coll_buffer_info_t *buffer_info, ucc_tl_nccl_task_t *task)
{
    ucc_reduction_op_t op = TASK_ARGS(task).op;

    if (ucc_to_nccl_reduce_op[op] == ncclOpUnsupported) {
        tl_debug(UCC_TASK_LIB(task), "reduction operation %s is not supported",
                 ucc_reduction_op_str(op));
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (ucc_to_nccl_dtype[UCC_DT_PREDEFINED_ID(buffer_info->datatype)] ==
        ncclDataTypeUnsupported) {
        if (op == UCC_OP_SUM) {
            switch (buffer_info->datatype) {
            case UCC_DT_FLOAT32_COMPLEX:
                ucc_tl_nccl_check_and_convert_buffer(buffer_info,
                                                     UCC_DT_FLOAT32);
                return UCC_OK;
            case UCC_DT_FLOAT64_COMPLEX:
                ucc_tl_nccl_check_and_convert_buffer(buffer_info,
                                                     UCC_DT_FLOAT64);
                return UCC_OK;
            default:
                break;
            }
        }
        tl_debug(UCC_TASK_LIB(task),
                 "datatype %s is not supported for reduction operation %s",
                 ucc_datatype_str(buffer_info->datatype),
                 ucc_reduction_op_str(op));
        return UCC_ERR_NOT_SUPPORTED;
    }
    return UCC_OK;
}

#if NCCL_HAS_UBR
/* Helper function to lazily register a memory region with NCCL communicator */
static inline ucc_status_t ucc_tl_nccl_lazy_register_memh(
    void *buffer, size_t length, ucc_tl_nccl_team_t *team,
    ucc_mem_map_mem_h memh)
{
    ucc_tl_nccl_context_t   *ctx = UCC_TL_NCCL_TEAM_CTX(team);
    ucc_tl_nccl_memh_data_t *m_data;
    ucc_mem_map_memh_t      *mem_handle;
    ncclResult_t             nccl_status;
    ncclComm_t              *new_comms;
    void                   **new_handles;
    void                    *nccl_handle;
    int                      i, new_max;
    uintptr_t                buf_start, buf_end, region_start, region_end;

    /* Skip if UBR is not available or memh not provided */
    if (!ctx->ubr_available || !memh) {
        return UCC_OK;
    }

    mem_handle = (ucc_mem_map_memh_t *)memh;
    m_data     = NULL;
    for (i = 0; i < mem_handle->num_tls; i++) {
        if (strcmp(mem_handle->tl_h[i].tl_name, "nccl") == 0) {
            m_data = (ucc_tl_nccl_memh_data_t *)mem_handle->tl_h[i].tl_data;
            break;
        }
    }

    if (!m_data) {
        /* No NCCL memh data - buffer not registered with TL/NCCL */
        return UCC_OK;
    }

    if (length > (UINTPTR_MAX - (uintptr_t)buffer)) {
        tl_error(UCC_TL_TEAM_LIB(team), "NCCL UBR: buffer size causes overflow");
        return UCC_ERR_INVALID_PARAM;
    }

    /* Verify that the entire buffer is within the registered memory region */
    buf_start    = (uintptr_t)buffer;
    buf_end      = buf_start + length;
    region_start = (uintptr_t)m_data->address;
    region_end   = region_start + m_data->length;

    if (buf_start < region_start || buf_end > region_end) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "NCCL UBR: buffer [%p, %p) is outside registered region [%p, %p)",
            buffer,
            (void *)buf_end,
            m_data->address,
            (void *)region_end);
        return UCC_ERR_INVALID_PARAM;
    }

    /* Verify team communicator is initialized */
    if (!team->nccl_comm) {
        tl_debug(UCC_TL_TEAM_LIB(team),
                 "NCCL UBR: communicator not initialized, skipping registration");
        return UCC_OK;
    }

    /* Check if already registered with this communicator */
    for (i = 0; i < m_data->num_comms; i++) {
        if (m_data->registered_comms[i] == team->nccl_comm) {
            /* Already registered */
            return UCC_OK;
        }
    }

    /* Need to register the memory region with this communicator */
    nccl_status = ncclCommRegister(
        team->nccl_comm, m_data->address, m_data->length, &nccl_handle);
    if (nccl_status != ncclSuccess) {
        tl_warn(
            UCC_TL_TEAM_LIB(team),
            "NCCL UBR: failed to register region %p, size %zu: %s",
            m_data->address,
            m_data->length,
            ncclGetErrorString(nccl_status));
        /* Don't fail - UBR is an optimization */
        return UCC_OK;
    }

    /* Add this comm and handle to the registered lists */
    if (m_data->num_comms >= m_data->max_comms) {
        /* Need to grow the arrays */
        new_max   = (m_data->max_comms == 0) ? 4 : (m_data->max_comms * 2);
        new_comms = (ncclComm_t *)ucc_realloc(
            m_data->registered_comms,
            new_max * sizeof(ncclComm_t),
            "nccl_registered_comms");
        if (!new_comms) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to allocate memory for registered comms array");
            /* Deregister the buffer since we can't track it */
            ncclCommDeregister(team->nccl_comm, nccl_handle);
            return UCC_ERR_NO_MEMORY;
        }
        m_data->registered_comms = new_comms;

        new_handles = (void **)ucc_realloc(
            m_data->nccl_handles, new_max * sizeof(void *), "nccl_handles");
        if (!new_handles) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to allocate memory for NCCL handles array");
            /* Deregister the buffer since we can't track it */
            ncclCommDeregister(team->nccl_comm, nccl_handle);
            return UCC_ERR_NO_MEMORY;
        }
        m_data->nccl_handles = new_handles;
        m_data->max_comms    = new_max;
    }

    m_data->registered_comms[m_data->num_comms] = team->nccl_comm;
    m_data->nccl_handles[m_data->num_comms]     = nccl_handle;
    m_data->num_comms++;

    tl_debug(
        UCC_TL_TEAM_LIB(team),
        "NCCL UBR: lazily registered region %p, size %zu with comm %p "
        "(for buffer [%p, %p))",
        m_data->address,
        m_data->length,
        team->nccl_comm,
        buffer,
        (void *)buf_end);

    return UCC_OK;
}
#endif

ucc_status_t ucc_tl_nccl_init_task(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_tl_nccl_task_t **coll_task)
{
    ucc_tl_nccl_team_t    *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_tl_nccl_context_t *nccl_ctx  = ucc_derived_of(team->context,
                                                      ucc_tl_nccl_context_t);
    ucc_tl_nccl_task_t    *task;
    ucc_status_t           status;
    ucc_coll_progress_fn_t progress_fn;

    if (!ucc_coll_args_is_predefined_dt(&coll_args->args, team->params.rank)) {
        tl_error(team->context->lib,
                 "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (ucc_unlikely(nccl_team->comm_state != TL_NCCL_COMM_STATE_READY)) {
        if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
            /* active set is not supported with lazy comm init*/
            return UCC_ERR_NOT_SUPPORTED;
        }
        status = ucc_tl_nccl_comm_init(nccl_team);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }
    }

    task = ucc_mpool_get(&nccl_ctx->req_mp);
    if (ucc_unlikely(!task)) {
        tl_error(team->context->lib, "failed to get task from mpool");
        return UCC_ERR_NO_MEMORY;
    }
    progress_fn = task->super.progress;

    ucc_coll_task_init(&task->super, coll_args, team);
    UCC_TL_NCCL_PROFILE_REQUEST_NEW(task, "tl_nccl_task", 0);
    task->super.finalize           = ucc_tl_nccl_coll_finalize;
    task->super.triggered_post     = ucc_tl_nccl_triggered_post;
    task->super.progress           = progress_fn;
    task->completed                = NULL;
    if (nccl_ctx->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT) {
        status = ucc_ec_create_event(&task->completed, UCC_EE_CUDA_STREAM);
        if (ucc_unlikely(status != UCC_OK)) {
            ucc_mpool_put(task);
            return status;
        }
    }

#if NCCL_HAS_UBR
    /* Lazily register memory regions if they were pre-mapped and UBR is enabled */
    if (nccl_ctx->ubr_available) {
        ucc_mem_map_mem_h src_memh = NULL;
        ucc_mem_map_mem_h dst_memh = NULL;
        ucc_rank_t        grank    = UCC_TL_TEAM_RANK(nccl_team);
        ucc_count_t       total_count;
        
        /* Register source buffer's memory region if memh provided */
        if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH) {
            /* Check if global or local memh */
            if ((coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
                (coll_args->args.flags & UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL)) {
                src_memh = coll_args->args.src_memh.global_memh[grank];
            } else {
                src_memh = coll_args->args.src_memh.local_memh;
            }

            if (coll_args->args.coll_type == UCC_COLL_TYPE_ALLGATHERV ||
                coll_args->args.coll_type == UCC_COLL_TYPE_ALLTOALLV ||
                coll_args->args.coll_type == UCC_COLL_TYPE_GATHERV ||
                coll_args->args.coll_type == UCC_COLL_TYPE_SCATTERV) {
                total_count = ucc_coll_args_get_v_buffer_size(
                    &coll_args->args,
                    coll_args->args.src.info_v.counts,
                    coll_args->args.src.info_v.displacements,
                    UCC_TL_TEAM_SIZE(nccl_team));
            } else {
                total_count = coll_args->args.src.info.count;
            }
            status = ucc_tl_nccl_lazy_register_memh(
                coll_args->args.src.info.buffer,
                total_count * ucc_dt_size(coll_args->args.src.info.datatype),
                nccl_team,
                src_memh);
            if (ucc_unlikely(status != UCC_OK)) {
                tl_error(
                    team->context->lib,
                    "NCCL UBR: lazy_register failed with status %d",
                    status);
                ucc_mpool_put(task);
                return status;
            }
        }

        /* Register destination buffer's memory region if memh provided */
        if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH) {
            /* Check if global or local memh */
            if ((coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
                (coll_args->args.flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL)) {
                dst_memh = coll_args->args.dst_memh.global_memh[grank];
            } else {
                dst_memh = coll_args->args.dst_memh.local_memh;
            }

            if (coll_args->args.coll_type == UCC_COLL_TYPE_ALLGATHERV ||
                coll_args->args.coll_type == UCC_COLL_TYPE_ALLTOALLV ||
                coll_args->args.coll_type == UCC_COLL_TYPE_GATHERV ||
                coll_args->args.coll_type == UCC_COLL_TYPE_SCATTERV) {
                total_count = ucc_coll_args_get_v_buffer_size(
                    &coll_args->args,
                    coll_args->args.dst.info_v.counts,
                    coll_args->args.dst.info_v.displacements,
                    UCC_TL_TEAM_SIZE(nccl_team));
            } else {
                total_count = coll_args->args.dst.info.count;
            }

            status = ucc_tl_nccl_lazy_register_memh(
                coll_args->args.dst.info.buffer,
                total_count * ucc_dt_size(coll_args->args.dst.info.datatype),
                nccl_team,
                dst_memh);
            if (ucc_unlikely(status != UCC_OK)) {
                ucc_mpool_put(task);
                return status;
            }
        }
    }
#endif

    *coll_task = task;
    return UCC_OK;
}

void ucc_tl_nccl_free_task(ucc_tl_nccl_task_t *task)
{
    UCC_TL_NCCL_PROFILE_REQUEST_FREE(task);
    if (task->completed) {
        ucc_ec_destroy_event(task->completed, UCC_EE_CUDA_STREAM);
    }
    ucc_mpool_put(task);
}

ucc_status_t ucc_tl_nccl_triggered_post(ucc_ee_h ee, ucc_ev_t *ev,
                                        ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task  = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t        status;
    ucc_ev_t            post_event;

    ucc_assert(ee->ee_type == UCC_EE_CUDA_STREAM);
    coll_task->ee = ee;
    tl_debug(UCC_TASK_LIB(task), "triggered post. task:%p", coll_task);
    status = coll_task->post(coll_task);
    if (ucc_likely(status == UCC_OK)) {
        post_event.ev_type         = UCC_EVENT_COLLECTIVE_POST;
        post_event.ev_context_size = 0;
        post_event.ev_context      = NULL;
        post_event.req             = &coll_task->super;
        ucc_ee_set_event_internal(coll_task->ee, &post_event,
                                  &coll_task->ee->event_out_queue);
    }
    return status;
}

ucc_status_t ucc_tl_nccl_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_status_t        status = UCC_OK;

    if (ucc_unlikely(task->super.super.status != UCC_OK)) {
        team->comm_state = TL_NCCL_COMM_STATE_ERROR;
    }
    tl_debug(UCC_TASK_LIB(task), "finalizing coll task %p", task);
    ucc_tl_nccl_free_task(task);
    return status;
}

ucc_status_t ucc_tl_nccl_collective_sync(ucc_tl_nccl_task_t *task,
                                         cudaStream_t stream)
{
    ucc_tl_nccl_context_t *ctx    = TASK_CTX(task);
    ucc_status_t           status = UCC_OK;
    enum cudaStreamCaptureStatus capture_st;
    CUresult cu_status;
    cudaError_t cuda_st;

    if (task->super.ee) {
        cuda_st =cudaStreamIsCapturing((cudaStream_t)task->super.ee->ee_context,
                                       &capture_st);
        if ((cuda_st == cudaSuccess) &&
            (capture_st != cudaStreamCaptureStatusNone)) {
            task->super.status = UCC_OK;
            return ucc_task_complete(&task->super);
        }
    }

    task->host_status = task->super.status;
    if (ctx->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT) {
        status = ucc_ec_event_post(stream, task->completed, UCC_EE_CUDA_STREAM);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }
    } else {
        cu_status = cuStreamWriteValue32(stream, (CUdeviceptr)task->dev_status,
                                         UCC_OK, 0);
        if (ucc_unlikely(cu_status != CUDA_SUCCESS)) {
            return UCC_ERR_NO_MESSAGE;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(TASK_TEAM(task))->pq,
                                      &task->super);
}

ucc_status_t ucc_tl_nccl_alltoall_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    ucc_rank_t          gsize  = UCC_TL_TEAM_SIZE(team);
    ucc_status_t        status = UCC_OK;
    ptrdiff_t           sbuf   = (ptrdiff_t)args->src.info.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)args->dst.info.buffer;
    size_t     data_size;

    task->super.status = UCC_INPROGRESS;
    data_size          = (size_t)(args->src.info.count / gsize) *
                ucc_dt_size(args->src.info.datatype);
    ucc_assert(args->src.info.count % gsize == 0);
    if (data_size == 0) {
        task->super.status = UCC_OK;
        return ucc_task_complete(&task->super);
    }
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_alltoall_start", 0);
#if NCCL_VERSION_CODE >= NCCL_VERSION(2,28,0)
    NCCLCHECK_GOTO(ncclAlltoAll((void *)sbuf, (void *)rbuf, data_size,
                                ncclChar, team->nccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 0);
#else
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 0);
    for (ucc_rank_t peer = 0; peer < gsize; peer++) {
        NCCLCHECK_GOTO(ncclSend((void *)(sbuf + peer * data_size), data_size,
                                ncclChar, peer, team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 0);
        NCCLCHECK_GOTO(ncclRecv((void *)(rbuf + peer * data_size), data_size,
                                ncclChar, peer, team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 0);
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 1);
#endif
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_alltoall_init(ucc_tl_nccl_task_t *task)
{
    if (UCC_IS_INPLACE(TASK_ARGS(task))) {
        tl_error(UCC_TASK_LIB(task), "inplace alltoallv is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post = ucc_tl_nccl_alltoall_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_alltoallv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    ucc_status_t        status = UCC_OK;
    ptrdiff_t           sbuf   = (ptrdiff_t)args->src.info_v.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)args->dst.info_v.buffer;
    size_t     sdt_size, rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.status = UCC_INPROGRESS;
    sdt_size           = ucc_dt_size(args->src.info_v.datatype);
    rdt_size           = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_alltoallv_start", 0);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 0);
    for (peer = 0; peer < UCC_TL_TEAM_SIZE(team); peer++) {
        count = ucc_coll_args_get_count(args, args->src.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(
                args, args->src.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclSend((void *)(sbuf + displ * sdt_size),
                                    count * sdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team),
                        &task->nccl_progress_st, team->nccl_comm, 0);
        }
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclRecv((void *)(rbuf + displ * rdt_size),
                                    count * rdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team),
                        &task->nccl_progress_st, team->nccl_comm, 0);
        }
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 1);
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_alltoallv_init(ucc_tl_nccl_task_t *task)
{
    if (UCC_IS_INPLACE(TASK_ARGS(task))) {
        tl_error(UCC_TASK_LIB(task), "inplace alltoall is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post = ucc_tl_nccl_alltoallv_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allreduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    =
        UCC_IS_INPLACE(*args) ? args->dst.info.buffer : args->src.info.buffer;
    ucc_status_t        status = UCC_OK;
    ncclRedOp_t         op     = ucc_to_nccl_reduce_op[args->op];
    size_t              count  = args->dst.info.count;
    ncclDataType_t      dt;

    dt = ucc_to_nccl_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)];
    task->super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task,
                                      args->coll_type == UCC_COLL_TYPE_BARRIER
                                          ? "nccl_barrier_start"
                                          : "nccl_allreduce_start",
                                      0);
    NCCLCHECK_GOTO(ncclAllReduce(src, dst, count, dt, op, team->nccl_comm,
                                 stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 0);
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allreduce_init(ucc_tl_nccl_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);

    if (!UCC_IS_INPLACE(*args)) {
        if (ucc_tl_nccl_check_and_convert_buffer_reduction(&args->src.info,
                                                           task) != UCC_OK) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    if (ucc_tl_nccl_check_and_convert_buffer_reduction(&args->dst.info, task) !=
        UCC_OK) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post = ucc_tl_nccl_allreduce_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allgather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          rank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    = args->src.info.buffer;
    ucc_status_t        status = UCC_OK;
    size_t              count  = args->dst.info.count;
    ncclDataType_t      dt;

    dt = ucc_to_nccl_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)];
    if (UCC_IS_INPLACE(*args)) {
        src = (void *)((ptrdiff_t)args->dst.info.buffer + (count / size) *
                       ucc_dt_size(args->dst.info.datatype) * rank);
    }
    task->super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgather_start", 0);
    NCCLCHECK_GOTO(ncclAllGather(src, dst, count / size, dt,
                                 team->nccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 0);
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgather_init(ucc_tl_nccl_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);

    if (!UCC_IS_INPLACE(*args)) {
        ucc_tl_nccl_check_and_convert_buffer(
            &args->src.info, UCC_TL_NCCL_DT_FOR_UNSUPPORTED);
    }

    ucc_tl_nccl_check_and_convert_buffer(&args->dst.info,
                                         UCC_TL_NCCL_DT_FOR_UNSUPPORTED);

    task->super.post = ucc_tl_nccl_allgather_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allgatherv_init(ucc_tl_nccl_task_t *task)
{
    task->super.post = ucc_tl_nccl_allgatherv_p2p_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *src    = args->src.info.buffer;
    ucc_status_t        status = UCC_OK;
    size_t              count  = args->src.info.count;
    ucc_rank_t          root   = args->root;
    ucc_rank_t          peer, rank, size;
    ncclDataType_t      dt;
    ucc_ep_map_t        map;

    dt = ucc_to_nccl_dtype[UCC_DT_PREDEFINED_ID(args->src.info.datatype)];
    task->super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_bcast_start", 0);

    if (UCC_COLL_ARGS_ACTIVE_SET(args)) {
        map  = ucc_active_set_to_ep_map(args);
        rank = UCC_TL_TEAM_RANK(team);
        size = (ucc_rank_t)args->active_set.size;
        if (root == rank) {
            NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status,
                           UCC_TL_TEAM_LIB(team), &task->nccl_progress_st,
                           team->nccl_comm, 0);
            for (peer = 0; peer < size; peer++) {
                if (ucc_ep_map_eval(map, peer) == rank) {
                    continue;
                }
                NCCLCHECK_GOTO(ncclSend(src, count, dt,
                                        ucc_ep_map_eval(map, peer),
                                        team->nccl_comm, stream),
                               exit_coll, status, UCC_TL_TEAM_LIB(team),
                               &task->nccl_progress_st, team->nccl_comm, 0);
            }
            NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team),
                           &task->nccl_progress_st, team->nccl_comm, 1);
        } else {
            NCCLCHECK_GOTO(ncclRecv(src, count, dt, root,
                                    team->nccl_comm, stream),
                           exit_coll, status, UCC_TL_TEAM_LIB(team),
                           &task->nccl_progress_st, team->nccl_comm, 1);
        }
    } else {
        NCCLCHECK_GOTO(ncclBroadcast(src, src, count, dt, root, team->nccl_comm,
                                     stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 0);
    }
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_bcast_init(ucc_tl_nccl_task_t *task)
{
    ucc_tl_nccl_check_and_convert_buffer(&TASK_ARGS(task).src.info,
                                         UCC_TL_NCCL_DT_FOR_UNSUPPORTED);

    task->super.post = ucc_tl_nccl_bcast_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_reduce_scatter_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    = args->src.info.buffer;
    ucc_status_t        status = UCC_OK;
    ncclRedOp_t         op     = ucc_to_nccl_reduce_op[args->op];
    size_t              count  = args->dst.info.count;
    ncclDataType_t      dt;

    dt = ucc_to_nccl_dtype[UCC_DT_PREDEFINED_ID(args->dst.info.datatype)];
    task->super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_reduce_scatter_start", 0);
    if (UCC_IS_INPLACE(*args)) {
        count /= UCC_TL_TEAM_SIZE(team);
        src = args->dst.info.buffer;
        dst = PTR_OFFSET(src, UCC_TL_TEAM_RANK(team) * count
                         * ucc_dt_size(args->dst.info.datatype));
    }
    NCCLCHECK_GOTO(ncclReduceScatter(src, dst, count, dt, op, team->nccl_comm,
                                     stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 0);
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_reduce_scatter_init(ucc_tl_nccl_task_t *task)
{
    ucc_coll_args_t *args = &TASK_ARGS(task);

    if (!UCC_IS_INPLACE(*args)) {
        if (ucc_tl_nccl_check_and_convert_buffer_reduction(&args->src.info,
                                                           task) != UCC_OK) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    if (ucc_tl_nccl_check_and_convert_buffer_reduction(&args->dst.info, task) !=
        UCC_OK) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post = ucc_tl_nccl_reduce_scatter_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_reduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task    = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team    = TASK_TEAM(task);
    ucc_ee_h            ee      = coll_task->ee;
    cudaStream_t        stream  = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst     = args->dst.info.buffer;
    void               *src     = args->src.info.buffer;
    ucc_datatype_t      ucc_dt  = args->src.info.datatype;
    size_t              count   = args->src.info.count;
    ncclRedOp_t         op      = ucc_to_nccl_reduce_op[args->op];
    ucc_status_t        status  = UCC_OK;
    ncclDataType_t      nccl_dt;

    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_reduce_start", 0);
    if (args->root == UCC_TL_TEAM_RANK(team)) {
        ucc_dt = TASK_ARGS(task).dst.info.datatype;
        count = TASK_ARGS(task).dst.info.count;
        if (UCC_IS_INPLACE(*args)) {
            src = args->dst.info.buffer;
        }
    }
    nccl_dt = ucc_to_nccl_dtype[UCC_DT_PREDEFINED_ID(ucc_dt)];
    task->super.status = UCC_INPROGRESS;
    NCCLCHECK_GOTO(ncclReduce(src, dst, count, nccl_dt, op, args->root,
                              team->nccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team),
                   &task->nccl_progress_st, team->nccl_comm, 0);
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_reduce_init(ucc_tl_nccl_task_t *task)
{
    ucc_coll_args_t *args    = &TASK_ARGS(task);
    int              is_root =
        UCC_IS_ROOT(TASK_ARGS(task), UCC_TL_TEAM_RANK(TASK_TEAM(task)));

    if (is_root) {
        if (ucc_tl_nccl_check_and_convert_buffer_reduction(&args->dst.info,
                                                           task) != UCC_OK) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    if (!(is_root && UCC_IS_INPLACE(*args))) {
        if (ucc_tl_nccl_check_and_convert_buffer_reduction(&args->src.info,
                                                           task) != UCC_OK) {
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    task->super.post = ucc_tl_nccl_reduce_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_barrier_init(ucc_tl_nccl_task_t *task)
{
    /* use 4-byte allreduce to accomplish barrier */
    ucc_coll_args_t *args   = &TASK_ARGS(task);

    args->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
    args->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    args->op     = UCC_OP_SUM;

    args->dst.info.buffer   = TASK_CTX(task)->scratch_buf;
    args->src.info.buffer   = args->dst.info.buffer;
    args->dst.info.datatype = args->src.info.datatype = UCC_DT_FLOAT32;
    args->dst.info.count = args->src.info.count = 1;

    task->super.post = ucc_tl_nccl_allreduce_start;

    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_gather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          rank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    = args->src.info.buffer;
    ucc_status_t        status = UCC_OK;
    size_t     send_size;
    ucc_rank_t peer;

    if (rank == args->root) {
        send_size = ucc_dt_size(args->dst.info.datatype) *
                    args->dst.info.count / size;
    } else {
        send_size = ucc_dt_size(args->src.info.datatype) * args->src.info.count;
    }

    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_gather_start", 0);
    if (rank == args->root) {
        if (!UCC_IS_INPLACE(*args)) {
            CUDA_CHECK_GOTO(cudaMemcpyAsync(PTR_OFFSET(dst, rank * send_size),
                                            src, send_size,
                                            cudaMemcpyDeviceToDevice, stream),
                            exit_coll, status);
        }
        NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status,
                       UCC_TL_TEAM_LIB(team), &task->nccl_progress_st,
                       team->nccl_comm, 0);
        for (peer = 0; peer < size; peer++) {
            if (peer == args->root) {
                continue;
            }
            NCCLCHECK_GOTO(ncclRecv(PTR_OFFSET(dst, peer * send_size),
                                    send_size, ncclChar, peer, team->nccl_comm,
                                    stream),
                           exit_coll, status, UCC_TL_TEAM_LIB(team),
                           &task->nccl_progress_st, team->nccl_comm, 0);
        }
        NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status,
                       UCC_TL_TEAM_LIB(team), &task->nccl_progress_st,
                       team->nccl_comm, 1);
    } else {
        NCCLCHECK_GOTO(ncclSend(src, send_size, ncclChar, args->root,
                                team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 1);
    }
    task->super.status = UCC_INPROGRESS;
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_gather_init(ucc_tl_nccl_task_t *task)
{
    task->super.post = ucc_tl_nccl_gather_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_gatherv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          rank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info_v.buffer;
    void               *src    = args->src.info.buffer;
    ucc_status_t        status = UCC_OK;
    size_t     count, displ, dt_size;
    ucc_rank_t peer;

    if (rank == args->root) {
        dt_size = ucc_dt_size(args->dst.info_v.datatype);
    } else {
        dt_size = ucc_dt_size(args->src.info.datatype);
    }

    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_gatherv_start", 0);
    if (rank == args->root) {
        if (!UCC_IS_INPLACE(*args)) {
            count = ucc_coll_args_get_count(args, args->dst.info_v.counts, rank);
            displ = ucc_coll_args_get_displacement(args,
                                                   args->dst.info_v.displacements,
                                                   rank);
            CUDA_CHECK_GOTO(cudaMemcpyAsync(PTR_OFFSET(dst, displ * dt_size),
                                            src, count * dt_size,
                                            cudaMemcpyDeviceToDevice, stream),
                            exit_coll, status);
        }
        NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status,
                       UCC_TL_TEAM_LIB(team), &task->nccl_progress_st,
                       team->nccl_comm, 0);
        for (peer = 0; peer < size; peer++) {
            if (peer == args->root) {
                continue;
            }
            count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
            displ = ucc_coll_args_get_displacement(args,
                                                   args->dst.info_v.displacements,
                                                   peer);
            NCCLCHECK_GOTO(ncclRecv(PTR_OFFSET(dst, displ * dt_size),
                                    count * dt_size, ncclChar,
                                    peer,team->nccl_comm, stream),
                           exit_coll, status, UCC_TL_TEAM_LIB(team),
                           &task->nccl_progress_st, team->nccl_comm, 0);
        }
        NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status,
                       UCC_TL_TEAM_LIB(team), &task->nccl_progress_st,
                       team->nccl_comm, 1);
    } else {
        NCCLCHECK_GOTO(ncclSend(src, args->src.info.count * dt_size,
                                ncclChar, args->root, team->nccl_comm,
                                stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 1);
    }
    task->super.status = UCC_INPROGRESS;
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_gatherv_init(ucc_tl_nccl_task_t *task)
{
    task->super.post = ucc_tl_nccl_gatherv_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_scatter_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          rank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    = args->src.info.buffer;
    ucc_status_t        status = UCC_OK;
    size_t     send_size;
    ucc_rank_t peer;

    if (rank == args->root) {
        send_size = ucc_dt_size(args->src.info.datatype) *
                    args->src.info.count / size;
    } else {
        send_size = ucc_dt_size(args->dst.info.datatype) * args->dst.info.count;
    }

    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_scatter_start", 0);
    if (rank == args->root) {
        if (!UCC_IS_INPLACE(*args)) {
            CUDA_CHECK_GOTO(cudaMemcpyAsync(dst,
                                            PTR_OFFSET(src, rank * send_size),
                                            send_size, cudaMemcpyDeviceToDevice,
                                            stream),
                            exit_coll, status);
        }
        NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status,
                       UCC_TL_TEAM_LIB(team), &task->nccl_progress_st,
                       team->nccl_comm, 0);
        for (peer = 0; peer < size; peer++) {
            if (peer == args->root) {
                continue;
            }
            NCCLCHECK_GOTO(ncclSend(PTR_OFFSET(src, peer * send_size),
                                    send_size, ncclChar, peer, team->nccl_comm,
                                    stream),
                           exit_coll, status, UCC_TL_TEAM_LIB(team),
                           &task->nccl_progress_st, team->nccl_comm, 0);
        }
        NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 1);
    } else {
        NCCLCHECK_GOTO(ncclRecv(dst, send_size, ncclChar, args->root,
                                team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 1);
    }
    task->super.status = UCC_INPROGRESS;
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_scatter_init(ucc_tl_nccl_task_t *task)
{
    task->super.post = ucc_tl_nccl_scatter_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_scatterv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          rank   = UCC_TL_TEAM_RANK(team);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    = args->src.info_v.buffer;
    ucc_status_t        status = UCC_OK;
    size_t     count, displ, dt_size;
    ucc_rank_t peer;

    if (rank == args->root) {
        dt_size = ucc_dt_size(args->src.info_v.datatype);
    } else {
        dt_size = ucc_dt_size(args->dst.info.datatype);
    }

    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_scatterv_start", 0);
    if (rank == args->root) {
        if (!UCC_IS_INPLACE(*args)) {
            count = ucc_coll_args_get_count(args, args->src.info_v.counts, rank);
            displ = ucc_coll_args_get_displacement(args,
                                                   args->src.info_v.displacements,
                                                   rank);
            CUDA_CHECK_GOTO(cudaMemcpyAsync(dst,
                                            PTR_OFFSET(src, displ * dt_size),
                                            count * dt_size,
                                            cudaMemcpyDeviceToDevice,
                                            stream),
                            exit_coll, status);
        }
        NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status,
                       UCC_TL_TEAM_LIB(team), &task->nccl_progress_st,
                       team->nccl_comm, 0);
        for (peer = 0; peer < size; peer++) {
            if (peer == args->root) {
                continue;
            }
            count = ucc_coll_args_get_count(args, args->src.info_v.counts, peer);
            displ = ucc_coll_args_get_displacement(args,
                                                   args->src.info_v.displacements,
                                                   peer);
            NCCLCHECK_GOTO(ncclSend(PTR_OFFSET(src, displ * dt_size),
                                    count * dt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                           exit_coll, status, UCC_TL_TEAM_LIB(team),
                           &task->nccl_progress_st, team->nccl_comm, 0);
        }
        NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 1);
    } else {
        NCCLCHECK_GOTO(ncclRecv(dst, args->dst.info.count * dt_size, ncclChar,
                                args->root, team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team),
                       &task->nccl_progress_st, team->nccl_comm, 1);
    }
    task->super.status = UCC_INPROGRESS;
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_scatterv_init(ucc_tl_nccl_task_t *task)
{
    task->super.post = ucc_tl_nccl_scatterv_start;
    return UCC_OK;
}

static inline int alg_id_from_str(ucc_coll_type_t coll_type, const char *str)
{
    switch (coll_type) {
    case UCC_COLL_TYPE_ALLGATHERV:
        return ucc_tl_nccl_allgatherv_alg_from_str(str);
    default:
        break;
    }
    return -1;
}

ucc_status_t ucc_tl_nccl_alg_id_to_init(int alg_id, const char *alg_id_str,
                                        ucc_coll_type_t   coll_type,
                                        ucc_memory_type_t mem_type, //NOLINT
                                        ucc_base_coll_init_fn_t *init)
{
    ucc_status_t status = UCC_OK;
    if (alg_id_str) {
        alg_id = alg_id_from_str(coll_type, alg_id_str);
    }

    switch (coll_type) {
    case UCC_COLL_TYPE_ALLGATHERV:
        switch (alg_id) {
        case UCC_TL_NCCL_ALLGATHERV_ALG_P2P:
            *init = ucc_tl_nccl_allgatherv_p2p_init;
            break;
        case UCC_TL_NCCL_ALLGATHERV_ALG_BCOPY:
            *init = ucc_tl_nccl_allgatherv_bcopy_init;
            break;
        case UCC_TL_NCCL_ALLGATHERV_ALG_BCAST:
            *init = ucc_tl_nccl_allgatherv_bcast_init;
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
