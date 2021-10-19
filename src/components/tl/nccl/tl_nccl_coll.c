/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define ncclOpUnsupported (ncclNumOps + 1)
#define ncclDataTypeUnsupported (ncclNumTypes + 1)

ncclDataType_t ucc_to_nccl_dtype[] = {
    [UCC_DT_INT8]        = (ncclDataType_t)ncclInt8,
    [UCC_DT_INT16]       = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_INT32]       = (ncclDataType_t)ncclInt32,
    [UCC_DT_INT64]       = (ncclDataType_t)ncclInt64,
    [UCC_DT_INT128]      = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_UINT8]       = (ncclDataType_t)ncclUint8,
    [UCC_DT_UINT16]      = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_UINT32]      = (ncclDataType_t)ncclUint32,
    [UCC_DT_UINT64]      = (ncclDataType_t)ncclUint64,
    [UCC_DT_UINT128]     = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_FLOAT16]     = (ncclDataType_t)ncclFloat16,
    [UCC_DT_FLOAT32]     = (ncclDataType_t)ncclFloat32,
    [UCC_DT_FLOAT64]     = (ncclDataType_t)ncclFloat64,
    [UCC_DT_USERDEFINED] = (ncclDataType_t)ncclDataTypeUnsupported,
    [UCC_DT_OPAQUE]      = (ncclDataType_t)ncclDataTypeUnsupported,
};

ncclRedOp_t ucc_to_nccl_reduce_op[] = {
    [UCC_OP_USERDEFINED] = (ncclRedOp_t)ncclOpUnsupported,
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

static inline ucc_status_t ucc_nccl_check_dt_supported(ucc_datatype_t dt1,
                                                       ucc_datatype_t dt2)
{
    if (ucc_unlikely((dt1 != dt2) ||
                     (ucc_to_nccl_dtype[dt1] == ncclDataTypeUnsupported))) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_collective_sync(ucc_tl_nccl_task_t *task,
                                         cudaStream_t stream)
{
    ucc_tl_nccl_context_t *ctx    = TASK_CTX(task);
    ucc_status_t           status = UCC_OK;
    CUresult cu_status;

    task->host_status = task->super.super.status;
    if (ctx->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT) {
        status = ucc_mc_ee_event_post(stream, task->completed,
                                      UCC_EE_CUDA_STREAM);
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

    status = task->super.progress(&task->super);
    if (status == UCC_INPROGRESS) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(TASK_TEAM(task))->pq,
                             &task->super);
        return UCC_OK;
    }

    return ucc_task_complete(&task->super);
}

ucc_status_t ucc_tl_nccl_alltoall_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    ucc_rank_t          gsize  = team->size;
    ucc_status_t        status = UCC_OK;
    ptrdiff_t           sbuf   = (ptrdiff_t)args->src.info.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)args->dst.info.buffer;
    size_t data_size;
    ucc_rank_t peer;

    task->super.super.status = UCC_INPROGRESS;
    data_size                = (size_t)(args->src.info.count / gsize) *
                ucc_dt_size(args->src.info.datatype);
    ucc_assert(args->src.info.count % gsize == 0);
    if (data_size == 0) {
        task->super.super.status = UCC_OK;
        return UCC_OK;
    }
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_alltoall_start", 0);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < gsize; peer++) {
        NCCLCHECK_GOTO(ncclSend((void *)(sbuf + peer * data_size), data_size,
                                ncclChar, peer, team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
        NCCLCHECK_GOTO(ncclRecv((void *)(rbuf + peer * data_size), data_size,
                                ncclChar, peer, team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
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
    if ((TASK_ARGS(task).src.info.datatype == UCC_DT_USERDEFINED) ||
        (TASK_ARGS(task).dst.info.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_alltoall_start;
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
    size_t sdt_size, rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.super.status = UCC_INPROGRESS;
    sdt_size                 = ucc_dt_size(args->src.info_v.datatype);
    rdt_size                 = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_alltoallv_start", 0);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < team->size; peer++) {
        count = ucc_coll_args_get_count(args, args->src.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(
                args, args->src.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclSend((void *)(sbuf + displ * sdt_size),
                                    count * sdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclRecv((void *)(rbuf + displ * rdt_size),
                                    count * rdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
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
    if ((TASK_ARGS(task).src.info_v.datatype == UCC_DT_USERDEFINED) ||
        (TASK_ARGS(task).dst.info_v.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_alltoallv_start;
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
    ncclDataType_t      dt     = ucc_to_nccl_dtype[args->dst.info.datatype];
    ncclRedOp_t         op = ucc_to_nccl_reduce_op[args->reduce.predefined_op];
    size_t              count = args->dst.info.count;

    task->super.super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allreduce_start", 0);
    NCCLCHECK_GOTO(ncclAllReduce(src, dst, count, dt, op, team->nccl_comm,
                                 stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allreduce_init(ucc_tl_nccl_task_t *task)
{
    if ((TASK_ARGS(task).mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) ||
        (ucc_to_nccl_reduce_op[TASK_ARGS(task).reduce.predefined_op] ==
         ncclOpUnsupported)) {
        tl_debug(UCC_TASK_LIB(task), "reduction operation is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (UCC_OK !=
        ucc_nccl_check_dt_supported(TASK_ARGS(task).dst.info.datatype,
                                    TASK_ARGS(task).dst.info.datatype)) {
        tl_debug(UCC_TASK_LIB(task), "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_allreduce_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allgather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    = args->src.info.buffer;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[args->dst.info.datatype];
    ucc_status_t        status = UCC_OK;
    size_t              count  = args->dst.info.count;

    if (UCC_IS_INPLACE(*args)) {
        src = (void *)((ptrdiff_t)args->dst.info.buffer + (count / team->size) *
                       ucc_dt_size(args->dst.info.datatype) * team->rank);
    }
    task->super.super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgather_start", 0);
    NCCLCHECK_GOTO(ncclAllGather(src, dst, count / team->size, dt,
                                 team->nccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgather_init(ucc_tl_nccl_task_t *task)
{
    ucc_datatype_t dt1 = UCC_IS_INPLACE(TASK_ARGS(task))
                             ? TASK_ARGS(task).dst.info.datatype
                             : TASK_ARGS(task).src.info.datatype;
    ucc_datatype_t dt2 = TASK_ARGS(task).dst.info.datatype;

    if (UCC_OK != ucc_nccl_check_dt_supported(dt1, dt2)) {
        /* TODO: can we use ncclChar if datatype is not supported? */
        tl_error(UCC_TASK_LIB(task), "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_allgather_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allgatherv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf   = args->src.info.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)args->dst.info_v.buffer;
    size_t sdt_size, rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.super.status = UCC_INPROGRESS;
    sdt_size                 = ucc_dt_size(args->src.info.datatype);
    rdt_size                 = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgatherv_start", 0);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    count = args->src.info.count;
    if (count != 0) {
        for (peer = 0; peer < team->size; peer++) {
            NCCLCHECK_GOTO(ncclSend(sbuf, count * sdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    for (peer = 0; peer < team->size; peer++) {
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclRecv((void *)(rbuf + displ * rdt_size),
                                    count * rdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_init(ucc_tl_nccl_task_t *task)
{
    if (UCC_IS_INPLACE(TASK_ARGS(task))) {
        tl_error(UCC_TASK_LIB(task), "inplace allgatherv is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if ((TASK_ARGS(task).src.info_v.datatype == UCC_DT_USERDEFINED) ||
        (TASK_ARGS(task).dst.info_v.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_allgatherv_start;
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
    ncclDataType_t      dt     = ucc_to_nccl_dtype[args->src.info.datatype];
    ucc_status_t        status = UCC_OK;
    size_t              count  = args->src.info.count;
    ucc_rank_t          root   = args->root;

    task->super.super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_bcast_start", 0);
    NCCLCHECK_GOTO(ncclBroadcast(src, src, count, dt, root, team->nccl_comm,
                                 stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_bcast_init(ucc_tl_nccl_task_t *task)
{
    if (UCC_OK !=
        ucc_nccl_check_dt_supported(TASK_ARGS(task).src.info.datatype,
                                    TASK_ARGS(task).src.info.datatype)) {
        /* TODO: can we use ncclChar if datatype is not supported? */
        tl_error(UCC_TASK_LIB(task), "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_bcast_start;
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
    ncclDataType_t      dt     = ucc_to_nccl_dtype[args->dst.info.datatype];
    ncclRedOp_t         op     = ucc_to_nccl_reduce_op[
                                    args->reduce.predefined_op];
    size_t              count  = args->dst.info.count;

    task->super.super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_reduce_scatter_start", 0);
    if (UCC_IS_INPLACE(*args)) {
        count /= team->size;
        src = args->dst.info.buffer;
        dst = PTR_OFFSET(src, team->rank * count * ucc_dt_size(args->dst.info.datatype));
    }
    NCCLCHECK_GOTO(ncclReduceScatter(src, dst, count, dt, op, team->nccl_comm,
                                     stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_reduce_scatter_init(ucc_tl_nccl_task_t *task)
{
    if ((TASK_ARGS(task).mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) ||
        (ucc_to_nccl_reduce_op[TASK_ARGS(task).reduce.predefined_op] ==
         ncclOpUnsupported)) {
        tl_debug(UCC_TASK_LIB(task), "reduction operation is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (UCC_OK !=
        ucc_nccl_check_dt_supported(TASK_ARGS(task).dst.info.datatype,
                                    TASK_ARGS(task).dst.info.datatype)) {
        tl_debug(UCC_TASK_LIB(task), "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_reduce_scatter_start;
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
    ncclRedOp_t         op      = ucc_to_nccl_reduce_op[
                                     args->reduce.predefined_op];
    ucc_status_t        status  = UCC_OK;
    ncclDataType_t      nccl_dt;

    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_reduce_start", 0);
    if (args->root == team->rank) {
        ucc_dt = TASK_ARGS(task).dst.info.datatype;
        count = TASK_ARGS(task).dst.info.count;
        if (UCC_IS_INPLACE(*args)) {
            src = args->dst.info.buffer;
        }
    }
    nccl_dt = ucc_to_nccl_dtype[ucc_dt];
    task->super.super.status = UCC_INPROGRESS;
    NCCLCHECK_GOTO(ncclReduce(src, dst, count, nccl_dt, op, args->root,
                              team->nccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_reduce_init(ucc_tl_nccl_task_t *task)
{
    ucc_datatype_t dt = (TASK_ARGS(task).root == TASK_TEAM(task)->rank) ?
                           TASK_ARGS(task).dst.info.datatype:
                           TASK_ARGS(task).src.info.datatype;

    if ((TASK_ARGS(task).mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) ||
        (ucc_to_nccl_reduce_op[TASK_ARGS(task).reduce.predefined_op] ==
         ncclOpUnsupported)) {
        tl_debug(UCC_TASK_LIB(task), "reduction operation is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_OK !=
        ucc_nccl_check_dt_supported(dt, dt)) {
        tl_debug(UCC_TASK_LIB(task), "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_reduce_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_barrier_init(ucc_tl_nccl_task_t* task) {
  // use 4-byte allreduce to accomplish barrier
  ucc_coll_args_t    *args   = &TASK_ARGS(task);

  args->mask |=
      (UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS | UCC_COLL_ARGS_FIELD_FLAGS);
  args->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
  args->reduce.predefined_op = UCC_OP_SUM;

  ucc_status_t status = ucc_mc_alloc(
      &task->scratch_mc_header, sizeof(float), UCC_MEMORY_TYPE_CUDA);
  if (ucc_unlikely(status != UCC_OK)) {
      return status;
  }
  args->dst.info.buffer = task->scratch_mc_header->addr;
  args->src.info.buffer = args->dst.info.buffer;
  args->dst.info.datatype = args->src.info.datatype = UCC_DT_FLOAT32;
  args->dst.info.count = args->src.info.count = 1;

  task->super.post = ucc_tl_nccl_allreduce_start;

  return UCC_OK;
}
