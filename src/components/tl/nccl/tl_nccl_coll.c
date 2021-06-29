/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
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
    ucc_coll_args_t    *args   = &coll_task->args;
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
    data_size =
        (size_t)args->src.info.count * ucc_dt_size(args->src.info.datatype);
    if (data_size == 0) {
        task->super.super.status = UCC_OK;
        return UCC_OK;
    }
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
    if (UCC_IS_INPLACE(task->super.args)) {
        tl_error(UCC_TASK_LIB(task), "inplace alltoallv is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if ((task->super.args.src.info.datatype == UCC_DT_USERDEFINED) ||
        (task->super.args.dst.info.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_alltoall_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_alltoallv_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &coll_task->args;
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
    if (UCC_IS_INPLACE(task->super.args)) {
        tl_error(UCC_TASK_LIB(task), "inplace alltoall is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if ((task->super.args.src.info_v.datatype == UCC_DT_USERDEFINED) ||
        (task->super.args.dst.info_v.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_alltoallv_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allreduce_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &coll_task->args;
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    =
        UCC_IS_INPLACE(*args) ? args->dst.info.buffer : args->src.info.buffer;
    ucc_status_t        status = UCC_OK;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[args->src.info.datatype];
    ncclRedOp_t         op = ucc_to_nccl_reduce_op[args->reduce.predefined_op];
    size_t              count = args->src.info.count;

    task->super.super.status = UCC_INPROGRESS;
    NCCLCHECK_GOTO(ncclAllReduce(src, dst, count, dt, op, team->nccl_comm,
                                 stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allreduce_init(ucc_tl_nccl_task_t *task)
{
    if ((task->super.args.mask & UCC_COLL_ARGS_FIELD_USERDEFINED_REDUCTIONS) ||
        (ucc_to_nccl_reduce_op[task->super.args.reduce.predefined_op] ==
         ncclOpUnsupported)) {
        tl_error(UCC_TASK_LIB(task), "reduction operation is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (UCC_OK !=
        ucc_nccl_check_dt_supported(task->super.args.src.info.datatype,
                                    task->super.args.src.info.datatype)) {
        tl_error(UCC_TASK_LIB(task), "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_allreduce_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_allgather_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &coll_task->args;
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *dst    = args->dst.info.buffer;
    void               *src    = args->src.info.buffer;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[args->dst.info.datatype];
    ucc_status_t        status = UCC_OK;
    size_t              count  = args->dst.info.count;

    if (UCC_IS_INPLACE(*args)) {
        src =
            (void *)((ptrdiff_t)args->dst.info.buffer +
                     count * ucc_dt_size(args->dst.info.datatype) * team->rank);
    }
    task->super.super.status = UCC_INPROGRESS;
    NCCLCHECK_GOTO(ncclAllGather(src, dst, count, dt, team->nccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgather_init(ucc_tl_nccl_task_t *task)
{
    ucc_datatype_t dt1 = UCC_IS_INPLACE(task->super.args)
                             ? task->super.args.dst.info.datatype
                             : task->super.args.src.info.datatype;
    ucc_datatype_t dt2 = task->super.args.dst.info.datatype;

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
    ucc_coll_args_t    *args   = &coll_task->args;
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
    if (UCC_IS_INPLACE(task->super.args)) {
        tl_error(UCC_TASK_LIB(task), "inplace allgatherv is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if ((task->super.args.src.info_v.datatype == UCC_DT_USERDEFINED) ||
        (task->super.args.dst.info_v.datatype == UCC_DT_USERDEFINED)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_allgatherv_start;
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &coll_task->args;
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context : team->stream;
    void               *src    = args->src.info.buffer;
    ncclDataType_t      dt     = ucc_to_nccl_dtype[args->src.info.datatype];
    ucc_status_t        status = UCC_OK;
    size_t              count  = args->src.info.count;
    ucc_rank_t          root   = args->root;

    task->super.super.status = UCC_INPROGRESS;
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
        ucc_nccl_check_dt_supported(task->super.args.src.info.datatype,
                                    task->super.args.src.info.datatype)) {
        /* TODO: can we use ncclChar if datatype is not supported? */
        tl_error(UCC_TASK_LIB(task), "dataype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }
    task->super.post     = ucc_tl_nccl_bcast_start;
    return UCC_OK;
}
