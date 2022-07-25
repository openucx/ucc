/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_rccl.h"
#include "allgatherv.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"

ucc_base_coll_alg_info_t
    ucc_tl_rccl_allgatherv_algs[UCC_TL_RCCL_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_TL_RCCL_ALLGATHERV_ALG_P2P] =
            {.id   = UCC_TL_RCCL_ALLGATHERV_ALG_P2P,
             .name = "p2p",
             .desc = "allgatherv based on rccl point-to-point"},
        [UCC_TL_RCCL_ALLGATHERV_ALG_BCOPY] =
            {.id   = UCC_TL_RCCL_ALLGATHERV_ALG_BCOPY,
             .name = "bcopy",
             .desc = "allgatherv with buffered copy"},
        [UCC_TL_RCCL_ALLGATHERV_ALG_BCAST] =
            {.id   = UCC_TL_RCCL_ALLGATHERV_ALG_BCAST,
             .name = "bcast",
             .desc = "allgatherv based on rccl bcast"},
        [UCC_TL_RCCL_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

#define CHECK_INPLACE(_args, _team)                                            \
    do {                                                                       \
        if (UCC_IS_INPLACE((_args))) {                                         \
            tl_error(UCC_TL_TEAM_LIB((_team)),                                 \
                     "inplace allgatherv is not supported");                   \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while(0)

#define CHECK_USERDEFINED_DT(_args, _team)                                     \
    do {                                                                       \
        if (!UCC_DT_IS_PREDEFINED((_args).src.info.datatype) ||                \
            !UCC_DT_IS_PREDEFINED((_args).dst.info_v.datatype)) {              \
            tl_error(UCC_TL_TEAM_LIB((_team)),                                 \
                     "user defined datatype is not supported");                \
            status = UCC_ERR_NOT_SUPPORTED;                                    \
            goto out;                                                          \
        }                                                                      \
    } while(0)

ucc_status_t ucc_tl_rccl_allgatherv_p2p_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_rccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_rccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_rccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    hipStream_t         stream = (ee) ? (hipStream_t) ee->ee_context :
                                                       team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf   = args->src.info.buffer;
    void               *rbuf   = args->dst.info_v.buffer;
    size_t sdt_size, rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.super.status = UCC_INPROGRESS;
    sdt_size                 = ucc_dt_size(args->src.info.datatype);
    rdt_size                 = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_RCCL_PROFILE_REQUEST_EVENT(coll_task, "rccl_allgatherv_start", 0);
    RCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    count = args->src.info.count;
    if (count != 0) {
        for (peer = 0; peer < size; peer++) {
            RCCLCHECK_GOTO(ncclSend(sbuf, count * sdt_size, ncclChar, peer,
                                    team->rccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    for (peer = 0; peer < size; peer++) {
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, peer);
            RCCLCHECK_GOTO(ncclRecv(PTR_OFFSET(rbuf, displ * rdt_size),
                                    count * rdt_size, ncclChar, peer,
                                    team->rccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    RCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_rccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_rccl_allgatherv_p2p_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *     team,
                                             ucc_coll_task_t **    task_h)
{
    ucc_tl_rccl_team_t *rccl_team = ucc_derived_of(team, ucc_tl_rccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_rccl_task_t *task;

    CHECK_INPLACE(*args, rccl_team);
    CHECK_USERDEFINED_DT(*args, rccl_team);
    task = ucc_tl_rccl_init_task(coll_args, team);
    if (!task) {
        return UCC_ERR_NO_MESSAGE;
    }
    task->super.post     = ucc_tl_rccl_allgatherv_p2p_start;
    *task_h = &task->super;
out:
    return status;
}

ucc_status_t ucc_tl_rccl_allgatherv_bcopy_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_rccl_task_t *task    = ucc_derived_of(coll_task, ucc_tl_rccl_task_t);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_tl_rccl_team_t *team    = TASK_TEAM(task);
    ucc_rank_t          size    = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee      = coll_task->ee;
    hipStream_t        stream  = (ee) ? (hipStream_t) ee->ee_context :
                                                        team->stream;
    ucc_status_t        status  = UCC_OK;
    void               *sbuf    = args->src.info.buffer;
    void               *rbuf    = args->dst.info_v.buffer;
    void               *scratch = task->allgatherv_bcopy.scratch->addr;
    size_t              max_count, rdt_size, sdt_size, displ, scount, rcount;
    ucc_rank_t          peer;

    task->super.super.status = UCC_INPROGRESS;
    UCC_TL_RCCL_PROFILE_REQUEST_EVENT(coll_task, "rccl_allgatherv_start", 0);
    max_count = task->allgatherv_bcopy.max_count;
    scount    = args->src.info.count;
    rdt_size  = ucc_dt_size(args->dst.info_v.datatype);
    sdt_size  = ucc_dt_size(args->src.info.datatype);
    if (max_count * rdt_size > scount * sdt_size) {
        HIPCHECK_GOTO(hipMemcpyAsync(PTR_OFFSET(scratch,
                                     max_count * rdt_size * size), sbuf,
                                     scount * sdt_size,
                                     hipMemcpyDeviceToDevice, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
        sbuf = PTR_OFFSET(scratch, max_count * rdt_size * size);
    }
    RCCLCHECK_GOTO(ncclAllGather(sbuf, scratch, max_count * rdt_size,
                                 ncclChar, team->rccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < size; peer++) {
        rcount = ucc_coll_args_get_count(args,
                                         args->dst.info_v.counts, peer);
        displ  = ucc_coll_args_get_displacement(args,
                                                args->dst.info_v.displacements,
                                                peer);
        HIPCHECK_GOTO(hipMemcpyAsync(PTR_OFFSET(rbuf, displ * rdt_size),
                                     PTR_OFFSET(scratch,
                                                peer * max_count * rdt_size),
                                     rcount * rdt_size,
                                     hipMemcpyDeviceToDevice, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
    }
    status = ucc_tl_rccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_rccl_allgatherv_bcopy_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_rccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_rccl_task_t);

    ucc_mc_free(task->allgatherv_bcopy.scratch);
    return ucc_tl_rccl_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_rccl_allgatherv_bcopy_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_rccl_team_t *rccl_team = ucc_derived_of(team, ucc_tl_rccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_rccl_task_t *task;
    size_t              max_count, sdt_size, rdt_size;
    ucc_rank_t          peer;

    CHECK_INPLACE(*args, rccl_team);
    CHECK_USERDEFINED_DT(*args, rccl_team);
    task = ucc_tl_rccl_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MESSAGE;
    }

    sdt_size = ucc_dt_size(args->src.info.datatype);
    rdt_size = ucc_dt_size(args->dst.info_v.datatype);
    max_count = ucc_coll_args_get_max_count(args, args->dst.info_v.counts, UCC_TL_TEAM_SIZE(rccl_team));
    for (peer = 1; peer < team->params.size; peer++) {
        max_count = ucc_max(ucc_coll_args_get_count(args,
                            args->dst.info_v.counts, peer), max_count);
    }
    task->allgatherv_bcopy.max_count = max_count;
    if (max_count * rdt_size > args->src.info.count * sdt_size) {
        status = ucc_mc_alloc(&task->allgatherv_bcopy.scratch,
                              (team->params.size + 1) * max_count *
                              ucc_dt_size(args->dst.info_v.datatype),
                              UCC_MEMORY_TYPE_ROCM);

    } else {
        status = ucc_mc_alloc(&task->allgatherv_bcopy.scratch, max_count *
                              team->params.size *
                              ucc_dt_size(args->dst.info_v.datatype),
                              UCC_MEMORY_TYPE_ROCM);
    }
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_rccl_free_task(task);
        return status;
    }
    task->super.post     = ucc_tl_rccl_allgatherv_bcopy_start;
    task->super.finalize = ucc_tl_rccl_allgatherv_bcopy_finalize;
    *task_h = &task->super;
out:
    return status;
}

ucc_status_t ucc_tl_rccl_allgatherv_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_rccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_rccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_rccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    hipStream_t        stream  = (ee) ? (hipStream_t) ee->ee_context :
	                                              team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf   = args->src.info.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)args->dst.info_v.buffer;
    size_t rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.super.status = UCC_INPROGRESS;
    rdt_size                 = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_RCCL_PROFILE_REQUEST_EVENT(coll_task, "rccl_allgatherv_start", 0);
    RCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < size; peer++) {
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        displ = ucc_coll_args_get_displacement(args,
                                               args->dst.info_v.displacements,
                                               peer);
        RCCLCHECK_GOTO(ncclBroadcast(sbuf, PTR_OFFSET(rbuf, displ * rdt_size),
                                     count * rdt_size, ncclChar, peer,
                                     team->rccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
    }
    RCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_rccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_rccl_allgatherv_bcast_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_rccl_team_t *rccl_team = ucc_derived_of(team, ucc_tl_rccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_rccl_task_t *task;

    CHECK_INPLACE(*args, rccl_team);
    CHECK_USERDEFINED_DT(*args, rccl_team);
    task = ucc_tl_rccl_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MESSAGE;
    }
    task->super.post = ucc_tl_rccl_allgatherv_bcast_start;
    *task_h          = &task->super;
out:
    return status;
}
