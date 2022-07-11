/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "allgatherv.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"
#include "utils/arch/cuda_def.h"

ucc_base_coll_alg_info_t
    ucc_tl_nccl_allgatherv_algs[UCC_TL_NCCL_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_TL_NCCL_ALLGATHERV_ALG_P2P] =
            {.id   = UCC_TL_NCCL_ALLGATHERV_ALG_P2P,
             .name = "p2p",
             .desc = "allgatherv based on nccl point-to-point"},
        [UCC_TL_NCCL_ALLGATHERV_ALG_BCOPY] =
            {.id   = UCC_TL_NCCL_ALLGATHERV_ALG_BCOPY,
             .name = "bcopy",
             .desc = "allgatherv with buffered copy"},
        [UCC_TL_NCCL_ALLGATHERV_ALG_BCAST] =
            {.id   = UCC_TL_NCCL_ALLGATHERV_ALG_BCAST,
             .name = "bcast",
             .desc = "allgatherv based on nccl bcast"},
        [UCC_TL_NCCL_ALLGATHERV_ALG_LAST] = {
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

ucc_status_t ucc_tl_nccl_allgatherv_p2p_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context :
                                                       team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf   = args->src.info.buffer;
    void               *rbuf   = args->dst.info_v.buffer;
    size_t sdt_size, rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.status = UCC_INPROGRESS;
    sdt_size           = ucc_dt_size(args->src.info.datatype);
    rdt_size           = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgatherv_start", 0);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    count = args->src.info.count;
    if (count != 0) {
        for (peer = 0; peer < size; peer++) {
            NCCLCHECK_GOTO(ncclSend(sbuf, count * sdt_size, ncclChar, peer,
                                    team->nccl_comm, stream),
                        exit_coll, status, UCC_TL_TEAM_LIB(team));
        }
    }
    for (peer = 0; peer < size; peer++) {
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        if (count != 0) {
            displ = ucc_coll_args_get_displacement(
                args, args->dst.info_v.displacements, peer);
            NCCLCHECK_GOTO(ncclRecv(PTR_OFFSET(rbuf, displ * rdt_size),
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

ucc_status_t ucc_tl_nccl_allgatherv_p2p_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *     team,
                                             ucc_coll_task_t **    task_h)
{
    ucc_tl_nccl_team_t *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_nccl_task_t *task;

    CHECK_INPLACE(*args, nccl_team);
    CHECK_USERDEFINED_DT(*args, nccl_team);
    task = ucc_tl_nccl_init_task(coll_args, team);
    if (!task) {
        return UCC_ERR_NO_MESSAGE;
    }
    task->super.post     = ucc_tl_nccl_allgatherv_p2p_start;
    *task_h = &task->super;
out:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_bcopy_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task    = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args    = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team    = TASK_TEAM(task);
    ucc_rank_t          size    = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee      = coll_task->ee;
    cudaStream_t        stream  = (ee) ? (cudaStream_t) ee->ee_context :
                                                        team->stream;
    ucc_status_t        status  = UCC_OK;
    void               *sbuf    = args->src.info.buffer;
    void               *rbuf    = args->dst.info_v.buffer;
    void               *scratch = task->allgatherv_bcopy.scratch->addr;
    size_t              max_count, rdt_size, sdt_size, displ, scount, rcount;
    ucc_rank_t          peer;

    task->super.status = UCC_INPROGRESS;
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgatherv_start", 0);
    max_count = task->allgatherv_bcopy.max_count;
    scount    = args->src.info.count;
    rdt_size  = ucc_dt_size(args->dst.info_v.datatype);
    sdt_size  = ucc_dt_size(args->src.info.datatype);
    if (max_count * rdt_size > scount * sdt_size) {
        CUDA_CHECK_GOTO(cudaMemcpyAsync(PTR_OFFSET(scratch,
                                        max_count * rdt_size * size), sbuf,
                                        scount * sdt_size,
                                        cudaMemcpyDeviceToDevice, stream),
                        exit_coll, status);
        sbuf = PTR_OFFSET(scratch, max_count * rdt_size * size);
    }
    NCCLCHECK_GOTO(ncclAllGather(sbuf, scratch, max_count * rdt_size,
                                 ncclChar, team->nccl_comm, stream),
                   exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < size; peer++) {
        rcount = ucc_coll_args_get_count(args,
                                         args->dst.info_v.counts, peer);
        displ  = ucc_coll_args_get_displacement(args,
                                                args->dst.info_v.displacements,
                                                peer);
        CUDA_CHECK_GOTO(cudaMemcpyAsync(PTR_OFFSET(rbuf, displ * rdt_size),
                                        PTR_OFFSET(scratch,
                                                   peer * max_count * rdt_size),
                                        rcount * rdt_size,
                                        cudaMemcpyDeviceToDevice, stream),
                       exit_coll, status);
    }
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_bcopy_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);

    ucc_mc_free(task->allgatherv_bcopy.scratch);
    return ucc_tl_nccl_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_nccl_allgatherv_bcopy_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_nccl_team_t *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_nccl_task_t *task;
    size_t              max_count, sdt_size, rdt_size;
    ucc_rank_t          peer;

    CHECK_INPLACE(*args, nccl_team);
    CHECK_USERDEFINED_DT(*args, nccl_team);
    task = ucc_tl_nccl_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MESSAGE;
    }

    sdt_size = ucc_dt_size(args->src.info.datatype);
    rdt_size = ucc_dt_size(args->dst.info_v.datatype);
    max_count = ucc_coll_args_get_count(args, args->dst.info_v.counts, 0);
    for (peer = 1; peer < team->params.size; peer++) {
        max_count = ucc_max(ucc_coll_args_get_count(args,
                            args->dst.info_v.counts, peer), max_count);
    }
    task->allgatherv_bcopy.max_count = max_count;
    if (max_count * rdt_size > args->src.info.count * sdt_size) {
        status = ucc_mc_alloc(&task->allgatherv_bcopy.scratch,
                              (team->params.size + 1) * max_count *
                              ucc_dt_size(args->dst.info_v.datatype),
                              UCC_MEMORY_TYPE_CUDA);

    } else {
        status = ucc_mc_alloc(&task->allgatherv_bcopy.scratch, max_count *
                              team->params.size *
                              ucc_dt_size(args->dst.info_v.datatype),
                              UCC_MEMORY_TYPE_CUDA);
    }
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_nccl_free_task(task);
        return status;
    }
    task->super.post     = ucc_tl_nccl_allgatherv_bcopy_start;
    task->super.finalize = ucc_tl_nccl_allgatherv_bcopy_finalize;
    *task_h = &task->super;
out:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_bcast_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task   = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_coll_args_t    *args   = &TASK_ARGS(task);
    ucc_tl_nccl_team_t *team   = TASK_TEAM(task);
    ucc_rank_t          size   = UCC_TL_TEAM_SIZE(team);
    ucc_ee_h            ee     = coll_task->ee;
    cudaStream_t        stream = (ee) ? (cudaStream_t) ee->ee_context :
                                                       team->stream;
    ucc_status_t        status = UCC_OK;
    void               *sbuf   = args->src.info.buffer;
    ptrdiff_t           rbuf   = (ptrdiff_t)args->dst.info_v.buffer;
    size_t rdt_size, count, displ;
    ucc_rank_t peer;

    task->super.status = UCC_INPROGRESS;
    rdt_size           = ucc_dt_size(args->dst.info_v.datatype);
    UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_allgatherv_start", 0);
    NCCLCHECK_GOTO(ncclGroupStart(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    for (peer = 0; peer < size; peer++) {
        count = ucc_coll_args_get_count(args, args->dst.info_v.counts, peer);
        displ = ucc_coll_args_get_displacement(args,
                                               args->dst.info_v.displacements,
                                               peer);
        NCCLCHECK_GOTO(ncclBroadcast(sbuf, PTR_OFFSET(rbuf, displ * rdt_size),
                                     count * rdt_size, ncclChar, peer,
                                     team->nccl_comm, stream),
                       exit_coll, status, UCC_TL_TEAM_LIB(team));
    }
    NCCLCHECK_GOTO(ncclGroupEnd(), exit_coll, status, UCC_TL_TEAM_LIB(team));
    status = ucc_tl_nccl_collective_sync(task, stream);
exit_coll:
    return status;
}

ucc_status_t ucc_tl_nccl_allgatherv_bcast_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *     team,
                                               ucc_coll_task_t **    task_h)
{
    ucc_tl_nccl_team_t *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_coll_args_t    *args      = &coll_args->args;
    ucc_status_t        status    = UCC_OK;
    ucc_tl_nccl_task_t *task;

    CHECK_INPLACE(*args, nccl_team);
    CHECK_USERDEFINED_DT(*args, nccl_team);
    task = ucc_tl_nccl_init_task(coll_args, team);
    if (!task) {
        return UCC_ERR_NO_MESSAGE;
    }
    task->super.post = ucc_tl_nccl_allgatherv_bcast_start;
    *task_h = &task->super;
out:
    return status;
}
