/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "tl_nccl_coll.h"

UCC_CLASS_INIT_FUNC(ucc_tl_nccl_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_nccl_context_t *ctx    =
        ucc_derived_of(tl_context, ucc_tl_nccl_context_t);
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super);
    self->oob       = params->params.oob;
    self->size      = self->oob.participants;
    self->rank      = params->rank;
    self->unique_id = ucc_malloc(sizeof(ncclUniqueId) * (self->size + 1),
                                 "tl_nccl_unique_id");
    if (!self->unique_id) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for unique_id array",
                 sizeof(ncclUniqueId) * (self->size + 1));
        return UCC_ERR_NO_MEMORY;
    }
    if (self->rank == 0) {
        ncclResult_t st;
        st = ncclGetUniqueId(&self->unique_id[self->size]);
        if (st != ncclSuccess) {
            tl_error(ctx->super.super.lib, "failed to get unique id");
            memset(&self->unique_id[self->size], 0, sizeof(ncclUniqueId));
        }
    }
    status = self->oob.allgather(&self->unique_id[self->size], self->unique_id,
                                 sizeof(ncclUniqueId), self->oob.coll_info,
                                 &self->oob_req);
    if (status != UCC_OK) {
        tl_error(ctx->super.super.lib, "failed to start oob allgather");
        goto free_unique_id;
    }
    return UCC_OK;

free_unique_id:
    ucc_free(self->unique_id);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_nccl_team_t)
{
    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
    if (self->nccl_comm) {
        ncclCommDestroy(self->nccl_comm);
        cudaStreamDestroy(self->stream);
    }
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_nccl_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_nccl_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_nccl_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_nccl_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_nccl_team_t *team = ucc_derived_of(tl_team, ucc_tl_nccl_team_t);
    ucc_status_t status;
    ncclResult_t nccl_status;
    ncclUniqueId errorid;
    status = team->oob.req_test(team->oob_req);
    if (status == UCC_INPROGRESS) {
        return UCC_INPROGRESS;
    }
    if (status != UCC_OK) {
        team->oob.req_free(team->oob_req);
        tl_error(tl_team->context->lib, "oob req test failed");
        goto free_unique_id;
    }
    status = team->oob.req_free(team->oob_req);
    if (status != UCC_OK) {
        tl_error(tl_team->context->lib, "oob req free failed");
        goto free_unique_id;
    }
    /* check unique id is valid */
    memset(&errorid, 0, sizeof(errorid));
    if (!memcmp(&errorid, team->unique_id, sizeof(errorid))) {
        tl_error(tl_team->context->lib, "incorrect unique id");
        goto free_unique_id;
    }

    CUDACHECK_GOTO(cudaStreamCreateWithFlags(&team->stream,
                   cudaStreamNonBlocking), free_unique_id, status,
                   tl_team->context->lib);
    nccl_status = ncclCommInitRank(&team->nccl_comm,team->size,
                                   team->unique_id[0], team->rank);
    if (nccl_status != ncclSuccess) {
        tl_info(tl_team->context->lib, "NCCL error %d %s",
                nccl_status, ncclGetErrorString(nccl_status));
        status = UCC_ERR_NO_MESSAGE;
        goto free_stream;
    }
    ucc_free(team->unique_id);
    tl_info(tl_team->context->lib, "initialized tl team: %p", team);
    return UCC_OK;

free_stream:
    cudaStreamDestroy(team->stream);
free_unique_id:
    ucc_free(team->unique_id);
    return status;
}

static ucc_status_t ucc_tl_nccl_coll_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task  = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t       status = UCC_OK ;

    tl_info(UCC_TL_TEAM_LIB(task->team), "finalizing coll task %p", task);
    CUDACHECK_GOTO(cudaEventDestroy(task->completed), put_task, status,
                   UCC_TL_TEAM_LIB(task->team));
put_task:
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_tl_nccl_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task_h)
{
    ucc_tl_nccl_team_t    *nccl_team = ucc_derived_of(team, ucc_tl_nccl_team_t);
    ucc_tl_nccl_context_t *nccl_ctx  = ucc_derived_of(team->context,
                                                      ucc_tl_nccl_context_t);
    ucc_tl_nccl_task_t    *task;
    ucc_status_t status;

    task = ucc_mpool_get(&nccl_ctx->req_mp);
    ucc_coll_task_init(&task->super);
    memcpy(&task->args, &coll_args->args, sizeof(ucc_coll_args_t));
    task->team = nccl_team;
    task->super.finalize = ucc_tl_nccl_coll_finalize;
    CUDACHECK_GOTO(cudaEventCreateWithFlags(&task->completed,
                   cudaEventDisableTiming), free_task, status,
                   team->context->lib);
    switch (coll_args->args.coll_type)
    {
    case UCC_COLL_TYPE_ALLTOALL:
        status = ucc_tl_nccl_alltoall_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        status = ucc_tl_nccl_alltoallv_init(task);
        break;
    default:
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "collective %d is not supported by nccl tl",
                 coll_args->args.coll_type);
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (status != UCC_OK) {
        goto free_event;
    }
    *task_h = &task->super;
    return status;

free_event:
    cudaEventDestroy(task->completed);
free_task:
    ucc_mpool_put(task);
    return status;
}
