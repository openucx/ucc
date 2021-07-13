/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "tl_nccl_coll.h"
#include "core/ucc_mc.h"
#include "core/ucc_ee.h"
#include "coll_score/ucc_coll_score.h"

UCC_CLASS_INIT_FUNC(ucc_tl_nccl_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_nccl_context_t *ctx    =
        ucc_derived_of(tl_context, ucc_tl_nccl_context_t);
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params->team);
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
    if (task->completed) {
        ucc_mc_ee_destroy_event(task->completed, UCC_EE_CUDA_STREAM);
    }
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_tl_nccl_triggered_post(ucc_ee_h ee, ucc_ev_t *ev, ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task  = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t status;
    ucc_ev_t *post_event;

    ucc_assert(ee->ee_type == UCC_EE_CUDA_STREAM);
    coll_task->ee = ee;
    tl_info(task->team->super.super.context->lib, "triggered post. task:%p", coll_task);

    status = coll_task->post(coll_task);
    if (ucc_likely(status == UCC_OK)) {
        /* TODO: mpool */
        post_event = ucc_malloc(sizeof(ucc_ev_t), "event");
        if (ucc_unlikely(post_event == NULL)) {
            tl_error(task->team->super.super.context->lib,
                    "failed to allocate memory for event");
            return UCC_ERR_NO_MEMORY;
        }

        post_event->ev_type = UCC_EVENT_COLLECTIVE_POST;
        post_event->ev_context_size = 0;
        post_event->req = &coll_task->super;
        ucc_ee_set_event_internal(coll_task->ee, post_event,
                                  &coll_task->ee->event_out_queue);
    }
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
    task->team                 = nccl_team;
    task->super.finalize       = ucc_tl_nccl_coll_finalize;
    task->super.triggered_post = ucc_tl_nccl_triggered_post;
    task->completed            = NULL;
    if (nccl_ctx->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT) {
        status = ucc_mc_ee_create_event((void **)&task->completed,
                                         UCC_EE_CUDA_STREAM);
        if (ucc_unlikely(status != UCC_OK)) {
            goto free_task;
        }
    }
    switch (coll_args->args.coll_type)
    {
    case UCC_COLL_TYPE_ALLGATHER:
        status = ucc_tl_nccl_allgather_init(task);
        break;
    case UCC_COLL_TYPE_ALLGATHERV:
        status = ucc_tl_nccl_allgatherv_init(task);
        break;
    case UCC_COLL_TYPE_ALLREDUCE:
        status = ucc_tl_nccl_allreduce_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALL:
        status = ucc_tl_nccl_alltoall_init(task);
        break;
    case UCC_COLL_TYPE_ALLTOALLV:
        status = ucc_tl_nccl_alltoallv_init(task);
        break;
    case UCC_COLL_TYPE_BCAST:
        status = ucc_tl_nccl_bcast_init(task);
        break;
    default:
        tl_error(UCC_TL_TEAM_LIB(task->team),
                 "collective %d is not supported by nccl tl",
                 coll_args->args.coll_type);
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (ucc_unlikely(status != UCC_OK)) {
        goto free_event;
    }
    tl_info(UCC_TL_TEAM_LIB(task->team), "init coll task %p", task);
    *task_h = &task->super;
    return status;

free_event:
    if (task->completed) {
        cudaEventDestroy(task->completed);
    }
free_task:
    ucc_mpool_put(task);
    return status;
}

ucc_status_t ucc_tl_nccl_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_nccl_team_t *team = ucc_derived_of(tl_team, ucc_tl_nccl_team_t);
    ucc_tl_nccl_lib_t * lib  = UCC_TL_NCCL_TEAM_LIB(team);
    ucc_memory_type_t   mt   = UCC_MEMORY_TYPE_CUDA;
    ucc_coll_score_t   *score;
    ucc_status_t        status;

    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status =
        ucc_coll_score_build_default(tl_team, UCC_TL_NCCL_DEFAULT_SCORE,
                           ucc_tl_nccl_coll_init, UCC_TL_NCCL_SUPPORTED_COLLS,
                           &mt, 1, &score);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    if (strlen(lib->super.super.score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            lib->super.super.score_str, score, team->size,
            ucc_tl_nccl_coll_init, &team->super.super,
            UCC_TL_NCCL_DEFAULT_SCORE, NULL);
        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }
    *score_p = score;
    return UCC_OK;
err:
    ucc_coll_score_free(score);
    return status;
}
