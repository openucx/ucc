/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "tl_nccl_coll.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_ee.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/arch/cuda_def.h"

UCC_CLASS_INIT_FUNC(ucc_tl_nccl_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_nccl_context_t *ctx = ucc_derived_of(tl_context,
                                                ucc_tl_nccl_context_t);
    ucc_team_oob_coll_t *oob;
    ucc_status_t status;
    ucc_rank_t size;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);
    oob = &(UCC_TL_TEAM_OOB(self));
    size = UCC_TL_TEAM_SIZE(self);
    self->stream     = NULL;
    self->nccl_comm  = NULL;
    self->unique_id  = ucc_malloc(sizeof(ncclUniqueId) * (size + 1),
                                  "tl_nccl_unique_id");
    if (!self->unique_id) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate %zd bytes for unique_id array",
                 sizeof(ncclUniqueId) * (size + 1));
        return UCC_ERR_NO_MEMORY;
    }

    if (UCC_TL_TEAM_RANK(self) == 0) {
        ncclResult_t st;
        st = ncclGetUniqueId(&self->unique_id[size]);
        if (st != ncclSuccess) {
            tl_error(ctx->super.super.lib, "failed to get unique id");
            memset(&self->unique_id[size], 0, sizeof(ncclUniqueId));
        }
    }

    status = oob->allgather(&self->unique_id[size],
                            self->unique_id, sizeof(ncclUniqueId),
                            oob->coll_info, &self->oob_req);
    if (status != UCC_OK) {
        tl_error(ctx->super.super.lib, "failed to start oob allgather");
        goto free_unique_id;
    }
    self->comm_state = TL_NCCL_COMM_STATE_OOB;

    return UCC_OK;

free_unique_id:
    ucc_free(self->unique_id);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_nccl_team_t)
{
    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_nccl_team_t, ucc_base_team_t);
UCC_CLASS_DEFINE(ucc_tl_nccl_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_nccl_team_destroy(ucc_base_team_t *tl_team)
{
    ucc_tl_nccl_team_t *team = ucc_derived_of(tl_team, ucc_tl_nccl_team_t);

#if NCCL_USE_NON_BLOCKING
    ncclResult_t nccl_status, st;

    if (team->comm_state == TL_NCCL_COMM_STATE_DESTROY_COMM) {
        goto check_finalize;
    }
#endif

    if (team->stream) {
        cudaStreamDestroy(team->stream);
        team->stream = NULL;
    }
    if (team->nccl_comm) {
        if (team->comm_state == TL_NCCL_COMM_STATE_ERROR) {
            ncclCommAbort(team->nccl_comm);
        } else {
#if NCCL_USE_NON_BLOCKING
            ncclCommFinalize(team->nccl_comm);
check_finalize:
            st = ncclCommGetAsyncError(team->nccl_comm, &nccl_status);
            if (st != ncclSuccess || (nccl_status != ncclSuccess)) {
                tl_debug(tl_team->context->lib, "NCCL error %d %s",
                         st != ncclSuccess ? st : nccl_status,
                ncclGetErrorString(st != ncclSuccess ? st : nccl_status));
                ncclCommAbort(team->nccl_comm);
                return UCC_ERR_NO_MESSAGE;
            } else if (nccl_status == ncclInProgress) {
                team->comm_state = TL_NCCL_COMM_STATE_DESTROY_COMM;
                return UCC_INPROGRESS;
            } else {
                ncclCommDestroy(team->nccl_comm);
            }
            team->comm_state = UCC_OK;
#else
            ncclCommDestroy(team->nccl_comm);
#endif
        }
    }

    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_nccl_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_comm_init(ucc_tl_nccl_team_t *team)
{
    ucc_rank_t   tsize    = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t   trank    = UCC_TL_TEAM_RANK(team);
    ucc_status_t status;
    ncclResult_t nccl_status;
#if NCCL_USE_NON_BLOCKING
    ncclConfig_t nccl_cfg = NCCL_CONFIG_INITIALIZER;
    ncclResult_t async_status;
#endif

    if (team->comm_state == TL_NCCL_COMM_STATE_READY) {
        return UCC_OK;
    } else if (team->comm_state == TL_NCCL_COMM_STATE_ERROR) {
        return UCC_ERR_NOT_SUPPORTED;
    } else if (team->comm_state == TL_NCCL_COMM_STATE_INIT_COMM) {
#if NCCL_USE_NON_BLOCKING
        goto nccl_async_init;
#else
        ucc_assert_always(0);
#endif
    }

    CUDA_CHECK_GOTO(cudaStreamCreateWithFlags(&team->stream,
                                              cudaStreamNonBlocking),
                    exit_err, status);
#if NCCL_USE_NON_BLOCKING
    /*
    * if NCCL comm initialized during first call to collective init a.k.a lazy init
    * we need to use blocking init to correctly fallback to other TL in case of error
    */
    nccl_cfg.blocking = (UCC_TL_NCCL_TEAM_CTX(team)->cfg.nccl_cfg_blocking ||
                         UCC_TL_NCCL_TEAM_CTX(team)->cfg.nccl_lazy_init) ? 1: 0;

    nccl_status = ncclCommInitRankConfig(&team->nccl_comm, tsize,
                                         team->unique_id[0], trank, &nccl_cfg);
    if ((nccl_status != ncclInProgress) && (nccl_status != ncclSuccess)) {
        goto nccl_comm_init_err;
    }
nccl_async_init:
    nccl_status = ncclCommGetAsyncError(team->nccl_comm, &async_status);
    if (nccl_status != ncclSuccess) {
        goto nccl_comm_init_err;
    }
    if (async_status == ncclInProgress) {
        team->comm_state = TL_NCCL_COMM_STATE_INIT_COMM;
    }
#else
    nccl_status = ncclCommInitRank(&team->nccl_comm, tsize, team->unique_id[0],
                                   trank);
    if (nccl_status != ncclSuccess) {
        goto nccl_comm_init_err;
    }
#endif

    team->comm_state = TL_NCCL_COMM_STATE_READY;
    return UCC_OK;

nccl_comm_init_err:
    tl_debug(team->super.super.context->lib, "NCCL error %d %s",
             nccl_status, ncclGetErrorString(nccl_status));
    if (nccl_status == ncclInvalidUsage) {
        /*
        * handles the case when trying to inititize multiple ranks
        * on the same GPU. Return "not supported" and fallback to other TL
        */
        status = UCC_ERR_NOT_SUPPORTED;
    } else {
        status = UCC_ERR_NO_RESOURCE;
    }
    team->comm_state = TL_NCCL_COMM_STATE_ERROR;

exit_err:
    return status;
}

ucc_status_t ucc_tl_nccl_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_nccl_team_t  *team = ucc_derived_of(tl_team, ucc_tl_nccl_team_t);
    ucc_team_oob_coll_t *oob  = &(UCC_TL_TEAM_OOB(team));
    ncclUniqueId errorid;
    ucc_status_t status;


    if (team->comm_state == TL_NCCL_COMM_STATE_OOB) {
        status = oob->req_test(team->oob_req);
        if (status == UCC_INPROGRESS) {
            return UCC_INPROGRESS;
        }

        oob->req_free(team->oob_req);
        if (status != UCC_OK) {
            tl_error(tl_team->context->lib, "oob req test failed");
            return status;
        }

        /* check unique id is valid */
        memset(&errorid, 0, sizeof(errorid));
        if (!memcmp(&errorid, team->unique_id, sizeof(errorid))) {
            tl_error(tl_team->context->lib, "incorrect unique id");
            return status;
        }

        team->comm_state = TL_NCCL_COMM_STATE_INIT_TEAM;
    }

    if (UCC_TL_NCCL_TEAM_CTX(team)->cfg.nccl_lazy_init) {
        return UCC_OK;
    }

    return ucc_tl_nccl_comm_init(team);
}

ucc_status_t ucc_tl_nccl_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *team,
                                   ucc_coll_task_t **task_h)
{
    ucc_tl_nccl_task_t *task;
    ucc_status_t        status;

    status = ucc_tl_nccl_init_task(coll_args, team, &task);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
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
    case UCC_COLL_TYPE_REDUCE_SCATTER:
        status = ucc_tl_nccl_reduce_scatter_init(task);
        break;
    case UCC_COLL_TYPE_REDUCE:
        status = ucc_tl_nccl_reduce_init(task);
        break;
    case UCC_COLL_TYPE_BARRIER:
        status = ucc_tl_nccl_barrier_init(task);
        break;
    case UCC_COLL_TYPE_GATHER:
        status = ucc_tl_nccl_gather_init(task);
        break;
    case UCC_COLL_TYPE_GATHERV:
        status = ucc_tl_nccl_gatherv_init(task);
        break;
    case UCC_COLL_TYPE_SCATTER:
        status = ucc_tl_nccl_scatter_init(task);
        break;
    case UCC_COLL_TYPE_SCATTERV:
        status = ucc_tl_nccl_scatterv_init(task);
        break;
    default:
        tl_debug(UCC_TASK_LIB(task),
                 "collective %s is not supported by nccl tl",
                 ucc_coll_type_str(coll_args->args.coll_type));
        status = UCC_ERR_NOT_SUPPORTED;
    }
    if (ucc_unlikely(status != UCC_OK)) {
        goto free_task;
    }
    tl_debug(UCC_TASK_LIB(task), "init coll task %p", task);
    *task_h = &task->super;
    return status;

free_task:
    ucc_tl_nccl_free_task(task);
    return status;
}

ucc_status_t ucc_tl_nccl_team_get_scores(ucc_base_team_t   *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_nccl_team_t *team   = ucc_derived_of(tl_team, ucc_tl_nccl_team_t);
    ucc_base_context_t *ctx    = UCC_TL_TEAM_CTX(team);
    ucc_memory_type_t   mts[2] = {UCC_MEMORY_TYPE_CUDA,
                                  UCC_MEMORY_TYPE_CUDA_MANAGED};
    ucc_coll_score_t   *score;
    ucc_status_t        status;
    int                 i;
    ucc_coll_score_team_info_t team_info;

    team_info.alg_fn              = ucc_tl_nccl_alg_id_to_init;
    team_info.default_score       = UCC_TL_NCCL_DEFAULT_SCORE;
    team_info.init                = ucc_tl_nccl_coll_init;
    team_info.num_mem_types       = 2;
    team_info.supported_mem_types = mts;
    team_info.supported_colls     = UCC_TL_NCCL_SUPPORTED_COLLS;
    team_info.size                = UCC_TL_TEAM_SIZE(team);
    /* There can be a different logic for different coll_type/mem_type.
       Right now just init everything the same way. */
    status =
        ucc_coll_score_build_default(tl_team, UCC_TL_NCCL_DEFAULT_SCORE,
                           ucc_tl_nccl_coll_init, UCC_TL_NCCL_SUPPORTED_COLLS,
                           mts, 2, &score);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    for (i = 0; i < UCC_TL_NCCL_N_DEFAULT_ALG_SELECT_STR; i++) {
        status = ucc_coll_score_update_from_str(
            ucc_tl_nccl_default_alg_select_str[i], &team_info,
            &team->super.super, score);
        if (ucc_unlikely(UCC_OK != status)) {
            tl_error(tl_team->context->lib,
                     "failed to apply default coll select setting: %s",
                     ucc_tl_nccl_default_alg_select_str[i]);
            goto err;
        }
    }

    // add barrier, which might be triggered from host memory type
    // use lower score
    status = ucc_coll_score_add_range(score, UCC_COLL_TYPE_BARRIER,
                                      UCC_MEMORY_TYPE_HOST, 0, UCC_MSG_MAX, 1,
                                      ucc_tl_nccl_coll_init, tl_team);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(ctx->score_str, &team_info,
                                                &team->super.super, score);
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
