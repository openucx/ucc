/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"
#include "tl_cuda_coll.h"
#include "tl_cuda_topo.h"
#include "tl_cuda_cache.h"
#include "core/ucc_team.h"
#include "coll_score/ucc_coll_score.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"
#include "utils/ucc_sys.h"
#include <sys/shm.h>

ucc_status_t ucc_tl_cuda_comm_init_post(ucc_tl_cuda_team_t *team)
{
    ucc_base_lib_t        *tl_lib         = UCC_TL_TEAM_LIB(team);
    ucc_tl_cuda_lib_t     *tl_cuda_lib    = ucc_derived_of(tl_lib,
                                                           ucc_tl_cuda_lib_t);
    const ucc_rank_t       tsize          = UCC_TL_TEAM_SIZE(team);
    const ucc_rank_t       trank          = UCC_TL_TEAM_RANK(team);
    const uint32_t         max_concurrent = tl_cuda_lib->cfg.max_concurrent;
    ucc_tl_cuda_rank_id_t *rank_id        = GET_RANK_ID(team->ids, tsize,
                                                        max_concurrent);
    ucc_tl_cuda_sync_t *sync;
    ucc_status_t status;
    CUresult cu_st;
    CUcontext cu_ctx;
    size_t scratch_size, rank_id_size;
    int i;

    rank_id_size = sizeof(ucc_tl_cuda_rank_id_t) +
                   (max_concurrent - 1) * sizeof(cudaIpcEventHandle_t);
    cu_st = cuCtxGetCurrent(&cu_ctx);
    if (cu_ctx == NULL || cu_st != CUDA_SUCCESS) {
        tl_debug(tl_lib,
                 "cannot create CUDA TL team without active CUDA context");
        team->device_id = TL_CUDA_DEVICE_INVALID;
        team->state     = TL_CUDA_STATE_ERROR;
        goto exchnage_rank_ids;
    }

    status = CUDA_FUNC(cudaGetDevice(&team->device));
    if (status != UCC_OK) {
        tl_debug(tl_lib, "failed to get current device id");
        team->device_id = TL_CUDA_DEVICE_INVALID;
        team->state     = TL_CUDA_STATE_ERROR;
        goto exchnage_rank_ids;
    }

    status = ucc_tl_cuda_topo_get_pci_id(team->device, &team->device_id);
    if (status != UCC_OK) {
        tl_error(tl_lib, "failed to get pci id");
        return status;
    }

    status = CUDA_FUNC(cudaStreamCreateWithFlags(&team->stream,
                       cudaStreamNonBlocking));
    if (status != UCC_OK) {
        tl_error(tl_lib, "failed to create CUDA stream");
        return status;
    }

/* create IPC events and get handles */
    for (i = 0; i < tl_cuda_lib->cfg.max_concurrent; i++) {
        sync = UCC_TL_CUDA_TEAM_SYNC(team, trank, i);
        CUDA_CHECK_GOTO(cudaEventCreateWithFlags(&sync->ipc_event_local,
                                                  cudaEventDisableTiming |
                                                  cudaEventInterprocess),
                        free_stream, status);
        CUDA_CHECK_GOTO(cudaIpcGetEventHandle(&rank_id->ev_handle[i],
                                              sync->ipc_event_local),
                        free_stream, status);
    }

/* allocate and map scratch buffer */
    scratch_size = tl_cuda_lib->cfg.max_concurrent *
                   tl_cuda_lib->cfg.scratch_size;
    status = CUDA_FUNC(cudaMalloc(&team->scratch.loc, scratch_size));
    if (status != UCC_OK) {
        tl_error(tl_lib, "failed to alloc scratch buffer");
        goto free_stream;
    }

    status = ucc_tl_cuda_mem_info_get(team->scratch.loc, scratch_size,
                                      &rank_id->scratch_info);
    if (status != UCC_OK) {
        tl_error(tl_lib, "failed to get scratch memory info");
        goto free_scratch;
    }

exchnage_rank_ids:
    rank_id->pci_id = team->device_id;
    status = team->oob.allgather(rank_id, team->ids, rank_id_size,
                                 team->oob.coll_info, &team->oob_req);
    if (UCC_OK != status) {
        tl_error(tl_lib, "failed to start oob allgather");
        goto free_scratch;
    }

    return UCC_OK;

free_scratch:
    cudaFree(team->scratch.loc);
free_stream:
    cudaStreamDestroy(team->stream);
    return status;
}

ucc_status_t ucc_tl_cuda_comm_init_test(ucc_tl_cuda_team_t *team)
{
    ucc_base_lib_t     *tl_lib         = UCC_TL_TEAM_LIB(team);
    ucc_tl_cuda_lib_t  *tl_cuda_lib    = ucc_derived_of(tl_lib, ucc_tl_cuda_lib_t);
    const ucc_rank_t    tsize          = UCC_TL_TEAM_SIZE(team);
    const ucc_rank_t    trank          = UCC_TL_TEAM_RANK(team);
    const uint32_t      max_concurrent = tl_cuda_lib->cfg.max_concurrent;
    ucc_tl_cuda_rank_id_t *rank_id;
    ucc_rank_t r, p;
    ucc_status_t status;
    ucc_tl_cuda_sync_t *sync;

    status = team->oob.req_test(team->oob_req);
    if (status == UCC_INPROGRESS) {
        return UCC_INPROGRESS;
    } else if (status < 0) {
        team->oob.req_free(team->oob_req);
        tl_error(tl_lib, "OOB allgather failed");
        team->state = TL_CUDA_STATE_ERROR;
        return status;
    }
    team->oob.req_free(team->oob_req);
    /* check all ranks have valid CUDA device set */
    for (r = 0; r < tsize; r++) {
        rank_id = GET_RANK_ID(team->ids, r, max_concurrent);
        if (ucc_tl_cuda_topo_device_id_equal(&rank_id->pci_id,
                                             &TL_CUDA_DEVICE_INVALID)) {
            tl_debug(tl_lib, "rank %d device is invalid, team can't be created",
                     r);
            team->state = TL_CUDA_STATE_ERROR;
            return UCC_ERR_NO_RESOURCE;
        }
    }

    status = ucc_tl_cuda_team_topo_create(&team->super, &team->topo);
    if (status != UCC_OK) {
        tl_debug(tl_lib, "failed to craete team topo %d (%s)", status,
                 ucc_status_string(status));
        return status;
    }

    if (UCC_TL_TEAM_LIB(team)->log_component.log_level >= UCC_LOG_LEVEL_DEBUG) {
        ucc_tl_cuda_team_topo_print_proxies(&team->super, team->topo);
        ucc_tl_cuda_team_topo_print_rings(&team->super, team->topo);
    }

    /* map memory handles for remote scratch buffers */
    for (r = 0; r < tsize; r++) {
        if ((r == trank) ||
            !ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo, r,
                                             trank)) {
            team->scratch.rem[r] = NULL;
            continue;
        }
        rank_id = GET_RANK_ID(team->ids, r, max_concurrent);
        status = ucc_tl_cuda_map_memhandle(rank_id->scratch_info.ptr,
                                           rank_id->scratch_info.length,
                                           rank_id->scratch_info.handle,
                                           &team->scratch.rem[r],
                                           ucc_tl_cuda_get_cache(team, r));
        memcpy(&team->scratch.rem_info[r], &rank_id->scratch_info,
               sizeof(ucc_tl_cuda_mem_info_t));
        if (status != UCC_OK) {
            goto exit_err;
        }
    }

    for (r = 0; r < tl_cuda_lib->cfg.max_concurrent; r++) {
        sync = UCC_TL_CUDA_TEAM_SYNC(team, trank, r);
        for (p = 0; p < tsize; p++) {
            if (p == trank) {
                continue;
            }
            rank_id = GET_RANK_ID(team->ids, p, max_concurrent);
            CUDA_CHECK_GOTO(cudaIpcOpenEventHandle(&sync->data[p].ipc_event_remote,
                                                   rank_id->ev_handle[r]),
                            exit_err, status);
        }
    }

    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_comm_init(ucc_tl_cuda_team_t *team)
{
    ucc_status_t status;

    if (team->state == TL_CUDA_STATE_READY) {
        return UCC_OK;
    } else if (team->state == TL_CUDA_STATE_ERROR) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    status = ucc_tl_cuda_comm_init_post(team);
    if (status != UCC_OK) {
        tl_debug(team->super.super.context->lib, "comm init post error %d %s",
                 status, ucc_status_string(status));
        team->state = TL_CUDA_STATE_ERROR;
        return status;
    }

    do {
        /* blocking coll, fix it when we can fallback during collecitve post */
        status = ucc_tl_cuda_comm_init_test(team);
    } while (status == UCC_INPROGRESS);
    if (status != UCC_OK) {
        tl_debug(team->super.super.context->lib, "comm init test error %d %s",
                 status, ucc_status_string(status));
        team->state = TL_CUDA_STATE_ERROR;
        return status;
    }
    team->state = TL_CUDA_STATE_READY;
    return UCC_OK;
}


UCC_CLASS_INIT_FUNC(ucc_tl_cuda_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_cuda_context_t *ctx            = ucc_derived_of(tl_context,
                                                           ucc_tl_cuda_context_t);
    ucc_tl_cuda_lib_t     *lib            = ucc_derived_of(tl_context->lib,
                                                           ucc_tl_cuda_lib_t);
    const ucc_rank_t       tsize          = params->size;
    const ucc_rank_t       trank          = params->rank;
    const uint32_t         max_concurrent = lib->cfg.max_concurrent;
    size_t rank_id_size;
    ucc_tl_cuda_shm_barrier_t *bar;
    ucc_status_t status;
    int shm_id, i, j;
    size_t ctrl_size, alloc_size;
    ucc_tl_cuda_rank_id_t *rank_id;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);
    self->oob    = params->params.oob;
    self->stream = NULL;
    self->topo   = NULL;
    self->device = -1;
    memset(&self->scratch, 0, sizeof(ucc_tl_cuda_scratch_t));

    if (!ucc_team_map_is_single_node(params->team, params->map)) {
        tl_debug(tl_context->lib, "multinode team is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    rank_id_size = sizeof(ucc_tl_cuda_rank_id_t) +
                   (max_concurrent - 1) * sizeof(cudaIpcEventHandle_t);
    self->ids = ucc_malloc((tsize + 1) * rank_id_size, "ids");
    if (!self->ids) {
        tl_error(tl_context->lib, "failed to alloc ranks id");
        return UCC_ERR_NO_MEMORY;
    }

    /* TODO: maybe lesss, check */
    ctrl_size = (sizeof(ucc_tl_cuda_sync_t) + sizeof(ucc_tl_cuda_sync_data_t) *
                (tsize - 1)) * tsize * max_concurrent +
                sizeof(ucc_tl_cuda_shm_barrier_t) * max_concurrent +
                sizeof(ucc_tl_cuda_sync_state_t) * max_concurrent;

    shm_id = -1;
    self->sync = (void*)-1;
    if (trank == 0) {
        alloc_size = ctrl_size;
        status = ucc_sysv_alloc(&alloc_size, (void**)&self->sync, &shm_id);
        if (status != UCC_OK) {
            tl_error(tl_context->lib, "failed to alloc sysv segment");
            /* proceed and notify other ranks about error */
            shm_id = -1;
            goto ids_exchange;
        }
        memset(self->sync, 0, ctrl_size);
        self->bar = (ucc_tl_cuda_shm_barrier_t*)UCC_TL_CUDA_TEAM_SYNC(self, 0,
                                                       max_concurrent);
        for (i = 0; i < max_concurrent; i++) {
            bar = UCC_TL_CUDA_TEAM_BARRIER(self, i);
            for (j = 0; j < tsize; j++) {
                status = ucc_tl_cuda_shm_barrier_init(tsize, j, bar);
                if (status != UCC_OK) {
                    tl_error(tl_context->lib,
                             "failed to initialize shm barrier");
                    ucc_sysv_free(self->sync);
                    shm_id = -1;
                    self->sync = (void*)(-1);
                    /* proceed and notify other ranks about error */
                    goto ids_exchange;
                }
            }
        }
    }
ids_exchange:
    rank_id= GET_RANK_ID(self->ids, tsize, max_concurrent);
    rank_id->shm = shm_id;
    status = self->oob.allgather(rank_id, self->ids, rank_id_size,
                                 self->oob.coll_info, &self->oob_req);
    if (UCC_OK != status) {
        tl_error(tl_context->lib, "failed to start oob allgather");
        goto free_devices;
    }
    tl_debug(tl_context->lib, "posted tl team: %p", self);

    self->state = TL_CUDA_STATE_SHM_ID_EXCHANGE;
    self->seq_num = 1;
    return UCC_OK;

free_devices:
    if (shm_id != -1) {
        ucc_sysv_free(self->sync);
        self->sync = (void*)(-1);
    }
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_team_t)
{
    ucc_tl_cuda_lib_t *lib = ucc_derived_of(self->super.super.context->lib,
                                            ucc_tl_cuda_lib_t);
    ucc_tl_cuda_sync_t *sync;
    cudaError_t st;
    int i, j;

    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);
    if (self->topo) {
        ucc_tl_cuda_team_topo_destroy(self->topo);
    }
    if (self->ids) {
        if (self->sync != (void*)-1) {
            for (i = 0; i < lib->cfg.max_concurrent; i++) {
                for (j = 0; j < UCC_TL_TEAM_SIZE(self); j++) {
                    if (j == UCC_TL_TEAM_RANK(self)) {
                        continue;
                    }
                    sync = UCC_TL_CUDA_TEAM_SYNC(self, j, i);
                    if (sync->data[j].ipc_event_remote) {
                        st = cudaEventDestroy(sync->data[j].ipc_event_remote);
                        if (st != cudaSuccess) {
                            tl_warn(UCC_TL_TEAM_LIB(self), "cudaEventDestroy "
                                    "failed: %d (%s)", st, cudaGetErrorName(st));
                        }
                    }
                }
                sync = UCC_TL_CUDA_TEAM_SYNC(self, UCC_TL_TEAM_RANK(self), i);
                if (sync->ipc_event_local) {
                    st = cudaEventDestroy(sync->ipc_event_local);
                    if (st != cudaSuccess) {
                        tl_warn(UCC_TL_TEAM_LIB(self), "cudaEventDestroy "
                                "failed: %d (%s)", st, cudaGetErrorName(st));
                    }
                }
            }
            ucc_sysv_free(self->sync);
        }
        ucc_free(self->ids);
    }
    if (self->stream) {
        st = cudaStreamDestroy(self->stream);
        if (st != cudaSuccess) {
            tl_warn(UCC_TL_TEAM_LIB(self), "cudaStreamDestroy failed: %d (%s)",
                    st, cudaGetErrorName(st));
        }
    }
    for (i = 0; i < UCC_TL_TEAM_SIZE(self); i++) {
        if (self->scratch.rem[i]) {
            ucc_tl_cuda_unmap_memhandle((uintptr_t)self->scratch.rem_info[i].ptr,
                                        self->scratch.rem[i],
                                        ucc_tl_cuda_get_cache(self, i), 1);
        }
    }

    if (self->scratch.loc) {
        cudaFree(self->scratch.loc);
    }
}

UCC_CLASS_DEFINE_DELETE_FUNC(ucc_tl_cuda_team_t, ucc_base_team_t);

UCC_CLASS_DEFINE(ucc_tl_cuda_team_t, ucc_tl_team_t);

ucc_status_t ucc_tl_cuda_team_destroy(ucc_base_team_t *tl_team)
{
    UCC_CLASS_DELETE_FUNC_NAME(ucc_tl_cuda_team_t)(tl_team);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_team_create_test(ucc_base_team_t *tl_team)
{
    ucc_tl_cuda_team_t    *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_context_t *ctx  = ucc_derived_of(UCC_TL_TEAM_CTX(team),
                                                 ucc_tl_cuda_context_t);
    ucc_tl_cuda_lib_t     *lib  = ucc_derived_of(tl_team->context->lib,
                                                 ucc_tl_cuda_lib_t);
    const ucc_rank_t       trank = UCC_TL_TEAM_RANK(team);
    ucc_status_t status;
    int shm_id;

    if (team->state == TL_CUDA_STATE_READY) {
        return UCC_OK;
    } else if (team->state == TL_CUDA_STATE_ERROR) {
        return UCC_ERR_NO_MESSAGE;
    }

    if (team->state == TL_CUDA_STATE_SHM_ID_EXCHANGE) {
        status = team->oob.req_test(team->oob_req);
        if (status == UCC_INPROGRESS) {
            return UCC_INPROGRESS;
        } else if (status < 0) {
            tl_error(tl_team->context->lib, "OOB allgather failed");
            team->oob.req_free(team->oob_req);
            team->state = TL_CUDA_STATE_ERROR;
            return status;
        }
        team->oob.req_free(team->oob_req);
        shm_id = GET_RANK_ID(team->ids, 0, lib->cfg.max_concurrent)->shm;
        if (shm_id < 0) {
            tl_error(tl_team->context->lib, "failed to create shmem region");
            team->state = TL_CUDA_STATE_ERROR;
            status = UCC_ERR_NO_MEMORY;
            return status;
        }
        if (trank != 0) {
            team->sync = shmat(shm_id, NULL, 0);
            if (team->sync == (void *)-1) {
                tl_error(tl_team->context->lib, "failed to shmat errno: %d (%s)",
                        errno, strerror(errno));
                team->state = TL_CUDA_STATE_ERROR;
                status = UCC_ERR_NO_MEMORY;
                return status;
            }
            team->bar = (ucc_tl_cuda_shm_barrier_t*)
                UCC_TL_CUDA_TEAM_SYNC(team, 0, lib->cfg.max_concurrent);
        }
        team->sync_state = (ucc_tl_cuda_sync_state_t*)
            PTR_OFFSET(team->bar, sizeof(ucc_tl_cuda_shm_barrier_t) *
                        lib->cfg.max_concurrent);
        team->state = TL_CUDA_STATE_COMM_INIT;
        if (!ctx->cfg.lazy_init) {
            status = ucc_tl_cuda_comm_init_post(team);
            if (status != UCC_OK) {
                team->state = TL_CUDA_STATE_ERROR;
                return status;
            }
        } else {
            return UCC_OK;
        }

    }
    if (team->state == TL_CUDA_STATE_COMM_INIT) {
        if (ctx->cfg.lazy_init) {
            return UCC_OK;
        }
        status = ucc_tl_cuda_comm_init_test(team);
        if (status == UCC_INPROGRESS) {
            return status;
        } else if (status < 0) {
            team->state = TL_CUDA_STATE_ERROR;
            return status;
        } else {
            team->state = TL_CUDA_STATE_ERROR;
        }
    }

    tl_debug(tl_team->context->lib, "initialized tl team: %p", team);
    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_team_get_scores(ucc_base_team_t *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_base_context_t *ctx  = UCC_TL_TEAM_CTX(team);
    ucc_memory_type_t   mt   = UCC_MEMORY_TYPE_CUDA;
    ucc_coll_score_t   *score;
    ucc_status_t        status;
    int                 i;

    status =
        ucc_coll_score_build_default(tl_team, UCC_TL_CUDA_DEFAULT_SCORE,
                                     ucc_tl_cuda_coll_init,
                                     UCC_TL_CUDA_SUPPORTED_COLLS,
                                     &mt, 1, &score);
    if (UCC_OK != status) {
        return status;
    }

    for (i = 0; i < UCC_TL_CUDA_N_DEFAULT_ALG_SELECT_STR; i++) {
        status = ucc_coll_score_update_from_str(
            ucc_tl_cuda_default_alg_select_str[i], score,
            UCC_TL_TEAM_SIZE(team), ucc_tl_cuda_coll_init, &team->super.super,
            UCC_TL_CUDA_DEFAULT_SCORE, ucc_tl_cuda_alg_id_to_init, &mt, 1);
        if (UCC_OK != status) {
            tl_error(tl_team->context->lib,
                     "failed to apply default coll select setting: %s",
                     ucc_tl_cuda_default_alg_select_str[i]);
            goto err;
        }
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_TL_TEAM_SIZE(team),
            ucc_tl_cuda_coll_init, &team->super.super,
            UCC_TL_CUDA_DEFAULT_SCORE, ucc_tl_cuda_alg_id_to_init, &mt, 1);
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
