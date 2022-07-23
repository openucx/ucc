/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

UCC_CLASS_INIT_FUNC(ucc_tl_cuda_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_cuda_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_cuda_context_t);
    ucc_tl_cuda_lib_t     *lib =
        ucc_derived_of(tl_context->lib, ucc_tl_cuda_lib_t);
    ucc_tl_cuda_shm_barrier_t *bar;
    ucc_status_t status;
    int shm_id, i, j;
    size_t ctrl_size, alloc_size, scratch_size;
    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    self->oob         = params->params.oob;
    self->stream      = NULL;
    self->topo        = NULL;
    self->scratch.loc = NULL;
    if (UCC_TL_TEAM_SIZE(self) < 2) {
        tl_trace(tl_context->lib, "team size is too small, min supported 2");
        return UCC_ERR_NOT_SUPPORTED;
    }
    if (UCC_TL_TEAM_SIZE(self) > UCC_TL_CUDA_MAX_PEERS) {
        tl_info(tl_context->lib, "team size is too large, max supported %d",
                UCC_TL_CUDA_MAX_PEERS);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!ucc_team_map_is_single_node(params->team, params->map)) {
        tl_info(tl_context->lib, "multinode team is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    self->ids = ucc_malloc((UCC_TL_TEAM_SIZE(self) + 1) * sizeof(*(self->ids)),
                            "ids");
    if (!self->ids) {
        tl_error(tl_context->lib, "failed to alloc ranks id");
        return UCC_ERR_NO_MEMORY;
    }

    scratch_size = lib->cfg.max_concurrent * lib->cfg.scratch_size;
    status = CUDA_FUNC(cudaMalloc(&self->scratch.loc, scratch_size));
    if (status != UCC_OK) {
        tl_error(tl_context->lib, "failed to alloc scratch buffer");
        goto free_ids;
    }

    status = ucc_tl_cuda_mem_info_get(self->scratch.loc, scratch_size,
                            &self->ids[UCC_TL_TEAM_SIZE(self)].scratch_info);
    if (status != UCC_OK) {
        tl_error(tl_context->lib, "failed to get scratch memory info");
        goto free_scratch;
    }

    ctrl_size = (sizeof(ucc_tl_cuda_sync_t) + sizeof(ucc_tl_cuda_sync_data_t) *
                (UCC_TL_TEAM_SIZE(self) - 1)) * UCC_TL_TEAM_SIZE(self) *
                lib->cfg.max_concurrent +
                sizeof(ucc_tl_cuda_shm_barrier_t) * lib->cfg.max_concurrent +
                sizeof(ucc_tl_cuda_sync_state_t) * lib->cfg.max_concurrent;

    shm_id = -1;
    self->sync = (void*)-1;
    if (UCC_TL_TEAM_RANK(self) == 0) {
        alloc_size = ctrl_size;
        status = ucc_sysv_alloc(&alloc_size, (void**)&self->sync, &shm_id);
        if (status != UCC_OK) {
            tl_error(tl_context->lib, "failed to alloc sysv segment");
            /* proceed and notify other ranks about error */
            shm_id = -1;
            goto ids_exchange;
        }
        memset(self->sync, 0, ctrl_size);
        self->bar  = (ucc_tl_cuda_shm_barrier_t*)UCC_TL_CUDA_TEAM_SYNC(self, 0,
                                                       lib->cfg.max_concurrent);
        for (i = 0; i < lib->cfg.max_concurrent; i++) {
            bar = UCC_TL_CUDA_TEAM_BARRIER(self, i);
            for (j = 0; j < UCC_TL_TEAM_SIZE(self); j++) {
                status = ucc_tl_cuda_shm_barrier_init(UCC_TL_TEAM_SIZE(self),
                                                      j, bar);
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
    self->ids[UCC_TL_TEAM_SIZE(self)].device = ctx->device;
    self->ids[UCC_TL_TEAM_SIZE(self)].pci_id = ctx->device_id;
    self->ids[UCC_TL_TEAM_SIZE(self)].shm    = shm_id;
    status = self->oob.allgather(&self->ids[UCC_TL_TEAM_SIZE(self)], self->ids,
                                 sizeof(ucc_tl_cuda_rank_id_t),
                                 self->oob.coll_info, &self->oob_req);
    if (UCC_OK != status) {
        tl_error(tl_context->lib, "failed to start oob allgather");
        goto free_devices;
    }
    tl_info(tl_context->lib, "posted tl team: %p", self);

    self->seq_num = 1;
    return UCC_OK;

free_devices:
    if (shm_id != -1) {
        ucc_sysv_free(self->sync);
        self->sync = (void*)(-1);
    }
free_scratch:
    cudaFree(self->scratch.loc);
free_ids:
    ucc_free(self->ids);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_team_t)
{
    ucc_tl_cuda_lib_t *lib = ucc_derived_of(self->super.super.context->lib,
                                            ucc_tl_cuda_lib_t);
    ucc_tl_cuda_sync_t *sync;
    cudaError_t st;
    int i, j;

    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
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
                                        ucc_tl_cuda_get_cache(self, i));
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
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_lib_t  *lib  = ucc_derived_of(tl_team->context->lib,
                                              ucc_tl_cuda_lib_t);
    ucc_status_t status;
    ucc_tl_cuda_sync_t *sync;
    ucc_tl_cuda_shm_barrier_t *bar;
    volatile ucc_tl_cuda_sync_t *peer_sync;
    int i, j, shm_id;

    if (team->oob_req == NULL) {
        return UCC_OK;
    } else if (team->oob_req == (void*)0x1) {
        goto barrier;
    }
    status = team->oob.req_test(team->oob_req);
    if (status == UCC_INPROGRESS) {
        return UCC_INPROGRESS;
    } else if (status < 0) {
        tl_error(tl_team->context->lib, "oob allgather failed");
        goto exit_err;
    }
    team->oob.req_free(team->oob_req);
    team->oob_req = (void*)0x1;
    status = ucc_tl_cuda_team_topo_create(&team->super, &team->topo);
    if (status != UCC_OK) {
        goto exit_err;
    }

    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
        if (i == UCC_TL_TEAM_RANK(team) ||
            !ucc_tl_cuda_team_topo_is_direct(&team->super, team->topo, i,
                                             UCC_TL_TEAM_RANK(team))) {
            team->scratch.rem[i] = NULL;
            continue;
        }
        status = ucc_tl_cuda_map_memhandle(team->ids[i].scratch_info.ptr,
                                           team->ids[i].scratch_info.length,
                                           team->ids[i].scratch_info.handle,
                                           &team->scratch.rem[i],
                                           ucc_tl_cuda_get_cache(team, i));
        memcpy(&team->scratch.rem_info[i], &team->ids[i].scratch_info,
               sizeof(ucc_tl_cuda_mem_info_t));
        if (status != UCC_OK) {
            goto exit_err;
        }
    }

    if (UCC_TL_TEAM_LIB(team)->log_component.log_level >= UCC_LOG_LEVEL_DEBUG) {
        ucc_tl_cuda_team_topo_print(&team->super, team->topo);
        ucc_tl_cuda_team_topo_print_rings(&team->super, team->topo);
    }

    shm_id = team->ids[0].shm;
    if (shm_id < 0) {
        tl_error(tl_team->context->lib, "failed to create shmem region");
        status = UCC_ERR_NO_MEMORY;
        goto exit_err;
    }
    if (UCC_TL_TEAM_RANK(team) != 0) {
        team->sync = shmat(shm_id, NULL, 0);
        if (team->sync == (void *)-1) {
            tl_error(tl_team->context->lib, "failed to shmat errno: %d (%s)",
                     errno, strerror(errno));
            status = UCC_ERR_NO_MEMORY;
            goto exit_err;
        }
        team->bar = (ucc_tl_cuda_shm_barrier_t*)UCC_TL_CUDA_TEAM_SYNC(team, 0,
                                                       lib->cfg.max_concurrent);
    }
    team->sync_state = (ucc_tl_cuda_sync_state_t*)PTR_OFFSET(team->bar,
                            sizeof(ucc_tl_cuda_shm_barrier_t) *
                            lib->cfg.max_concurrent);
    CUDA_CHECK_GOTO(cudaStreamCreateWithFlags(&team->stream,
                    cudaStreamNonBlocking), exit_err, status);
    for (i = 0; i < lib->cfg.max_concurrent; i++) {
        sync = UCC_TL_CUDA_TEAM_SYNC(team, UCC_TL_TEAM_RANK(team), i);
        CUDA_CHECK_GOTO(cudaEventCreateWithFlags(&sync->ipc_event_local,
                                                cudaEventDisableTiming |
                                                cudaEventInterprocess),
                        exit_err, status);
        CUDA_CHECK_GOTO(cudaIpcGetEventHandle(&sync->ev_handle,
                                             sync->ipc_event_local),
                        exit_err, status);
    }

    ucc_memory_cpu_store_fence();
    bar = UCC_TL_CUDA_TEAM_BARRIER(team, 0);
    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(team), bar);
    if (status != UCC_OK) {
        tl_error(tl_team->context->lib, "failed to start shm barrier");
        goto exit_err;
    }

barrier:
    bar = UCC_TL_CUDA_TEAM_BARRIER(team, 0);
    status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(team), bar);
    if (status == UCC_INPROGRESS) {
        return status;
    } else if (status != UCC_OK) {
        goto exit_err;
    }

    for (i = 0; i < lib->cfg.max_concurrent; i++) {
        sync = UCC_TL_CUDA_TEAM_SYNC(team, UCC_TL_TEAM_RANK(team), i);
        for (j = 0 ; j < UCC_TL_TEAM_SIZE(team); j++) {
            if (j == UCC_TL_TEAM_RANK(team)) {
                continue;
            }
            peer_sync = UCC_TL_CUDA_TEAM_SYNC(team, j, i);
            CUDA_CHECK_GOTO(cudaIpcOpenEventHandle(&sync->data[j].ipc_event_remote,
                                                   peer_sync->ev_handle),
                            exit_err, status);
        }
    }
    team->oob_req = NULL;
    tl_info(tl_team->context->lib, "initialized tl team: %p", team);
    return UCC_OK;

exit_err:
    return status;
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
            UCC_TL_CUDA_DEFAULT_SCORE, ucc_tl_cuda_alg_id_to_init);
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
            UCC_TL_CUDA_DEFAULT_SCORE, ucc_tl_cuda_alg_id_to_init);
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
