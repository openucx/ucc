/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"
#include "tl_cuda_coll.h"
#include "core/ucc_team.h"
#include "coll_score/ucc_coll_score.h"
#include <sys/shm.h>

UCC_CLASS_INIT_FUNC(ucc_tl_cuda_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_cuda_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_cuda_context_t);
    ucc_tl_cuda_lib_t     *lib =
        ucc_derived_of(tl_context->lib, ucc_tl_cuda_lib_t);
    ucc_tl_cuda_sync_t *sync;
    ucc_status_t status;
    int shm_id, i, j;
    size_t ctrl_size;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params->team);

    if (self->size > UCC_TL_CUDA_MAX_PEERS) {
        tl_info(tl_context->lib, "team size is too large, max supported %d",
                UCC_TL_CUDA_MAX_PEERS);
        return UCC_ERR_NOT_SUPPORTED;
    }
    self->rank   = params->rank;
    self->oob    = params->params.oob;
    self->size   = self->oob.n_oob_eps;
    self->stream = NULL;
    for (i = 0; i < self->size; i++) {
        if (!ucc_rank_on_local_node(i, params->team)) {
            tl_info(tl_context->lib, "rank %d is on different node, "
                    "multinode isn't supported", i);
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    ctrl_size = (sizeof(ucc_tl_cuda_sync_t) + sizeof(ucc_tl_cuda_sync_data_t) *
                (self->size - 1)) * self->size * lib->cfg.max_concurrent;
    shm_id = -1;
    self->sync = (void*)-1;
    if (self->rank == 0) {
        shm_id = shmget(IPC_PRIVATE, ctrl_size, IPC_CREAT | 0666);
        if (shm_id < 0) {
            tl_error(tl_context->lib, "failed to shmget with IPC_PRIVATE, "
                     "size %zd, IPC_CREAT errno: %d(%s)", ctrl_size,
                     errno, strerror(errno));
            status = UCC_ERR_NO_MEMORY;
            goto exit_err;
        }
        self->sync = shmat(shm_id, NULL, 0);
        shmctl(shm_id, IPC_RMID, NULL);
        if (self->sync == (void *)-1) {
            tl_error(tl_context->lib, "failed to shmat errno: %d(%s)",
                     errno, strerror(errno));
            status = UCC_ERR_NO_MEMORY;
            goto free_shm;
        }
        memset(self->sync, 0, ctrl_size);
        for (i = 0; i < lib->cfg.max_concurrent; i++) {
            for (j = 0; j < self->size; j++) {
                sync = UCC_TL_CUDA_TEAM_SYNC(self, j, i);
                sync->status = UCC_INPROGRESS;
            }
        }
    }

    self->ids = ucc_malloc((self->size + 1) * sizeof(*(self->ids)), "ids");
    if (!self->ids) {
        tl_error(tl_context->lib, "failed to alloc ranks id");
        status = UCC_ERR_NO_MEMORY;
        goto free_shmdt;
    }
    self->ids[self->size].device = ctx->device;
    self->ids[self->size].shm    = shm_id;
    status = self->oob.allgather(&self->ids[self->size], self->ids,
                                 sizeof(ucc_tl_cuda_rank_id_t),
                                 self->oob.coll_info, &self->oob_req);
    if (UCC_OK != status) {
        tl_error(tl_context->lib, "failed to start oob allgather");
        goto free_devices;
    }
    tl_info(tl_context->lib, "posted tl team: %p", self);

    self->seq_num   = 1;
    return UCC_OK;

free_devices:
    ucc_free(self->ids);
free_shmdt:
    if (self->sync != (void*)-1) {
        shmdt(self->sync);
    }
free_shm:
    if (shm_id != -1) {
        shmctl(shm_id, IPC_RMID, NULL);
    }
exit_err:
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_team_t)
{
    ucc_tl_cuda_lib_t *lib = ucc_derived_of(self->super.super.context->lib,
                                            ucc_tl_cuda_lib_t);
    ucc_tl_cuda_sync_t *sync;
    int i, j;

    tl_info(self->super.super.context->lib, "finalizing tl team: %p", self);
    if (self->ids) {
        if (self->sync != (void*)-1) {
            for (i = 0; i < lib->cfg.max_concurrent; i++) {
                for (j = 0; j < self->size; j++) {
                    if (j == self->rank) {
                        continue;
                    }
                    sync = UCC_TL_CUDA_TEAM_SYNC(self, j, i);
                    if (sync->data[j].ipc_event_remote) {
                        cudaEventDestroy(sync->data[j].ipc_event_remote);
                    }
                }
                sync = UCC_TL_CUDA_TEAM_SYNC(self, self->rank, i);
                if (sync->ipc_event_local) {
                    cudaEventDestroy(sync->ipc_event_local);
                }
            }
            shmdt(self->sync);
        }
        if (self->rank == 0) {
            shmctl(self->ids[0].shm, IPC_RMID, NULL);
        }
        ucc_free(self->ids);
    }
    if (self->stream) {
        cudaStreamDestroy(self->stream);
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
    volatile ucc_tl_cuda_sync_t *peer_sync;
    int i, j, peer_access, dev, peer_dev, shm_id;

    if (team->oob_req == NULL) {
        return UCC_OK;
    }
    status = team->oob.req_test(team->oob_req);
    if (status == UCC_INPROGRESS) {
        return UCC_INPROGRESS;
    } else if (status < 0) {
        tl_error(tl_team->context->lib, "oob allgather failed");
        goto exit_err;
    }
    team->oob.req_free(team->oob_req);
    team->oob_req = NULL;
    dev = team->ids[team->rank].device;
    for (i = 0; i < team->size; i++) {
        if (i != team->rank) {
            peer_dev = team->ids[i].device;
            CUDACHECK_GOTO(cudaDeviceCanAccessPeer(&peer_access, dev, peer_dev),
                           exit_err, status, tl_team->context->lib);
            if (!peer_access) {
                tl_info(tl_team->context->lib,
                        "dev %d rank %d is not accesible from dev %d rank %d",
                        peer_dev, i, dev, team->rank);
                status = UCC_ERR_NOT_SUPPORTED;
                goto exit_err;
            }
        }
    }
    shm_id = team->ids[0].shm;
    if (team->rank != 0) {
        team->sync = shmat(shm_id, NULL, 0);
        if (team->sync == (void *)-1) {
            tl_error(tl_team->context->lib, "failed to shamt errno: %d (%s)",
                     errno, strerror(errno));
            status = UCC_ERR_NO_MEMORY;
            goto exit_err;
        }
    }
    CUDACHECK_GOTO(cudaStreamCreateWithFlags(&team->stream,
                                             cudaStreamNonBlocking),
                   exit_err, status, tl_team->context->lib);

    for (i = 0; i < lib->cfg.max_concurrent; i++) {
        sync = UCC_TL_CUDA_TEAM_SYNC(team, team->rank, i);
        CUDACHECK_GOTO(cudaEventCreateWithFlags(&sync->ipc_event_local,
                                                cudaEventDisableTiming |
                                                cudaEventInterprocess),
                       exit_err, status, tl_team->context->lib);
        CUDACHECK_GOTO(cudaIpcGetEventHandle(&sync->ev_handle,
                                             sync->ipc_event_local),
                       exit_err, status, tl_team->context->lib);
        sync->status = UCC_OK;
        __sync_synchronize();
        asm volatile("": : :"memory");
        for (j = 0; j < team->size; j++) {
            if (j == team->rank) {
                continue;
            }
            peer_sync = UCC_TL_CUDA_TEAM_SYNC(team, j, i);
            while (peer_sync->status != UCC_OK);
            CUDACHECK_GOTO(cudaIpcOpenEventHandle(&sync->data[j].ipc_event_remote,
                                                  peer_sync->ev_handle),
                           exit_err, status, tl_team->context->lib);
        }
    }

    tl_info(tl_team->context->lib, "initialized tl team: %p", team);
    return UCC_OK;

exit_err:
    return status;
}

ucc_status_t ucc_tl_cuda_team_get_scores(ucc_base_team_t *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_lib_t  *lib  = UCC_TL_CUDA_TEAM_LIB(team);
    ucc_coll_score_t   *score;
    ucc_status_t        status;

    status =
        ucc_coll_score_build_default(tl_team, UCC_TL_CUDA_DEFAULT_SCORE,
                                     ucc_tl_cuda_coll_init,
                                     UCC_TL_CUDA_SUPPORTED_COLLS,
                                     NULL, 0, &score);
    if (UCC_OK != status) {
        return status;
    }

    if (strlen(lib->super.super.score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            lib->super.super.score_str, score, team->size,
            ucc_tl_cuda_coll_init, &team->super.super,
            UCC_TL_CUDA_DEFAULT_SCORE, NULL);
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
