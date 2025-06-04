/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sys/syscall.h>

static ucc_status_t
ucc_tl_cuda_team_init_nvls_multicast(ucc_tl_cuda_team_t *self,
                                     ucc_base_context_t *tl_context)
{
    size_t symmetric_size = 1024ULL * 1024ULL * 512ULL;
    // multicast
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmulticastObjectProp mcProp = {};
    mcProp.numDevices            = UCC_TL_TEAM_SIZE(self);
    mcProp.size                  = symmetric_size;
    mcProp.handleTypes           = handleType;
    mcProp.flags                 = 0;

    ucc_print("RANK %d: numDevices: %d, size: %zu, handleTypes: %lld, flags: %lld\n", UCC_TL_TEAM_RANK(self), mcProp.numDevices, mcProp.size, mcProp.handleTypes, mcProp.flags);
    size_t minGran, gran;
    gran    = 0;
    minGran = 0;


    CUcontext cu_ctx;
    CUresult cu_st;
    cu_st = cuCtxGetCurrent(&cu_ctx);
    if (cu_st != CUDA_SUCCESS) {
        ucc_error("failed to get current CUDA context");
        return UCC_ERR_NOT_SUPPORTED;
    }
    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    ucc_print("RANK %d: device: %d\n", UCC_TL_TEAM_RANK(self), device);
    int supported;
    cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
    ucc_print("MULTICAST_SUPPORTED: %d\n", supported);
    int fabric_supported;
    cuDeviceGetAttribute(&fabric_supported, CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED, device);
    ucc_print("HANDLE_TYPE_FABRIC_SUPPORTED: %d\n", fabric_supported);

    // Try to initialize NVLS multicast, but continue if not supported
    ucc_status_t mc_status = CUDADRV_FUNC(cuMulticastGetGranularity(&minGran, &mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
    if (mc_status != UCC_OK) {
        ucc_error("failed to get multicast granularity minimum");
        return UCC_ERR_NOT_SUPPORTED;
    }
    mc_status = CUDADRV_FUNC(cuMulticastGetGranularity(&gran, &mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    if (mc_status != UCC_OK) {
        ucc_error("failed to get multicast granularity recommended");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (UCC_TL_TEAM_RANK(self) == 0) {
        ucc_print("NVLS multicast granularity: gran = %lu, minGrad = %lu\n",
                  gran, minGran);
    }

    size_t mcSize = ((symmetric_size + gran - 1) / gran) * gran;
    mcProp.size   = mcSize;
    
    // only one rank creates the multicast object
    CUmemGenericAllocationHandle mcHandle;
    int export_handle = 0;
    ucc_status_t status = UCC_OK;

    int myDevice;
    cuCtxGetDevice(&myDevice);
    ucc_print("RANK %d: myDevice: %d\n", UCC_TL_TEAM_RANK(self), myDevice);

    if (UCC_TL_TEAM_RANK(self) == 0) {
        // Now create the multicast object
        mc_status = CUDADRV_FUNC(cuMulticastCreate(&mcHandle, &mcProp));
        if (mc_status != UCC_OK) {
            ucc_error("failed to create multicast object");
            return UCC_ERR_NOT_SUPPORTED;
        }

        mc_status = CUDADRV_FUNC(cuMemExportToShareableHandle(
            &export_handle, mcHandle, handleType, 0 /*flags*/));
        if (mc_status != UCC_OK) {
            ucc_error("failed to export shareable handle");
            goto error;
        }
    }

    self->shared_handles = ucc_malloc(UCC_TL_TEAM_SIZE(self) * sizeof(export_handle), "shared_handles");
    if (!self->shared_handles) {
        goto error;
    }

    pid_t currentPid = getpid();
    pid_t *shared_pids = ucc_malloc(UCC_TL_TEAM_SIZE(self) * sizeof(currentPid), "shared_pids");
    if (!shared_pids) {
        goto error;
    }
    status = self->oob.allgather(&currentPid, shared_pids, sizeof(currentPid), self->oob.coll_info, &self->oob_req);
    if (UCC_OK != status) {
        goto error;
    }
    while (UCC_OK != (status = self->oob.req_test(self->oob_req))) {
        if (status < 0) {
            goto error;
        }
    }
    self->oob.req_free(self->oob_req);
    self->oob_req = NULL;

    if (UCC_TL_TEAM_RANK(self) != 0) {
        currentPid = shared_pids[0];
    }

    int pidFd = syscall(SYS_pidfd_open, currentPid, 0);

    // send the handle to all ranks using allgather and wait for all ranks to receive it
    if (UCC_TL_TEAM_RANK(self) == 0) {
        self->shared_handles[0] = export_handle;
    }
    status = self->oob.allgather(&export_handle, self->shared_handles, sizeof(export_handle), self->oob.coll_info, &self->oob_req);
    if (UCC_OK != status) {
        goto error;
    }
    while (UCC_OK != (status = self->oob.req_test(self->oob_req))) {
        if (status < 0) {
            goto error;
        }
    }
    self->oob.req_free(self->oob_req);
    self->oob_req = NULL;

    if (UCC_TL_TEAM_RANK(self) == 0) {
        ucc_print("rank %d: export_handle: %d\n", UCC_TL_TEAM_RANK(self),
                  export_handle);
    } else {
        ucc_print("rank %d: export_handle: %d\n", UCC_TL_TEAM_RANK(self),
                  self->shared_handles[0]);
        export_handle = self->shared_handles[0];
    }

    int peerFd = 0;
    peerFd = syscall(SYS_pidfd_getfd, pidFd, export_handle, 0);
    if (peerFd < 0) {
        ucc_error("failed to get peer fd");
        goto error;
    }

    if (UCC_TL_TEAM_RANK(self) != 0) {
        // receive the handle from rank 0
        void * p = (void*) ((uint64_t) peerFd);
        ucc_print("rank %d: export_handle: %d\n", UCC_TL_TEAM_RANK(self),
                  export_handle);
        status = CUDADRV_FUNC(cuMemImportFromShareableHandle(
            &mcHandle, p, handleType));
        if (status != UCC_OK) {
            ucc_error("failed to import handle from rank 0");
            return status;
        }
    }
    status = CUDADRV_FUNC(cuMulticastAddDevice(mcHandle, device));
    if (status != UCC_OK) {
        ucc_error("failed to add device to multicast");
        goto error;
    }
    ucc_debug("rank %d: added device %d to multicast\n", UCC_TL_TEAM_RANK(self), device);

    // wait for all ranks to add the device to the multicast object
    // TODO: rework this to use a more efficient barrier
    ucc_tl_cuda_shm_barrier_t *bar = UCC_TL_CUDA_TEAM_BARRIER(self, 0);

    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(self), bar);
    if (status != UCC_OK) {
        ucc_error("failed to start shm barrier");
        goto error;
    }
    status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(self), bar);
    while (status == UCC_INPROGRESS) {
        status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(self), bar);
    }
    if (status != UCC_OK) {
        ucc_error("failed to test shm barrier");
        goto error;
    }

    ucc_print("rank %d: barrier ok\n", UCC_TL_TEAM_RANK(self));

    // allocate memory and bind to the multicast object
    CUmemGenericAllocationHandle memhandle;
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device;
    prop.requestedHandleTypes = handleType;

    // allocate physical memory (data buffer)
    status = CUDADRV_FUNC(cuMemCreate(&memhandle, mcSize, &prop, 0 /*flags*/));
    if (status != UCC_OK) {
        ucc_error("failed to create memory allocation for multicast");
        goto error;
    }


    void* uc_va;
    void* mc_va;
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id = device;
    accessDesc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Map a VA to MC space
    status = CUDADRV_FUNC(
        cuMemAddressReserve((CUdeviceptr *)&uc_va, mcSize, minGran, 0U, 0));
    if (status != UCC_OK) {
        ucc_error("failed to reserve virtual address space");
        goto error;
    }
    // CUDA_CHECK(cudaMemset(uc_va, 0, mcSize));

    status = CUDADRV_FUNC(cuMemMap((CUdeviceptr)uc_va, mcSize, 0, memhandle, 0));
    if (status != UCC_OK) {
        ucc_error("failed to map memory allocation");
        goto error;
    }
    status = CUDADRV_FUNC(cuMemSetAccess((CUdeviceptr)uc_va, mcSize, &accessDesc, 1));
    if (status != UCC_OK) {
        ucc_error("failed to set memory access");
        goto error;
    }

    size_t mcOffset = 0;
    status = CUDADRV_FUNC(cuMulticastBindAddr(mcHandle, mcOffset, (CUdeviceptr)uc_va, mcSize, 0));
    if (status != UCC_OK) {
        ucc_error("failed to bind memory to multicast");
        goto error;
    }

    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(self), bar);
    if (status != UCC_OK) {
        ucc_error("failed to start shm barrier after init");
        goto error;
    }
    status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(self), bar);
    while (status == UCC_INPROGRESS) {
        status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(self), bar);
    }
    if (status != UCC_OK) {
        ucc_error("failed to test shm barrier after init");
        goto error;
    }

    ucc_print("rank %d: barrier after init ok\n", UCC_TL_TEAM_RANK(self));

    // Map a VA to MC space
    CUDADRV_FUNC(
        cuMemAddressReserve((CUdeviceptr *)&mc_va, mcSize, minGran, 0U, 0));
    CUDADRV_FUNC(cuMemMap((CUdeviceptr)mc_va, mcSize, 0, mcHandle, 0));
    // set access on MC address
    CUDADRV_FUNC(cuMemSetAccess((CUdeviceptr)mc_va, mcSize, &accessDesc, 1));

    ucc_print("Rank: %d symmetric memory is set: %p [%ld bytes]\n", UCC_TL_TEAM_RANK(self), mc_va, mcSize);

    // Store the handles for cleanup in team destroy
    self->mc_handle = mcHandle;
    self->mc_va = (CUdeviceptr) mc_va;
    self->uc_va = (CUdeviceptr) uc_va;
    self->mc_memhandle = memhandle;
    self->mc_size = mcSize;
    self->mc_offset = mcOffset;


    return UCC_OK;
error:
    if (UCC_TL_TEAM_RANK(self) == 0) {
        CUDADRV_FUNC(cuMemUnmap(self->mc_va, self->mc_size));
        CUDADRV_FUNC(cuMemRelease(self->mc_handle));
        CUDADRV_FUNC(cuMemAddressFree(self->mc_va, self->mc_size));
    }
    return status;
}

UCC_CLASS_INIT_FUNC(ucc_tl_cuda_team_t, ucc_base_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    ucc_tl_cuda_context_t *ctx =
        ucc_derived_of(tl_context, ucc_tl_cuda_context_t);
    ucc_tl_cuda_lib_t     *lib =
        ucc_derived_of(tl_context->lib, ucc_tl_cuda_lib_t);
    // Number of preallocated resource groups for tasks, including the active set.
    uint32_t      resource_num = lib->cfg.max_concurrent * 2;
    ucc_tl_cuda_shm_barrier_t *bar;
    ucc_status_t status;
    int shm_id, i, j;
    size_t ctrl_size, alloc_size, scratch_size;
    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_team_t, &ctx->super, params);

    self->oob         = params->params.oob;
    self->stream      = NULL;
    self->topo        = NULL;
    self->scratch.loc = NULL;

    if (!ucc_team_map_is_single_node(params->team, params->map)) {
        tl_debug(tl_context->lib, "multinode team is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    self->ids = ucc_malloc((UCC_TL_TEAM_SIZE(self) + 1) * sizeof(*(self->ids)),
                            "ids");
    if (!self->ids) {
        tl_error(tl_context->lib, "failed to alloc ranks id");
        return UCC_ERR_NO_MEMORY;
    }

    // active set
    scratch_size = resource_num * lib->cfg.scratch_size;
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
    ctrl_size *= 2; // active sets

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
        self->bar = (ucc_tl_cuda_shm_barrier_t *)UCC_TL_CUDA_TEAM_SYNC(
            self, 0, resource_num);
        /* active set */
        for (i = 0; i < resource_num; i++) {
            bar = UCC_TL_CUDA_TEAM_BARRIER(self, i);
            bar->tag = UCC_TL_CUDA_TAG_FREE; // mark as free
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
    self->ids[UCC_TL_TEAM_SIZE(self)].pci_id = ctx->device_id;
    self->ids[UCC_TL_TEAM_SIZE(self)].shm    = shm_id;
    status = self->oob.allgather(&self->ids[UCC_TL_TEAM_SIZE(self)], self->ids,
                                 sizeof(ucc_tl_cuda_rank_id_t),
                                 self->oob.coll_info, &self->oob_req);
    if (UCC_OK != status) {
        tl_error(tl_context->lib, "failed to start oob allgather");
        goto free_devices;
    }
    tl_debug(tl_context->lib, "posted tl team: %p", self);

    self->seq_num = 1;
    self->seq_num_active_set = 1;
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
    // Number of preallocated resource groups for tasks, including the active set.
    uint32_t  resource_num = lib->cfg.max_concurrent * 2;
    ucc_tl_cuda_sync_t *sync;
    cudaError_t st;
    int i, j;

    tl_debug(self->super.super.context->lib, "finalizing tl team: %p", self);
    if (self->topo) {
        ucc_tl_cuda_team_topo_destroy(self->topo);
    }
    if (self->mc_handle) {
        if (UCC_TL_TEAM_RANK(self) == 0) {
            CUDADRV_FUNC(cuMulticastUnbind(self->mc_handle, 0 /* device */, self->mc_offset, self->mc_size));
            CUDADRV_FUNC(cuMemRelease(self->mc_memhandle));
            
            CUDADRV_FUNC(cuMemUnmap(self->mc_va, self->mc_size));
            CUDADRV_FUNC(cuMemAddressFree(self->mc_va, self->mc_size));
            CUDADRV_FUNC(cuMemRelease(self->mc_handle));
        }
        // self->mc_handle = NULL;
        self->mc_va = 0;
        self->mc_size = 0;
    }
    if (self->ids) {
        if (self->sync != (void*)-1) {
            for (i = 0; i < resource_num; i++) {
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
    ucc_tl_cuda_team_t *team = ucc_derived_of(tl_team, ucc_tl_cuda_team_t);
    ucc_tl_cuda_lib_t  *lib  = ucc_derived_of(tl_team->context->lib,
                                              ucc_tl_cuda_lib_t);
    // Number of preallocated resource groups for tasks, including the active set.
    uint32_t    resource_num = lib->cfg.max_concurrent * 2;
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

    for (i = 0; i < UCC_TL_TEAM_SIZE(team); i++) {
           team->scratch.rem[i] = NULL;
    }

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
        ucc_tl_cuda_team_topo_print_proxies(&team->super, team->topo);
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
                                                       resource_num);
    }
    team->sync_state = (ucc_tl_cuda_sync_state_t*)PTR_OFFSET(team->bar,
                            sizeof(ucc_tl_cuda_shm_barrier_t) *
                            resource_num);
    CUDA_CHECK_GOTO(cudaStreamCreateWithFlags(&team->stream,
                    cudaStreamNonBlocking), exit_err, status);
    for (i = 0; i < resource_num; i++) {
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

    for (i = 0; i < resource_num; i++) {
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
    tl_debug(tl_team->context->lib, "initialized tl team: %p", team);

    status = ucc_tl_cuda_team_init_nvls_multicast(team, tl_team->context);
    if (status != UCC_OK) {
        ucc_print("failed to init nvls multicast");
        goto exit_err;
    }

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
    ucc_coll_score_team_info_t team_info;

    team_info.alg_fn              = ucc_tl_cuda_alg_id_to_init;
    team_info.default_score       = UCC_TL_CUDA_DEFAULT_SCORE;
    team_info.init                = ucc_tl_cuda_coll_init;
    team_info.num_mem_types       = 1;
    team_info.supported_mem_types = &mt;
    team_info.supported_colls     = UCC_TL_CUDA_SUPPORTED_COLLS;
    team_info.size                = UCC_TL_TEAM_SIZE(team);

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
            ucc_tl_cuda_default_alg_select_str[i], &team_info,
            &team->super.super, score);
        if (UCC_OK != status) {
            tl_error(tl_team->context->lib,
                     "failed to apply default coll select setting: %s",
                     ucc_tl_cuda_default_alg_select_str[i]);
            goto err;
        }
    }

    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(ctx->score_str, &team_info,
                                                &team->super.super, score);
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
