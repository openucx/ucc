/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_nvls.h"

#include "tl_cuda.h"
#include "tl_cuda_coll.h"
#include "core/ucc_team.h"
#include "utils/arch/cuda_def.h"

#include <sys/syscall.h> // for pidfd_open and pidfd_getfd
#include <unistd.h>      // for close()

static ucc_status_t ucc_tl_cuda_nvls_check_support(int device)
{
    int      supported, fabric_supported;
    CUresult res;

    res =     cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                               device);
    if (res != CUDA_SUCCESS) {
        ucc_error("failed to query multicast support on device %d: %d", device, res);
        return UCC_ERR_NOT_SUPPORTED;
    }

    res = cuDeviceGetAttribute(&fabric_supported,
                               CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                               device);
    if (res != CUDA_SUCCESS) {
        ucc_error("failed to query fabric handle support on device %d: %d", device, res);
        return UCC_ERR_NOT_SUPPORTED;
    }

    ucc_debug("MULTICAST_SUPPORTED: %d, HANDLE_TYPE_FABRIC_SUPPORTED: %d\n",
              supported, fabric_supported);

    if (!supported) {
        ucc_error("multicast not supported on device %d", device);
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_nvls_get_granularity(CUmulticastObjectProp *mcProp, size_t *minGran,
                                 size_t *gran)
{
    ucc_status_t status;

    status = CUDADRV_FUNC(cuMulticastGetGranularity(
        minGran, mcProp, CU_MULTICAST_GRANULARITY_MINIMUM));
    if (status != UCC_OK) {
        ucc_error("failed to get multicast granularity minimum");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuMulticastGetGranularity(
        gran, mcProp, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    if (status != UCC_OK) {
        ucc_error("failed to get multicast granularity recommended");
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_nvls_create_multicast_object(CUmulticastObjectProp        *mcProp,
                                         CUmemGenericAllocationHandle *mcHandle,
                                         void *export_handle)
{
    ucc_status_t status;

    status = CUDADRV_FUNC(cuMulticastCreate(mcHandle, mcProp));
    if (status != UCC_OK) {
        ucc_error("failed to create multicast object");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuMemExportToShareableHandle(
        export_handle, *mcHandle, mcProp->handleTypes, 0));
    if (status != UCC_OK) {
        ucc_error("failed to export shareable handle");
        CUDADRV_FUNC(cuMemRelease(*mcHandle));
        return status;
    }

    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_nvls_share_handles_posix(struct ucc_tl_cuda_team *self, int export_handle,
                                     pid_t *shared_pids)
{
    ucc_status_t status;
    pid_t        currentPid = getpid();

    // Share PIDs across ranks
    status = self->oob.allgather(&currentPid, shared_pids, sizeof(currentPid),
                                 self->oob.coll_info, &self->oob_req);
    if (UCC_OK != status) {
        return status;
    }

    while (UCC_OK != (status = self->oob.req_test(self->oob_req))) {
        if (status < 0) {
            return status;
        }
    }
    self->oob.req_free(self->oob_req);
    self->oob_req = NULL;

    // Share the export handle
    if (UCC_TL_TEAM_RANK(self) == 0) {
        self->shared_handles[0] = export_handle;
    }

    status = self->oob.allgather(&export_handle, self->shared_handles,
                                 sizeof(export_handle), self->oob.coll_info,
                                 &self->oob_req);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(self), "failed to allgather export handle");
        return status;
    }

    while (UCC_OK != (status = self->oob.req_test(self->oob_req))) {
        if (status < 0) {
            tl_error(UCC_TL_TEAM_LIB(self), "failed to test allgather export handle");
            return status;
        }
    }
    self->oob.req_free(self->oob_req);
    self->oob_req = NULL;

    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_nvls_share_handles_fabric(struct ucc_tl_cuda_team *self, CUmemFabricHandle export_handle,
                                      CUmemFabricHandle *shared_fabric_handles)
{
    ucc_status_t status;

    // Share the export handle
    if (UCC_TL_TEAM_RANK(self) == 0) {
        shared_fabric_handles[0] = export_handle;
    }

    status = self->oob.allgather(&export_handle, shared_fabric_handles,
                                 sizeof(export_handle), self->oob.coll_info,
                                 &self->oob_req);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(self), "failed to allgather export fabric handle");
        return status;
    }

    while (UCC_OK != (status = self->oob.req_test(self->oob_req))) {
        if (status < 0) {
            tl_error(UCC_TL_TEAM_LIB(self), "failed to test allgather export fabric handle");
            return status;
        }
    }
    self->oob.req_free(self->oob_req);
    self->oob_req = NULL;

    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_nvls_import_handle_posix(struct ucc_tl_cuda_team *self,
                               int export_handle,
                               pid_t targetPid,
                               CUmemGenericAllocationHandle *mcHandle)
{
    int          pidFd, peerFd;
    ucc_status_t status;

    pidFd = syscall(SYS_pidfd_open, targetPid, 0);
    if (pidFd < 0) {
        tl_error(UCC_TL_TEAM_LIB(self), "failed to open pidfd for pid %d", targetPid);
        return UCC_ERR_NO_RESOURCE;
    }

    peerFd = syscall(SYS_pidfd_getfd, pidFd, export_handle, 0);
    if (peerFd < 0) {
        tl_error(UCC_TL_TEAM_LIB(self), "failed to get peer fd: %s (errno=%d)", strerror(errno), errno);
        close(pidFd);
        return UCC_ERR_NO_RESOURCE;
    }

    void *p = (void *)((uint64_t)peerFd);
    status  = CUDADRV_FUNC(cuMemImportFromShareableHandle(
        mcHandle, p, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    close(peerFd);
    close(pidFd);

    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(self),
                 "failed to import POSIX file descriptor handle from rank 0");
        return status;
    }

    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_nvls_import_handle_fabric(struct ucc_tl_cuda_team *self,
                               CUmemFabricHandle export_handle,
                               CUmemGenericAllocationHandle *mcHandle)
{
    ucc_status_t status = CUDADRV_FUNC(cuMemImportFromShareableHandle(
        mcHandle, &export_handle, CU_MEM_HANDLE_TYPE_FABRIC));

    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(self),
                 "failed to import fabric handle from rank 0");
        return status;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_sync_barrier(struct ucc_tl_cuda_team *team)
{
    ucc_debug("RANK %d: syncing barrier using oob allgather", UCC_TL_TEAM_RANK(team));
    int32_t barrier_value = 0x1234;
    int32_t *shared_barrier_values = ucc_malloc(UCC_TL_TEAM_SIZE(team) * sizeof(barrier_value), "shared_barrier_values");
    if (!shared_barrier_values) {
        return UCC_ERR_NO_MEMORY;
    }
    // instead of barrier, launch collective operation using oob
    ucc_status_t status = team->oob.allgather(&barrier_value, shared_barrier_values, sizeof(barrier_value),
        team->oob.coll_info, &team->oob_req);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(team), "nvls sync barrier failed to init oob allgather %s", ucc_status_string(status));
        ucc_free(shared_barrier_values);
        return status;
    }
    // Wait for barrier completion
    while (UCC_OK != (status = team->oob.req_test(team->oob_req))) {
        if (status < 0) {
            tl_error(UCC_TL_TEAM_LIB(team), "nvls sync barrier failed to test oob req %s", ucc_status_string(status));
            ucc_free(shared_barrier_values);
            return status;
        }
    }
    team->oob.req_free(team->oob_req);
    team->oob_req = NULL;
    ucc_free(shared_barrier_values);

    ucc_debug("RANK %d: synced barrier using oob allgather", UCC_TL_TEAM_RANK(team));

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_nvls_init(struct ucc_tl_cuda_team *self,
                                   ucc_base_context_t      *tl_context)
{
    ucc_tl_cuda_lib_t            *lib                    = ucc_derived_of(tl_context->lib, ucc_tl_cuda_lib_t);
    ucc_tl_cuda_nvls_t           *nvls                   = &self->nvls;
    const size_t                  symmetric_size         = lib->cfg.max_concurrent * (lib->cfg.nvls_symmetric_size + NVLS_CONTROL_SIZE);
    size_t                        minGran                = 0;
    size_t                        gran                   = 0;
    size_t                        mcSize                 = 0;
    int                           export_handle          = 0;
    int                           device                 = 0;
    int                           multi_node             = 0;
    pid_t                        *shared_pids            = NULL;
    CUmemFabricHandle            *shared_fabric_handles  = NULL;
    CUmemFabricHandle             fabric_handle          = {};
    void                         *uc_va                  = NULL;
    void                         *mc_va                  = NULL;
    ucc_status_t                  status                 = UCC_OK;
    CUmemGenericAllocationHandle  mcHandle               = 0;
    CUmemGenericAllocationHandle  memhandle              = 0;
    CUmulticastObjectProp         mcProp                 = {};
    CUmemAllocationHandleType     handleType;

    // Initialize nvls struct to ensure safe cleanup on error
    memset(nvls, 0, sizeof(*nvls));

    multi_node = !ucc_team_map_is_single_node(self->super.super.params.team,
                                               self->super.super.params.map);
    handleType = multi_node ? CU_MEM_HANDLE_TYPE_FABRIC :
                              CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;


    // Initialize multicast properties
    mcProp.numDevices  = UCC_TL_TEAM_SIZE(self);
    mcProp.size        = symmetric_size;
    mcProp.handleTypes = handleType;
    mcProp.flags       = 0;

    ucc_debug(
        "RANK %d: numDevices: %d, size: %zu, handleTypes: %lld, flags: %lld\n",
        UCC_TL_TEAM_RANK(self), mcProp.numDevices, mcProp.size,
        mcProp.handleTypes, mcProp.flags);

    // Get current device and check support
    CUDA_CHECK(cudaGetDevice(&device));
    ucc_debug("RANK %d: device: %d\n", UCC_TL_TEAM_RANK(self), device);

    status = ucc_tl_cuda_nvls_check_support(device);
    if (status != UCC_OK) {
        return status;
    }

    // Get granularity requirements
    status = ucc_tl_cuda_nvls_get_granularity(&mcProp, &minGran, &gran);
    if (status != UCC_OK) {
        return status;
    }

    if (UCC_TL_TEAM_RANK(self) == 0) {
        ucc_debug("NVLS multicast granularity: gran = %lu, minGran = %lu\n",
                  gran, minGran);
    }

    // Calculate aligned size
    mcSize      = ((symmetric_size + gran - 1) / gran) * gran;
    mcProp.size = mcSize;

    // Allocate shared handles array
    self->shared_handles = ucc_malloc(
        UCC_TL_TEAM_SIZE(self) * sizeof(export_handle), "shared_handles");
    if (!self->shared_handles) {
        status = UCC_ERR_NO_MEMORY;
        goto cleanup;
    }

    // Allocate shared PIDs array
    shared_pids =
        ucc_malloc(UCC_TL_TEAM_SIZE(self) * sizeof(pid_t), "shared_pids");
    if (!shared_pids) {
        status = UCC_ERR_NO_MEMORY;
        goto cleanup;
    }

    // Allocate shared fabric handles array
    shared_fabric_handles = ucc_malloc(
        UCC_TL_TEAM_SIZE(self) * sizeof(fabric_handle), "shared_fabric_handles");
    if (!shared_fabric_handles) {
        status = UCC_ERR_NO_MEMORY;
        goto cleanup;
    }

    // Create multicast object on rank 0
    if (UCC_TL_TEAM_RANK(self) == 0) {
        if (multi_node) {
            status = ucc_tl_cuda_nvls_create_multicast_object(&mcProp, &mcHandle,
                                                          &fabric_handle);
        } else {
            status = ucc_tl_cuda_nvls_create_multicast_object(&mcProp, &mcHandle,
                                                          &export_handle);
        }
        if (status != UCC_OK) {
            ucc_error("failed to create multicast object");
            goto cleanup;
        }
    }

    // Share handles across ranks
    if (multi_node) {
        status = ucc_tl_cuda_nvls_share_handles_fabric(self, fabric_handle, shared_fabric_handles);
    } else {
        status = ucc_tl_cuda_nvls_share_handles_posix(self, export_handle, shared_pids);
    }
    if (status != UCC_OK) {
        goto cleanup;
    }

    // Import handle on non-root ranks
    if (UCC_TL_TEAM_RANK(self) != 0) {
        if (multi_node) {
            status        = ucc_tl_cuda_nvls_import_handle_fabric(self, shared_fabric_handles[0], &mcHandle);
        } else {
            status        = ucc_tl_cuda_nvls_import_handle_posix(self, self->shared_handles[0],
                                                           shared_pids[0], &mcHandle);
        }
        if (status != UCC_OK) {
            goto cleanup;
        }
    }

    // Add device to multicast object
    status = CUDADRV_FUNC(cuMulticastAddDevice(mcHandle, device));
    if (status != UCC_OK) {
        ucc_error("failed to add device to multicast");
        goto cleanup;
    }
    ucc_debug("RANK %d: added device %d to multicast\n", UCC_TL_TEAM_RANK(self),
              device);

    // Synchronize all ranks after adding devices
    status = ucc_tl_cuda_nvls_sync_barrier(self);
    if (status != UCC_OK) {
        goto cleanup;
    }

    // Allocate physical memory
    CUmemAllocationProp prop  = {};
    prop.type                 = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type        = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id          = device;
    prop.requestedHandleTypes = handleType;

    status = CUDADRV_FUNC(cuMemCreate(&memhandle, mcSize, &prop, 0));
    if (status != UCC_OK) {
        ucc_error("failed to create memory allocation for multicast");
        goto cleanup;
    }

    // Set up memory access descriptor
    CUmemAccessDesc accessDesc = {};
    accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
    accessDesc.location.id     = device;
    accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    // Reserve and map unicast virtual address space
    status = CUDADRV_FUNC(
        cuMemAddressReserve((CUdeviceptr *)&uc_va, mcSize, minGran, 0U, 0));
    if (status != UCC_OK) {
        ucc_error("failed to reserve virtual address space");
        goto cleanup;
    }

    status =
        CUDADRV_FUNC(cuMemMap((CUdeviceptr)uc_va, mcSize, 0, memhandle, 0));
    if (status != UCC_OK) {
        ucc_error("failed to map memory allocation");
        goto cleanup;
    }

    status = CUDADRV_FUNC(
        cuMemSetAccess((CUdeviceptr)uc_va, mcSize, &accessDesc, 1));
    if (status != UCC_OK) {
        ucc_error("failed to set memory access");
        goto cleanup;
    }

    // Bind memory to multicast object
    size_t mcOffset = 0;
    status          = CUDADRV_FUNC(
        cuMulticastBindAddr(mcHandle, mcOffset, (CUdeviceptr)uc_va, mcSize, 0));
    if (status != UCC_OK) {
        ucc_error("failed to bind memory to multicast");
        goto cleanup;
    }

    // Synchronize after binding
    status = ucc_tl_cuda_nvls_sync_barrier(self);
    if (status != UCC_OK) {
        goto cleanup;
    }

    // Reserve and map multicast virtual address space
    status = CUDADRV_FUNC(
        cuMemAddressReserve((CUdeviceptr *)&mc_va, mcSize, minGran, 0U, 0));
    if (status != UCC_OK) {
        ucc_error("failed to reserve multicast virtual address space");
        goto cleanup;
    }

    status = CUDADRV_FUNC(cuMemMap((CUdeviceptr)mc_va, mcSize, 0, mcHandle, 0));
    if (status != UCC_OK) {
        ucc_error("failed to map multicast memory");
        goto cleanup;
    }

    status = CUDADRV_FUNC(
        cuMemSetAccess((CUdeviceptr)mc_va, mcSize, &accessDesc, 1));
    if (status != UCC_OK) {
        ucc_error("failed to set multicast memory access");
        goto cleanup;
    }

    ucc_debug("Rank: %d symmetric memory is set: %p [%ld bytes]\n",
              UCC_TL_TEAM_RANK(self), mc_va, mcSize);

    // Store the handles for cleanup in team destroy
    nvls->mc_handle    = mcHandle;
    nvls->mc_va        = (CUdeviceptr)mc_va;
    nvls->uc_va        = (CUdeviceptr)uc_va;
    nvls->mc_memhandle = memhandle;
    nvls->mc_size      = mcSize;
    nvls->mc_offset    = mcOffset;
    nvls->coll_ids     = ucc_malloc(lib->cfg.max_concurrent * sizeof(size_t), "coll_ids");
    if (!nvls->coll_ids) {
        status = UCC_ERR_NO_MEMORY;
        goto cleanup;
    }
    // Initialize the coll_ids to 0
    memset(nvls->coll_ids, 0, lib->cfg.max_concurrent * sizeof(size_t));

    if (UCC_TL_TEAM_RANK(self) == 0) {
        // root rank zero-initializes the control region for each task slot
        size_t stride = lib->cfg.nvls_symmetric_size + NVLS_CONTROL_SIZE;
        void  *control_uc0 =
            PTR_OFFSET((void *)uc_va, lib->cfg.nvls_symmetric_size);
        CUDA_CHECK(cudaMemset2D(control_uc0, stride, 0, NVLS_CONTROL_SIZE,
                                lib->cfg.max_concurrent));
    }

    ucc_free(shared_pids);
    ucc_free(shared_fabric_handles);
    return UCC_OK;

cleanup:
    // Cleanup on error - free temporary allocations
    if (shared_pids) {
        ucc_free(shared_pids);
    }
    if (shared_fabric_handles) {
        ucc_free(shared_fabric_handles);
    }

    // Clean up CUDA resources - check local variables, not nvls struct
    // since nvls fields may not be initialized yet
    if (mc_va) {
        CUDADRV_FUNC(cuMemUnmap((CUdeviceptr)mc_va, mcSize));
        CUDADRV_FUNC(cuMemAddressFree((CUdeviceptr)mc_va, mcSize));
    }

    if (uc_va) {
        CUDADRV_FUNC(cuMemUnmap((CUdeviceptr)uc_va, mcSize));
        CUDADRV_FUNC(cuMemAddressFree((CUdeviceptr)uc_va, mcSize));
    }

    if (memhandle) {
        CUDADRV_FUNC(cuMemRelease(memhandle));
    }

    // Only rank 0 owns the multicast handle
    if (UCC_TL_TEAM_RANK(self) == 0 && mcHandle) {
        CUDADRV_FUNC(cuMemRelease(mcHandle));
    }

    if (self->shared_handles) {
        ucc_free(self->shared_handles);
        self->shared_handles = NULL;
    }

    return status;
}

ucc_status_t ucc_tl_cuda_nvls_destroy(struct ucc_tl_cuda_team *self,
                                      ucc_base_context_t      *tl_context)
{
    int device = 0;

    // Get the actual device used during init
    if (cudaGetDevice(&device) != cudaSuccess) {
        tl_error(UCC_TL_TEAM_LIB(self), "failed to get current device during cleanup");
        device = 0; // fallback to device 0
    }

    if (UCC_TL_TEAM_RANK(self) == 0) {
        // Rank 0: unbind and release multicast object
        if (self->nvls.mc_handle) {
            CUDADRV_FUNC(cuMulticastUnbind(self->nvls.mc_handle, device,
                                           self->nvls.mc_offset,
                                           self->nvls.mc_size));
            CUDADRV_FUNC(cuMemRelease(self->nvls.mc_handle));
        }
    }

    // ALL ranks: clean up their mapped memory
    if (self->nvls.mc_va) {
        CUDADRV_FUNC(cuMemUnmap(self->nvls.mc_va, self->nvls.mc_size));
        CUDADRV_FUNC(cuMemAddressFree(self->nvls.mc_va, self->nvls.mc_size));
    }

    if (self->nvls.uc_va) {
        CUDADRV_FUNC(cuMemUnmap(self->nvls.uc_va, self->nvls.mc_size));
        CUDADRV_FUNC(cuMemAddressFree(self->nvls.uc_va, self->nvls.mc_size));
    }

    // ALL ranks: release their memory handle
    if (self->nvls.mc_memhandle) {
        CUDADRV_FUNC(cuMemRelease(self->nvls.mc_memhandle));
    }

    // ALL ranks: free coll_ids
    if (self->nvls.coll_ids) {
        ucc_free(self->nvls.coll_ids);
        self->nvls.coll_ids = NULL;
    }

    // ALL ranks: free shared_handles
    if (self->shared_handles) {
        ucc_free(self->shared_handles);
        self->shared_handles = NULL;
    }

    return UCC_OK;
}
