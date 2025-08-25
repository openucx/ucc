/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda_nvls.h"

#include "tl_cuda.h"
#include "tl_cuda_coll.h"
#include "utils/arch/cuda_def.h"

#include <sys/syscall.h> // for pidfd_open and pidfd_getfd
#include <unistd.h>      // for close()

static ucc_status_t ucc_tl_cuda_nvls_check_support(int device)
{
    int supported, fabric_supported;

    cuDeviceGetAttribute(&supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
                         device);
    cuDeviceGetAttribute(&fabric_supported,
                         CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                         device);

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
                                         int *export_handle)
{
    ucc_status_t status;

    status = CUDADRV_FUNC(cuMulticastCreate(mcHandle, mcProp));
    if (status != UCC_OK) {
        ucc_error("failed to create multicast object");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuMemExportToShareableHandle(
        export_handle, *mcHandle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));
    if (status != UCC_OK) {
        ucc_error("failed to export shareable handle");
        CUDADRV_FUNC(cuMemRelease(*mcHandle));
        return status;
    }

    return UCC_OK;
}

static ucc_status_t
ucc_tl_cuda_nvls_share_handles(struct ucc_tl_cuda_team *self, int export_handle,
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
ucc_tl_cuda_nvls_import_handle(struct ucc_tl_cuda_team *self, int export_handle,
                               pid_t                         targetPid,
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
        ucc_error("failed to import handle from rank 0");
        return status;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_sync_barrier(struct ucc_tl_cuda_team *self)
{
    ucc_status_t               status;
    ucc_tl_cuda_shm_barrier_t *bar = UCC_TL_CUDA_TEAM_BARRIER(self, 0);

    status = ucc_tl_cuda_shm_barrier_start(UCC_TL_TEAM_RANK(self), bar);
    if (status != UCC_OK) {
        ucc_error("failed to start shm barrier");
        return status;
    }

    while ((status = ucc_tl_cuda_shm_barrier_test(UCC_TL_TEAM_RANK(self),
                                                  bar)) == UCC_INPROGRESS) {
        // Wait for barrier completion
    }

    if (status != UCC_OK) {
        ucc_error("failed to test shm barrier");
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_nvls_init(struct ucc_tl_cuda_team *self,
                                   ucc_base_context_t      *tl_context)
{
    ucc_tl_cuda_lib_t *lib             = ucc_derived_of(tl_context->lib, ucc_tl_cuda_lib_t);
    ucc_tl_cuda_nvls_t *nvls           = &self->nvls;
    const size_t        symmetric_size = lib->cfg.max_concurrent * (lib->cfg.nvls_symmetric_size + NVLS_CONTROL_SIZE);
    size_t              minGran = 0, gran = 0, mcSize = 0;
    int                 export_handle = 0, device = 0;
    pid_t              *shared_pids   = NULL;
    void               *uc_va = NULL, *mc_va = NULL;
    ucc_status_t        status = UCC_OK;
    int                 i = 0;

    CUmemGenericAllocationHandle mcHandle = 0;
    CUmemGenericAllocationHandle memhandle = 0;
    CUmemAllocationHandleType handleType = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;
    CUmulticastObjectProp        mcProp  = {};


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

    // Create multicast object on rank 0
    if (UCC_TL_TEAM_RANK(self) == 0) {
        status = ucc_tl_cuda_nvls_create_multicast_object(&mcProp, &mcHandle,
                                                          &export_handle);
        if (status != UCC_OK) {
            ucc_error("failed to create multicast object");
            goto cleanup;
        }
    }

    // Share handles across ranks
    status = ucc_tl_cuda_nvls_share_handles(self, export_handle, shared_pids);
    if (status != UCC_OK) {
        goto cleanup;
    }

    // Import handle on non-root ranks
    if (UCC_TL_TEAM_RANK(self) != 0) {
        export_handle = self->shared_handles[0];
        status        = ucc_tl_cuda_nvls_import_handle(self, export_handle,
                                                       shared_pids[0], &mcHandle);
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
    ucc_debug("rank %d: added device %d to multicast\n", UCC_TL_TEAM_RANK(self),
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
        // root rank initializes the arrival counter for each task
        ucc_tl_cuda_nvls_control_t control;
        control.arrival_counter = 0;

        for (i = 0; i < lib->cfg.max_concurrent; ++i) {
            void *control_uc =
                PTR_OFFSET((void *)uc_va, i * (lib->cfg.nvls_symmetric_size +
                                               NVLS_CONTROL_SIZE) +
                                              lib->cfg.nvls_symmetric_size);
            CUDA_CHECK(cudaMemcpy(control_uc, &control, sizeof(ucc_tl_cuda_nvls_control_t),
                       cudaMemcpyHostToDevice));
        }
    }

    ucc_free(shared_pids);
    return UCC_OK;

cleanup:
    // Cleanup on error
    if (shared_pids) {
        ucc_free(shared_pids);
    }

    if (UCC_TL_TEAM_RANK(self) == 0) {
        if (nvls->mc_va) {
            CUDADRV_FUNC(cuMemUnmap(nvls->mc_va, nvls->mc_size));
            CUDADRV_FUNC(cuMemAddressFree(nvls->mc_va, nvls->mc_size));
        }
        if (nvls->mc_handle) {
            CUDADRV_FUNC(cuMemRelease(nvls->mc_handle));
        }
    }

    if (memhandle) {
        CUDADRV_FUNC(cuMemRelease(memhandle));
    }

    if (uc_va) {
        CUDADRV_FUNC(cuMemUnmap((CUdeviceptr)uc_va, mcSize));
        CUDADRV_FUNC(cuMemAddressFree((CUdeviceptr)uc_va, mcSize));
    }

    if (mc_va) {
        CUDADRV_FUNC(cuMemUnmap((CUdeviceptr)mc_va, mcSize));
        CUDADRV_FUNC(cuMemAddressFree((CUdeviceptr)mc_va, mcSize));
    }

    return status;
}

ucc_status_t ucc_tl_cuda_nvls_destroy(struct ucc_tl_cuda_team *self,
                                      ucc_base_context_t      *tl_context)
{
    if (UCC_TL_TEAM_RANK(self) == 0) {
        // unbind the multicast object
        CUDADRV_FUNC(cuMulticastUnbind(self->nvls.mc_handle, 0 /* device */,
                                       self->nvls.mc_offset,
                                       self->nvls.mc_size));
        // release the multicast memory handle
        CUDADRV_FUNC(cuMemRelease(self->nvls.mc_memhandle));
        // unmap the multicast memory
        CUDADRV_FUNC(cuMemUnmap(self->nvls.mc_va, self->nvls.mc_size));
        // free the multicast memory address
        CUDADRV_FUNC(cuMemAddressFree(self->nvls.mc_va, self->nvls.mc_size));
        // release the multicast handle
        CUDADRV_FUNC(cuMemRelease(self->nvls.mc_handle));
    }

    if (self->nvls.coll_ids) {
        ucc_free(self->nvls.coll_ids);
    }

    return UCC_OK;
}
