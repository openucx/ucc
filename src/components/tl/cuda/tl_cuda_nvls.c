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

    res = cuDeviceGetAttribute(
        &supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device);
    if (res != CUDA_SUCCESS) {
        ucc_error(
            "failed to query multicast support on device %d: %d", device, res);
        return UCC_ERR_NOT_SUPPORTED;
    }

    res = cuDeviceGetAttribute(
        &fabric_supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        device);
    if (res != CUDA_SUCCESS) {
        ucc_error(
            "failed to query fabric handle support on device %d: %d",
            device,
            res);
        return UCC_ERR_NOT_SUPPORTED;
    }

    ucc_debug(
        "MULTICAST_SUPPORTED: %d, HANDLE_TYPE_FABRIC_SUPPORTED: %d\n",
        supported,
        fabric_supported);

    if (!supported) {
        ucc_error("multicast not supported on device %d", device);
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_get_granularity(
    CUmulticastObjectProp *mcProp, size_t *minGran, size_t *gran)
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

static ucc_status_t ucc_tl_cuda_nvls_create_multicast_object(
    CUmulticastObjectProp *mcProp, CUmemGenericAllocationHandle *mcHandle,
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

static ucc_status_t ucc_tl_cuda_nvls_share_handles_posix(
    struct ucc_tl_cuda_team *self, int export_handle)
{
    ucc_status_t status;

    // If oob_req is NULL, initiate the allgather
    if (self->oob_req == NULL) {
        // Prepare local data (each rank populates its own data)
        if (UCC_TL_TEAM_RANK(self) == 0) {
            self->nvls.share_data[0].handle = export_handle;
            self->nvls.share_data[0].pid    = getpid();
        }

        // Initiate single allgather for both PID and export handle
        status = self->oob.allgather(
            self->nvls.share_data,
            self->nvls.share_data,
            sizeof(ucc_tl_cuda_nvls_share_data_t),
            self->oob.coll_info,
            &self->oob_req);
        if (status != UCC_OK) {
            ucc_error("failed to initiate allgather for PIDs and handles");
            ucc_free(self->nvls.share_data);
            self->nvls.share_data = NULL;
            return status;
        }
    }

    // Test the request status
    status = self->oob.req_test(self->oob_req);

    // If still in progress, return immediately (non-blocking)
    if (status > 0) {
        return UCC_INPROGRESS;
    }

    // If error occurred
    if (status < 0) {
        tl_error(
            UCC_TL_TEAM_LIB(self),
            "failed to test allgather for PIDs and handles");
        goto cleanup_on_error;
    }

    // Clean up
    self->oob.req_free(self->oob_req);
    self->oob_req = NULL;

    return UCC_OK;

cleanup_on_error:
    ucc_free(self->nvls.share_data);
    self->nvls.share_data = NULL;
    self->oob.req_free(self->oob_req);
    self->oob_req = NULL;
    return status;
}

static ucc_status_t ucc_tl_cuda_nvls_share_handles_fabric(
    struct ucc_tl_cuda_team *self, CUmemFabricHandle export_handle,
    CUmemFabricHandle *shared_fabric_handles)
{
    ucc_status_t status;

    if (self->oob_req == NULL) {
        // Share the export handle
        if (UCC_TL_TEAM_RANK(self) == 0) {
            shared_fabric_handles[0] = export_handle;
        }
        status = self->oob.allgather(
            &export_handle,
            shared_fabric_handles,
            sizeof(export_handle),
            self->oob.coll_info,
            &self->oob_req);
        if (UCC_OK != status) {
            tl_error(
                UCC_TL_TEAM_LIB(self),
                "failed to allgather export fabric handle");
            return status;
        }
    }

    status = self->oob.req_test(self->oob_req);
    if (status != UCC_OK) {
        if (status < 0) {
            tl_error(
                UCC_TL_TEAM_LIB(self),
                "failed to test allgather export fabric handle");
            self->oob.req_free(self->oob_req);
            self->oob_req = NULL;
            return status;
        }
        return status;
    }
    self->oob.req_free(self->oob_req);
    self->oob_req = NULL;
    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_import_handle_posix(
    struct ucc_tl_cuda_team *self, int export_handle, pid_t targetPid,
    CUmemGenericAllocationHandle *mcHandle)
{
    void        *p;
    int          pidFd, peerFd;
    ucc_status_t status;

    pidFd = syscall(SYS_pidfd_open, targetPid, 0);
    if (pidFd < 0) {
        tl_error(
            UCC_TL_TEAM_LIB(self),
            "failed to open pidfd for pid %d",
            targetPid);
        return UCC_ERR_NO_RESOURCE;
    }

    peerFd = syscall(SYS_pidfd_getfd, pidFd, export_handle, 0);
    if (peerFd < 0) {
        tl_error(
            UCC_TL_TEAM_LIB(self),
            "failed to get peer fd: %s (errno=%d)",
            strerror(errno),
            errno);
        close(pidFd);
        return UCC_ERR_NO_RESOURCE;
    }

    p      = (void *)((uint64_t)peerFd);
    status = CUDADRV_FUNC(cuMemImportFromShareableHandle(
        mcHandle, p, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    if (close(peerFd) != 0) {
        tl_error(
            UCC_TL_TEAM_LIB(self),
            "failed to close peerFd: %s (errno=%d)",
            strerror(errno),
            errno);
    }
    if (close(pidFd) != 0) {
        tl_error(
            UCC_TL_TEAM_LIB(self),
            "failed to close pidFd: %s (errno=%d)",
            strerror(errno),
            errno);
    }

    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(self),
            "failed to import POSIX file descriptor handle from rank 0");
        return status;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_import_handle_fabric(
    struct ucc_tl_cuda_team *self, CUmemFabricHandle export_handle,
    CUmemGenericAllocationHandle *mcHandle)
{
    ucc_status_t status = CUDADRV_FUNC(cuMemImportFromShareableHandle(
        mcHandle, &export_handle, CU_MEM_HANDLE_TYPE_FABRIC));

    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(self),
            "failed to import fabric handle from rank 0");
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_nvls_init(
    struct ucc_tl_cuda_team *self, ucc_base_context_t *tl_context)
{
    ucc_tl_cuda_lib_t *lib = ucc_derived_of(tl_context->lib, ucc_tl_cuda_lib_t);
    ucc_tl_cuda_nvls_t *nvls           = &self->nvls;
    const size_t        symmetric_size = lib->cfg.max_concurrent *
                                  (lib->cfg.nvls_symmetric_size +
                                   NVLS_CONTROL_SIZE);
    size_t                       mcSize    = 0;
    CUdeviceptr                  uc_va     = 0;
    CUdeviceptr                  mc_va     = 0;
    ucc_status_t                 status    = UCC_OK;
    CUmemGenericAllocationHandle mcHandle  = 0;
    CUmemGenericAllocationHandle memhandle = 0;
    CUmulticastObjectProp        mcProp    = {};
    CUmemAllocationHandleType    handleType;

    switch (self->state) {
    case UCC_TL_CUDA_NVLS_STATE_INIT:
        // Initialize nvls struct to ensure safe cleanup on error
        memset(nvls, 0, sizeof(*nvls));

        nvls->is_multinode = !ucc_team_map_is_single_node(
            self->super.super.params.team, self->super.super.params.map);
        handleType         = nvls->is_multinode
                                 ? CU_MEM_HANDLE_TYPE_FABRIC
                                 : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        // Initialize multicast properties
        mcProp.numDevices  = UCC_TL_TEAM_SIZE(self);
        mcProp.size        = symmetric_size;
        mcProp.handleTypes = handleType;
        mcProp.flags       = 0;

        ucc_debug(
            "RANK %d: numDevices: %d, size: %zu, handleTypes: %lld, "
            "flags: %lld\n",
            UCC_TL_TEAM_RANK(self),
            mcProp.numDevices,
            mcProp.size,
            mcProp.handleTypes,
            mcProp.flags);

        // Get current device and check support
        CUDA_CHECK(cudaGetDevice(&nvls->device));
        ucc_debug(
            "RANK %d: device: %d\n", UCC_TL_TEAM_RANK(self), nvls->device);

        status = ucc_tl_cuda_nvls_check_support(nvls->device);
        if (status != UCC_OK) {
            return status;
        }

        // Get granularity requirements
        status = ucc_tl_cuda_nvls_get_granularity(
            &mcProp, &nvls->minGran, &nvls->gran);
        if (status != UCC_OK) {
            return status;
        }

        if (UCC_TL_TEAM_RANK(self) == 0) {
            ucc_debug(
                "NVLS multicast granularity: gran = %lu, minGran = %lu\n",
                nvls->gran,
                nvls->minGran);
        }

        // Calculate aligned size
        mcSize = ((symmetric_size + nvls->gran - 1) / nvls->gran) * nvls->gran;
        mcProp.size      = mcSize;
        nvls->mc_size    = mcSize; // Store for later use

        // Allocate buffer for gathering data from all ranks
        nvls->share_data = ucc_malloc(
            sizeof(ucc_tl_cuda_nvls_share_data_t) * UCC_TL_TEAM_SIZE(self),
            "nvls_share_data");
        if (!nvls->share_data) {
            return UCC_ERR_NO_MEMORY;
        }

        // Allocate shared PIDs array
        nvls->shared_pids = ucc_malloc(
            UCC_TL_TEAM_SIZE(self) * sizeof(pid_t), "shared_pids");
        if (!nvls->shared_pids) {
            status = UCC_ERR_NO_MEMORY;
            goto cleanup;
        }

        // Allocate shared fabric handles array
        nvls->shared_fabric_handles = ucc_malloc(
            UCC_TL_TEAM_SIZE(self) * sizeof(CUmemFabricHandle),
            "shared_fabric_handles");
        if (!nvls->shared_fabric_handles) {
            status = UCC_ERR_NO_MEMORY;
            goto cleanup;
        }

        // Create multicast object on rank 0
        if (UCC_TL_TEAM_RANK(self) == 0) {
            if (nvls->is_multinode) {
                status = ucc_tl_cuda_nvls_create_multicast_object(
                    &mcProp, &mcHandle, &nvls->fabric_handle);
            } else {
                status = ucc_tl_cuda_nvls_create_multicast_object(
                    &mcProp, &mcHandle, &nvls->export_handle);
            }
            if (status != UCC_OK) {
                ucc_error("failed to create multicast object");
                goto cleanup;
            }
            nvls->mc_handle = mcHandle;
        }
        self->state = UCC_TL_CUDA_NVLS_STATE_SHARE_HANDLES;
        // fall through
    case UCC_TL_CUDA_NVLS_STATE_SHARE_HANDLES:
        // Share handles across ranks (uses state-stored values)
        if (nvls->is_multinode) {
            status = ucc_tl_cuda_nvls_share_handles_fabric(
                self, nvls->fabric_handle, nvls->shared_fabric_handles);
        } else {
            status = ucc_tl_cuda_nvls_share_handles_posix(
                self, nvls->export_handle);
        }
        if (status == UCC_INPROGRESS) {
            return status;
        }
        if (status != UCC_OK) {
            goto cleanup;
        }
        self->state = UCC_TL_CUDA_NVLS_STATE_IMPORT_HANDLE;
        // fall through
    case UCC_TL_CUDA_NVLS_STATE_IMPORT_HANDLE:
        // Import handle on non-root ranks
        if (UCC_TL_TEAM_RANK(self) != 0) {
            if (nvls->is_multinode) {
                status = ucc_tl_cuda_nvls_import_handle_fabric(
                    self, nvls->shared_fabric_handles[0], &mcHandle);
            } else {
                status = ucc_tl_cuda_nvls_import_handle_posix(
                    self,
                    nvls->share_data[0].handle,
                    nvls->share_data[0].pid,
                    &mcHandle);
            }
            if (status != UCC_OK) {
                goto cleanup;
            }
            nvls->mc_handle = mcHandle;
        }
        self->state = UCC_TL_CUDA_NVLS_STATE_ADD_DEVICE;
        // fall through
    case UCC_TL_CUDA_NVLS_STATE_ADD_DEVICE:
        // Add device to multicast object
        status = CUDADRV_FUNC(
            cuMulticastAddDevice(nvls->mc_handle, nvls->device));
        if (status != UCC_OK) {
            ucc_error("failed to add device to multicast");
            goto cleanup;
        }
        ucc_debug(
            "RANK %d: added device %d to multicast\n",
            UCC_TL_TEAM_RANK(self),
            nvls->device);

        // Allocate physical memory
        CUmemAllocationProp prop = {};
        prop.type                = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type       = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id         = nvls->device;
        prop.requestedHandleTypes =
            nvls->is_multinode ? CU_MEM_HANDLE_TYPE_FABRIC
                               : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        mcSize = nvls->mc_size;
        status = CUDADRV_FUNC(cuMemCreate(&memhandle, mcSize, &prop, 0));
        if (status != UCC_OK) {
            ucc_error("failed to create memory allocation for multicast");
            goto cleanup;
        }

        // Set up memory access descriptor
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id     = nvls->device;
        accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // Reserve and map unicast virtual address space
        status                     = CUDADRV_FUNC(
            cuMemAddressReserve(&uc_va, mcSize, nvls->minGran, 0U, 0));
        if (status != UCC_OK) {
            ucc_error("failed to reserve virtual address space");
            goto cleanup;
        }

        status = CUDADRV_FUNC(cuMemMap(uc_va, mcSize, 0, memhandle, 0));
        if (status != UCC_OK) {
            ucc_error("failed to map memory allocation");
            goto cleanup;
        }

        status = CUDADRV_FUNC(cuMemSetAccess(uc_va, mcSize, &accessDesc, 1));
        if (status != UCC_OK) {
            ucc_error("failed to set memory access");
            goto cleanup;
        }

        // Bind memory to multicast object
        status = CUDADRV_FUNC(cuMulticastBindAddr(
            nvls->mc_handle, 0 /*mcOffset*/, uc_va, mcSize, 0));
        if (status != UCC_OK) {
            ucc_error("failed to bind memory to multicast");
            goto cleanup;
        }

        // Reserve and map multicast virtual address space
        status = CUDADRV_FUNC(
            cuMemAddressReserve(&mc_va, mcSize, nvls->minGran, 0U, 0));
        if (status != UCC_OK) {
            ucc_error("failed to reserve multicast virtual address space");
            goto cleanup;
        }

        status = CUDADRV_FUNC(cuMemMap(mc_va, mcSize, 0, nvls->mc_handle, 0));
        if (status != UCC_OK) {
            ucc_error("failed to map multicast memory");
            goto cleanup;
        }

        status = CUDADRV_FUNC(cuMemSetAccess(mc_va, mcSize, &accessDesc, 1));
        if (status != UCC_OK) {
            ucc_error("failed to set multicast memory access");
            goto cleanup;
        }

        ucc_debug(
            "Rank: %d symmetric memory is set: %p [%ld bytes]\n",
            UCC_TL_TEAM_RANK(self),
            (void *)mc_va,
            mcSize);

        // Store the handles for cleanup in team destroy
        nvls->mc_va        = mc_va;
        nvls->uc_va        = uc_va;
        nvls->mc_memhandle = memhandle;
        nvls->mc_offset    = 0; // mcOffset;
        nvls->coll_ids     = ucc_malloc(
            lib->cfg.max_concurrent * sizeof(size_t), "coll_ids");
        if (!nvls->coll_ids) {
            status = UCC_ERR_NO_MEMORY;
            goto cleanup;
        }
        // Initialize the coll_ids to 0
        memset(nvls->coll_ids, 0, lib->cfg.max_concurrent * sizeof(size_t));

        if (UCC_TL_TEAM_RANK(self) == 0) {
            // root rank zero-initializes the control region for each task slot
            size_t stride = lib->cfg.nvls_symmetric_size + NVLS_CONTROL_SIZE;
            void  *control_uc0 = (void *)(uc_va + lib->cfg.nvls_symmetric_size);
            CUDA_CHECK(cudaMemset2D(
                control_uc0,
                stride,
                0,
                NVLS_CONTROL_SIZE,
                lib->cfg.max_concurrent));
        }

        // Free state-stored temporary buffers
        ucc_free(nvls->shared_pids);
        nvls->shared_pids = NULL;
        ucc_free(nvls->shared_fabric_handles);
        nvls->shared_fabric_handles = NULL;
        break;
    default:
        break;
    }

    return UCC_OK;

cleanup:
    // Cleanup on error - free state-stored temporary allocations
    if (nvls->shared_pids) {
        ucc_free(nvls->shared_pids);
        nvls->shared_pids = NULL;
    }
    if (nvls->shared_fabric_handles) {
        ucc_free(nvls->shared_fabric_handles);
        nvls->shared_fabric_handles = NULL;
    }
    if (nvls->share_data) {
        ucc_free(nvls->share_data);
        nvls->share_data = NULL;
    }

    // Clean up CUDA resources - check local variables for partial allocations
    // Unmap and free multicast VA if it was reserved/mapped
    if (mc_va != 0) {
        if (CUDADRV_FUNC(cuMemUnmap(mc_va, mcSize)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(self), "failed to unmap mc_va during cleanup");
        }
        if (CUDADRV_FUNC(cuMemAddressFree(mc_va, mcSize)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(self), "failed to free mc_va during cleanup");
        }
    }

    // Unmap and free unicast VA if it was reserved/mapped
    if (uc_va != 0) {
        if (CUDADRV_FUNC(cuMemUnmap(uc_va, mcSize)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(self), "failed to unmap uc_va during cleanup");
        }
        if (CUDADRV_FUNC(cuMemAddressFree(uc_va, mcSize)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(self), "failed to free uc_va during cleanup");
        }
    }

    // Release memory handle if it was created
    if (memhandle != 0) {
        if (CUDADRV_FUNC(cuMemRelease(memhandle)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(self),
                "failed to release memhandle during cleanup");
        }
    }

    // Release multicast handle if it was created or imported
    if (nvls->mc_handle != 0) {
        if (CUDADRV_FUNC(cuMemRelease(nvls->mc_handle)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(self),
                "failed to release mc_handle during cleanup");
        }
        nvls->mc_handle = 0;
    }

    // Free coll_ids if allocated
    if (nvls->coll_ids) {
        ucc_free(nvls->coll_ids);
        nvls->coll_ids = NULL;
    }

    return status;
}

ucc_status_t ucc_tl_cuda_nvls_destroy(
    struct ucc_tl_cuda_team *self, ucc_base_context_t *tl_context)
{
    int device = 0;

    // Get the actual device used during init
    if (cudaGetDevice(&device) != cudaSuccess) {
        tl_error(
            UCC_TL_TEAM_LIB(self),
            "failed to get current device during cleanup");
        device = 0; // fallback to device 0
    }

    // Rank 0: unbind the multicast object
    if (UCC_TL_TEAM_RANK(self) == 0 && self->nvls.mc_handle) {
        CUDADRV_FUNC(cuMulticastUnbind(
            self->nvls.mc_handle,
            device,
            self->nvls.mc_offset,
            self->nvls.mc_size));
    }

    // ALL ranks: release their multicast handle (created or imported)
    if (self->nvls.mc_handle) {
        CUDADRV_FUNC(cuMemRelease(self->nvls.mc_handle));
        self->nvls.mc_handle = 0;
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
    if (self->nvls.share_data) {
        ucc_free(self->nvls.share_data);
        self->nvls.share_data = NULL;
    }
    return UCC_OK;
}
