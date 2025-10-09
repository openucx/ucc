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

static ucc_status_t ucc_tl_cuda_nvls_check_support(
    ucc_tl_cuda_team_t *team, int device)
{
    int          mutlicast_supported, fabric_supported;
    ucc_status_t status;

    status = CUDADRV_FUNC(cuDeviceGetAttribute(
        &mutlicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device));
    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to query multicast support on device %d: %d",
            device,
            status);
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuDeviceGetAttribute(
        &fabric_supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        device));
    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to query fabric handle support on device %d: %d",
            device,
            status);
        return UCC_ERR_NOT_SUPPORTED;
    }

    tl_debug(
        UCC_TL_TEAM_LIB(team),
        "MULTICAST_SUPPORTED: %d, HANDLE_TYPE_FABRIC_SUPPORTED: %d\n",
        mutlicast_supported,
        fabric_supported);

    if (!mutlicast_supported) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "multicast not supported on device %d",
            device);
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_get_granularity(
    ucc_tl_cuda_team_t *team, CUmulticastObjectProp *mc_prop, size_t *min_gran,
    size_t *gran)
{
    ucc_status_t status;

    status = CUDADRV_FUNC(cuMulticastGetGranularity(
        min_gran, mc_prop, CU_MULTICAST_GRANULARITY_MINIMUM));
    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to get multicast granularity minimum");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuMulticastGetGranularity(
        gran, mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to get multicast granularity recommended");
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_create_multicast_object(
    ucc_tl_cuda_team_t *team, CUmulticastObjectProp *mc_prop,
    CUmemGenericAllocationHandle *mc_handle, void *export_handle)
{
    ucc_status_t status;

    status = CUDADRV_FUNC(cuMulticastCreate(mc_handle, mc_prop));
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to create multicast object");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuMemExportToShareableHandle(
        export_handle, *mc_handle, mc_prop->handleTypes, 0));
    if (status != UCC_OK) {
        tl_error(UCC_TL_TEAM_LIB(team), "failed to export shareable handle");
        CUDADRV_FUNC(cuMemRelease(*mc_handle));
        return status;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_share_handles_posix(
    ucc_tl_cuda_team_t *team, int export_handle)
{
    ucc_status_t status;

    // If oob_req is NULL, initiate the allgather
    if (team->oob_req == NULL) {
        // Prepare local data (each rank populates its own data)
        if (UCC_TL_TEAM_RANK(team) == 0) {
            team->nvls.share_data[0].handle = export_handle;
            team->nvls.share_data[0].pid    = getpid();
        }

        // Initiate single allgather for both PID and export handle
        status = team->oob.allgather(
            team->nvls.share_data,
            team->nvls.share_data,
            sizeof(ucc_tl_cuda_nvls_share_data_t),
            team->oob.coll_info,
            &team->oob_req);
        if (status != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to initiate allgather for PIDs and handles");
            ucc_free(team->nvls.share_data);
            team->nvls.share_data = NULL;
            return status;
        }
    }

    // Test the request status
    status = team->oob.req_test(team->oob_req);

    // If still in progress, return immediately (non-blocking)
    if (status > 0) {
        return UCC_INPROGRESS;
    }

    // If error occurred
    if (status < 0) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to test allgather for PIDs and handles");
        goto cleanup_on_error;
    }

    // Clean up
    team->oob.req_free(team->oob_req);
    team->oob_req = NULL;

    return UCC_OK;

cleanup_on_error:
    ucc_free(team->nvls.share_data);
    team->nvls.share_data = NULL;
    team->oob.req_free(team->oob_req);
    team->oob_req = NULL;
    return status;
}

static ucc_status_t ucc_tl_cuda_nvls_share_handles_fabric(
    struct ucc_tl_cuda_team *team, CUmemFabricHandle export_handle,
    CUmemFabricHandle *shared_fabric_handles)
{
    ucc_status_t status;

    if (team->oob_req == NULL) {
        // Share the export handle
        if (UCC_TL_TEAM_RANK(team) == 0) {
            shared_fabric_handles[0] = export_handle;
        }
        status = team->oob.allgather(
            &export_handle,
            shared_fabric_handles,
            sizeof(export_handle),
            team->oob.coll_info,
            &team->oob_req);
        if (UCC_OK != status) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to allgather export fabric handle");
            return status;
        }
    }

    status = team->oob.req_test(team->oob_req);
    if (status != UCC_OK) {
        if (status < 0) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to test allgather export fabric handle");
            team->oob.req_free(team->oob_req);
            team->oob_req = NULL;
            return status;
        }
        return status;
    }
    team->oob.req_free(team->oob_req);
    team->oob_req = NULL;
    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_import_handle_posix(
    struct ucc_tl_cuda_team *team, int export_handle, pid_t target_pid,
    CUmemGenericAllocationHandle *mc_handle)
{
    void        *p;
    int          pid_fd, peer_fd;
    ucc_status_t status;

    pid_fd = syscall(SYS_pidfd_open, target_pid, 0);
    if (pid_fd < 0) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to open pidfd for pid %d",
            target_pid);
        return UCC_ERR_NO_RESOURCE;
    }

    peer_fd = syscall(SYS_pidfd_getfd, pid_fd, export_handle, 0);
    if (peer_fd < 0) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to get peer fd: %s (errno=%d)",
            strerror(errno),
            errno);
        close(pid_fd);
        return UCC_ERR_NO_RESOURCE;
    }

    p      = (void *)((uint64_t)peer_fd);
    status = CUDADRV_FUNC(cuMemImportFromShareableHandle(
        mc_handle, p, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    if (close(peer_fd) != 0) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to close peer_fd: %s (errno=%d)",
            strerror(errno),
            errno);
    }
    if (close(pid_fd) != 0) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to close pid_fd: %s (errno=%d)",
            strerror(errno),
            errno);
    }

    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to import POSIX file descriptor handle from rank 0");
        return status;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_import_handle_fabric(
    struct ucc_tl_cuda_team *team, CUmemFabricHandle export_handle,
    CUmemGenericAllocationHandle *mc_handle)
{
    ucc_status_t status = CUDADRV_FUNC(cuMemImportFromShareableHandle(
        mc_handle, &export_handle, CU_MEM_HANDLE_TYPE_FABRIC));

    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to import fabric handle from rank 0");
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_nvls_init(
    struct ucc_tl_cuda_team *team, ucc_base_context_t *tl_context)
{
    ucc_tl_cuda_lib_t *lib = ucc_derived_of(tl_context->lib, ucc_tl_cuda_lib_t);
    ucc_tl_cuda_nvls_t *nvls           = &team->nvls;
    const size_t        symmetric_size = lib->cfg.max_concurrent *
                                  (lib->cfg.nvls_symmetric_size +
                                   NVLS_CONTROL_SIZE);
    size_t                       mc_size    = 0;
    CUdeviceptr                  uc_va      = 0;
    CUdeviceptr                  mc_va      = 0;
    ucc_status_t                 status     = UCC_OK;
    CUmemGenericAllocationHandle mc_handle  = 0;
    CUmemGenericAllocationHandle mem_handle = 0;
    CUmulticastObjectProp        mc_prop    = {};
    CUmemAllocationProp          prop       = {};
    CUmemAllocationHandleType    handle_types;

    switch (team->state) {
    case UCC_TL_CUDA_NVLS_STATE_INIT:
        // Initialize nvls struct to ensure safe cleanup on error
        memset(nvls, 0, sizeof(*nvls));

        nvls->is_multinode = !ucc_team_map_is_single_node(
            team->super.super.params.team, team->super.super.params.map);
        handle_types        = nvls->is_multinode
                                  ? CU_MEM_HANDLE_TYPE_FABRIC
                                  : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        // Initialize multicast properties
        mc_prop.numDevices  = UCC_TL_TEAM_SIZE(team);
        mc_prop.size        = symmetric_size;
        mc_prop.handleTypes = handle_types;
        mc_prop.flags       = 0;

        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "RANK %d: numDevices: %d, size: %zu, handleTypes: %lld, "
            "flags: %lld\n",
            UCC_TL_TEAM_RANK(team),
            mc_prop.numDevices,
            mc_prop.size,
            mc_prop.handleTypes,
            mc_prop.flags);

        // Get current device and check support
        CUDA_CHECK(cudaGetDevice(&nvls->device));
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "RANK %d: device: %d\n",
            UCC_TL_TEAM_RANK(team),
            nvls->device);

        status = ucc_tl_cuda_nvls_check_support(team, nvls->device);
        if (status != UCC_OK) {
            return status;
        }

        // Get granularity requirements
        status = ucc_tl_cuda_nvls_get_granularity(
            team, &mc_prop, &nvls->minGran, &nvls->gran);
        if (status != UCC_OK) {
            return status;
        }

        if (UCC_TL_TEAM_RANK(team) == 0) {
            tl_debug(
                UCC_TL_TEAM_LIB(team),
                "NVLS multicast granularity: gran = %lu, minGran = %lu\n",
                nvls->gran,
                nvls->minGran);
        }

        // Calculate aligned size
        mc_size = ((symmetric_size + nvls->gran - 1) / nvls->gran) * nvls->gran;
        mc_prop.size     = mc_size;
        nvls->mc_size    = mc_size; // Store for later use

        // Allocate buffer for gathering data from all ranks
        nvls->share_data = ucc_malloc(
            sizeof(ucc_tl_cuda_nvls_share_data_t) * UCC_TL_TEAM_SIZE(team),
            "nvls_share_data");
        if (!nvls->share_data) {
            return UCC_ERR_NO_MEMORY;
        }

        // Allocate shared PIDs array
        nvls->shared_pids = ucc_malloc(
            UCC_TL_TEAM_SIZE(team) * sizeof(pid_t), "shared_pids");
        if (!nvls->shared_pids) {
            status = UCC_ERR_NO_MEMORY;
            goto cleanup;
        }

        // Allocate shared fabric handles array
        nvls->shared_fabric_handles = ucc_malloc(
            UCC_TL_TEAM_SIZE(team) * sizeof(CUmemFabricHandle),
            "shared_fabric_handles");
        if (!nvls->shared_fabric_handles) {
            status = UCC_ERR_NO_MEMORY;
            goto cleanup;
        }

        // Create multicast object on rank 0
        if (UCC_TL_TEAM_RANK(team) == 0) {
            if (nvls->is_multinode) {
                status = ucc_tl_cuda_nvls_create_multicast_object(
                    team, &mc_prop, &mc_handle, &nvls->fabric_handle);
            } else {
                status = ucc_tl_cuda_nvls_create_multicast_object(
                    team, &mc_prop, &mc_handle, &nvls->export_handle);
            }
            if (status != UCC_OK) {
                tl_error(
                    UCC_TL_TEAM_LIB(team), "failed to create multicast object");
                goto cleanup;
            }
            nvls->mc_handle = mc_handle;
        }
        team->state = UCC_TL_CUDA_NVLS_STATE_SHARE_HANDLES;
        // fall through
    case UCC_TL_CUDA_NVLS_STATE_SHARE_HANDLES:
        // Share handles across ranks (uses state-stored values)
        if (nvls->is_multinode) {
            status = ucc_tl_cuda_nvls_share_handles_fabric(
                team, nvls->fabric_handle, nvls->shared_fabric_handles);
        } else {
            status = ucc_tl_cuda_nvls_share_handles_posix(
                team, nvls->export_handle);
        }
        if (status == UCC_INPROGRESS) {
            return status;
        }
        if (status != UCC_OK) {
            goto cleanup;
        }
        team->state = UCC_TL_CUDA_NVLS_STATE_IMPORT_HANDLE;
        // fall through
    case UCC_TL_CUDA_NVLS_STATE_IMPORT_HANDLE:
        // Import handle on non-root ranks
        if (UCC_TL_TEAM_RANK(team) != 0) {
            if (nvls->is_multinode) {
                status = ucc_tl_cuda_nvls_import_handle_fabric(
                    team, nvls->shared_fabric_handles[0], &mc_handle);
            } else {
                status = ucc_tl_cuda_nvls_import_handle_posix(
                    team,
                    nvls->share_data[0].handle,
                    nvls->share_data[0].pid,
                    &mc_handle);
            }
            if (status != UCC_OK) {
                goto cleanup;
            }
            nvls->mc_handle = mc_handle;
        }
        team->state = UCC_TL_CUDA_NVLS_STATE_ADD_DEVICE;
        // fall through
    case UCC_TL_CUDA_NVLS_STATE_ADD_DEVICE:
        // Add device to multicast object
        status = CUDADRV_FUNC(
            cuMulticastAddDevice(nvls->mc_handle, nvls->device));
        if (status != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team), "failed to add device to multicast");
            goto cleanup;
        }
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "RANK %d: added device %d to multicast\n",
            UCC_TL_TEAM_RANK(team),
            nvls->device);

        // Allocate physical memory
        prop.type          = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id   = nvls->device;
        prop.requestedHandleTypes =
            nvls->is_multinode ? CU_MEM_HANDLE_TYPE_FABRIC
                               : CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

        mc_size = nvls->mc_size;
        status  = CUDADRV_FUNC(cuMemCreate(&mem_handle, mc_size, &prop, 0));
        if (status != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to create memory allocation for multicast");
            goto cleanup;
        }

        // Set up memory access descriptor
        CUmemAccessDesc accessDesc = {};
        accessDesc.location.type   = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id     = nvls->device;
        accessDesc.flags           = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // Reserve and map unicast virtual address space
        status                     = CUDADRV_FUNC(
            cuMemAddressReserve(&uc_va, mc_size, nvls->minGran, 0U, 0));
        if (status != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to reserve virtual address space");
            goto cleanup;
        }

        status = CUDADRV_FUNC(cuMemMap(uc_va, mc_size, 0, mem_handle, 0));
        if (status != UCC_OK) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to map memory allocation");
            goto cleanup;
        }

        status = CUDADRV_FUNC(cuMemSetAccess(uc_va, mc_size, &accessDesc, 1));
        if (status != UCC_OK) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to set memory access");
            goto cleanup;
        }

        // Bind memory to multicast object
        status = CUDADRV_FUNC(cuMulticastBindAddr(
            nvls->mc_handle, 0 /*mcOffset*/, uc_va, mc_size, 0));
        if (status != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team), "failed to bind memory to multicast");
            goto cleanup;
        }

        // Reserve and map multicast virtual address space
        status = CUDADRV_FUNC(
            cuMemAddressReserve(&mc_va, mc_size, nvls->minGran, 0U, 0));
        if (status != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to reserve multicast virtual address space");
            goto cleanup;
        }

        status = CUDADRV_FUNC(cuMemMap(mc_va, mc_size, 0, nvls->mc_handle, 0));
        if (status != UCC_OK) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to map multicast memory");
            goto cleanup;
        }

        status = CUDADRV_FUNC(cuMemSetAccess(mc_va, mc_size, &accessDesc, 1));
        if (status != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team), "failed to set multicast memory access");
            goto cleanup;
        }

        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "Rank: %d symmetric memory is set: %p [%ld bytes]\n",
            UCC_TL_TEAM_RANK(team),
            (void *)mc_va,
            mc_size);

        // Store the handles for cleanup in team destroy
        nvls->mc_va        = mc_va;
        nvls->uc_va        = uc_va;
        nvls->mc_memhandle = mem_handle;
        nvls->mc_offset    = 0; // mcOffset;
        nvls->coll_ids     = ucc_malloc(
            lib->cfg.max_concurrent * sizeof(size_t), "coll_ids");
        if (!nvls->coll_ids) {
            status = UCC_ERR_NO_MEMORY;
            goto cleanup;
        }
        // Initialize the coll_ids to 0
        memset(nvls->coll_ids, 0, lib->cfg.max_concurrent * sizeof(size_t));

        if (UCC_TL_TEAM_RANK(team) == 0) {
            // root rank zero-initializes the control region for each task slot
            CUDA_CHECK(cudaMemset2D(
                (void *)(uc_va + lib->cfg.nvls_symmetric_size),
                lib->cfg.nvls_symmetric_size + NVLS_CONTROL_SIZE,
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
        if (CUDADRV_FUNC(cuMemUnmap(mc_va, mc_size)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team), "failed to unmap mc_va during cleanup");
        }
        if (CUDADRV_FUNC(cuMemAddressFree(mc_va, mc_size)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team), "failed to free mc_va during cleanup");
        }
    }

    // Unmap and free unicast VA if it was reserved/mapped
    if (uc_va != 0) {
        if (CUDADRV_FUNC(cuMemUnmap(uc_va, mc_size)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team), "failed to unmap uc_va during cleanup");
        }
        if (CUDADRV_FUNC(cuMemAddressFree(uc_va, mc_size)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team), "failed to free uc_va during cleanup");
        }
    }

    // Release memory handle if it was created
    if (mem_handle != 0) {
        if (CUDADRV_FUNC(cuMemRelease(mem_handle)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to release mem_handle during cleanup");
        }
    }

    // Release multicast handle if it was created or imported
    if (nvls->mc_handle != 0) {
        if (CUDADRV_FUNC(cuMemRelease(nvls->mc_handle)) != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
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
    struct ucc_tl_cuda_team *team, ucc_base_context_t *tl_context)
{
    int device = 0;

    // Get the actual device used during init
    if (CUDA_FUNC(cudaGetDevice(&device)) != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to get current device during cleanup");
        device = 0; // fallback to device 0
    }

    // Rank 0: unbind the multicast object
    if (UCC_TL_TEAM_RANK(team) == 0 && team->nvls.mc_handle) {
        CUDADRV_FUNC(cuMulticastUnbind(
            team->nvls.mc_handle,
            device,
            team->nvls.mc_offset,
            team->nvls.mc_size));
    }

    // ALL ranks: release their multicast handle (created or imported)
    if (team->nvls.mc_handle) {
        CUDADRV_FUNC(cuMemRelease(team->nvls.mc_handle));
        team->nvls.mc_handle = 0;
    }

    // ALL ranks: clean up their mapped memory
    if (team->nvls.mc_va) {
        CUDADRV_FUNC(cuMemUnmap(team->nvls.mc_va, team->nvls.mc_size));
        CUDADRV_FUNC(cuMemAddressFree(team->nvls.mc_va, team->nvls.mc_size));
    }
    if (team->nvls.uc_va) {
        CUDADRV_FUNC(cuMemUnmap(team->nvls.uc_va, team->nvls.mc_size));
        CUDADRV_FUNC(cuMemAddressFree(team->nvls.uc_va, team->nvls.mc_size));
    }
    // ALL ranks: release their memory handle
    if (team->nvls.mc_memhandle) {
        CUDADRV_FUNC(cuMemRelease(team->nvls.mc_memhandle));
    }
    // ALL ranks: free coll_ids
    if (team->nvls.coll_ids) {
        ucc_free(team->nvls.coll_ids);
        team->nvls.coll_ids = NULL;
    }
    if (team->nvls.share_data) {
        ucc_free(team->nvls.share_data);
        team->nvls.share_data = NULL;
    }
    return UCC_OK;
}
