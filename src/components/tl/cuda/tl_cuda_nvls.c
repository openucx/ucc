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
#include "utils/ucc_math.h"

#include <sys/syscall.h> // for pidfd_open and pidfd_getfd
#include <sys/prctl.h>   // for prctl()
#include <unistd.h>      // for close()

ucc_status_t ucc_tl_cuda_nvls_check_support(
    ucc_tl_cuda_lib_t *lib, int device, int is_multinode)
{
    int          multicast_supported, fabric_supported;
    ucc_status_t status;

    status = CUDADRV_FUNC(cuDeviceGetAttribute(
        &multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED, device));
    if (status != UCC_OK) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuDeviceGetAttribute(
        &fabric_supported,
        CU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
        device));
    if (status != UCC_OK) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    tl_debug(
        lib,
        "MULTICAST_SUPPORTED: %d, HANDLE_TYPE_FABRIC_SUPPORTED: %d\n",
        multicast_supported,
        fabric_supported);

    if (!multicast_supported) {
        tl_debug(lib, "multicast not supported on device %d", device);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (is_multinode && !fabric_supported) {
        tl_debug(lib, "fabric handle not supported on device %d", device);
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_get_granularity(
    CUmulticastObjectProp *mc_prop, size_t *min_gran, size_t *gran)
{
    ucc_status_t status;

    status = CUDADRV_FUNC(cuMulticastGetGranularity(
        min_gran, mc_prop, CU_MULTICAST_GRANULARITY_MINIMUM));
    if (status != UCC_OK) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuMulticastGetGranularity(
        gran, mc_prop, CU_MULTICAST_GRANULARITY_RECOMMENDED));
    if (status != UCC_OK) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_create_multicast_object_posix(
    CUmulticastObjectProp *mc_prop, CUmemGenericAllocationHandle *mc_handle,
    int *export_handle)
{
    ucc_status_t status;

    status = CUDADRV_FUNC(cuMulticastCreate(mc_handle, mc_prop));
    if (status != UCC_OK) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    // For POSIX file descriptors, CUDA API writes the fd to the memory pointed to by the first arg
    // The API expects void* but will write an int-sized value
    status = CUDADRV_FUNC(cuMemExportToShareableHandle(
        (void *)export_handle, *mc_handle, mc_prop->handleTypes, 0));
    if (status != UCC_OK) {
        CUDADRV_FUNC(cuMemRelease(*mc_handle));
        return status;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_create_multicast_object_fabric(
    CUmulticastObjectProp *mc_prop, CUmemGenericAllocationHandle *mc_handle,
    CUmemFabricHandle *export_handle)
{
    ucc_status_t status;

    status = CUDADRV_FUNC(cuMulticastCreate(mc_handle, mc_prop));
    if (status != UCC_OK) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = CUDADRV_FUNC(cuMemExportToShareableHandle(
        export_handle, *mc_handle, mc_prop->handleTypes, 0));
    if (status != UCC_OK) {
        CUDADRV_FUNC(cuMemRelease(*mc_handle));
        return status;
    }

    return UCC_OK;
}

static ucc_status_t ucc_tl_cuda_nvls_share_handles(
    ucc_tl_cuda_team_t *team, ucc_tl_cuda_nvls_handle_t *local_handle)
{
    ucc_status_t status;

    // If oob_req is NULL, initiate the allgather
    if (team->oob_req == NULL) {
        // Prepare local data based on handle type
        // In both cases, only rank 0 creates and exports the handle
        if (UCC_TL_TEAM_RANK(team) == 0) {
            // Copy the local handle to the share buffer
            team->nvls.share_data[0] = *local_handle;
        }

        // Initiate single allgather for handle data
        status = team->oob.allgather(
            team->nvls.share_data,
            team->nvls.share_data,
            sizeof(ucc_tl_cuda_nvls_handle_t),
            team->oob.coll_info,
            &team->oob_req);
        if (status != UCC_OK) {
            tl_error(
                UCC_TL_TEAM_LIB(team),
                "failed to initiate allgather for handles (type: %s)",
                local_handle->type == UCC_TL_CUDA_NVLS_HANDLE_TYPE_FABRIC
                    ? "fabric"
                    : "posix");
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
            "failed to test allgather for handles (type: %s)",
            local_handle->type == UCC_TL_CUDA_NVLS_HANDLE_TYPE_FABRIC
                ? "fabric"
                : "posix");
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

static ucc_status_t ucc_tl_cuda_nvls_import_handle_posix(
    struct ucc_tl_cuda_team *team, ucc_tl_cuda_nvls_handle_t *share_data,
    CUmemGenericAllocationHandle *mc_handle)
{
    void        *os_handle;
    int          pid_fd, peer_fd;
    ucc_status_t status;
    int          export_handle;
    pid_t        target_pid;

    if (share_data->type != UCC_TL_CUDA_NVLS_HANDLE_TYPE_POSIX) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "expected POSIX handle type but got fabric handle type");
        return UCC_ERR_INVALID_PARAM;
    }

    export_handle = share_data->data.posix.handle;
    target_pid    = share_data->data.posix.pid;

    pid_fd        = syscall(SYS_pidfd_open, target_pid, 0);
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

    os_handle = (void *)((uint64_t)peer_fd);
    status    = CUDADRV_FUNC(cuMemImportFromShareableHandle(
        mc_handle, os_handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

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
    struct ucc_tl_cuda_team *team, ucc_tl_cuda_nvls_handle_t *share_data,
    CUmemGenericAllocationHandle *mc_handle)
{
    ucc_status_t status;

    if (share_data->type != UCC_TL_CUDA_NVLS_HANDLE_TYPE_FABRIC) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "expected fabric handle type but got POSIX handle type");
        return UCC_ERR_INVALID_PARAM;
    }

    status = CUDADRV_FUNC(cuMemImportFromShareableHandle(
        mc_handle, &share_data->data.fabric, CU_MEM_HANDLE_TYPE_FABRIC));

    if (status != UCC_OK) {
        tl_error(
            UCC_TL_TEAM_LIB(team),
            "failed to import fabric handle from rank 0. status (%d) %s",
            status,
            ucc_status_string(status));
        return status;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_cuda_nvls_init(
    struct ucc_tl_cuda_team *team, struct ucc_base_context *tl_context)
{
    ucc_tl_cuda_context_t *ctx = ucc_derived_of(
        tl_context, ucc_tl_cuda_context_t);
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
    CUmemAccessDesc              accessDesc = {};
    CUmemAllocationHandleType    handle_types;

    switch (team->state) {
    case UCC_TL_CUDA_NVLS_STATE_INIT:
        // Initialize nvls struct to ensure safe cleanup on error
        memset(nvls, 0, sizeof(*nvls));
        // Get current device from context
        nvls->device       = ctx->device;
        nvls->is_multinode = !ucc_team_map_is_single_node(
            team->super.super.params.team, team->super.super.params.map);

        status = ucc_tl_cuda_nvls_check_support(
            lib, nvls->device, nvls->is_multinode);
        if (status != UCC_OK) {
            return status;
        }

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

        // Get granularity requirements
        status = ucc_tl_cuda_nvls_get_granularity(
            &mc_prop, &nvls->minGran, &nvls->gran);
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
        mc_size          = ucc_align_up(symmetric_size, nvls->gran);
        mc_prop.size     = mc_size;
        nvls->mc_size    = mc_size; // Store for later use

        // Allocate buffer for gathering data from all ranks
        nvls->share_data = (ucc_tl_cuda_nvls_handle_t *)ucc_malloc(
            sizeof(ucc_tl_cuda_nvls_handle_t) * UCC_TL_TEAM_SIZE(team),
            "nvls_share_data");
        if (!nvls->share_data) {
            return UCC_ERR_NO_MEMORY;
        }

        // Create multicast object on rank 0
        if (UCC_TL_TEAM_RANK(team) == 0) {
            // Initialize local handle type
            nvls->local_handle.type = nvls->is_multinode
                                          ? UCC_TL_CUDA_NVLS_HANDLE_TYPE_FABRIC
                                          : UCC_TL_CUDA_NVLS_HANDLE_TYPE_POSIX;

            if (nvls->is_multinode) {
                status = ucc_tl_cuda_nvls_create_multicast_object_fabric(
                    &mc_prop, &mc_handle, &nvls->local_handle.data.fabric);
            } else {
                status = ucc_tl_cuda_nvls_create_multicast_object_posix(
                    &mc_prop,
                    &mc_handle,
                    &nvls->local_handle.data.posix.handle);
            }
            if (status != UCC_OK) {
                tl_error(
                    UCC_TL_TEAM_LIB(team),
                    "failed to create multicast object. status (%d) %s",
                    status,
                    ucc_status_string(status));
                // goto cleanup;
                // Store the error status to the caller
                // We need to share invalid handle to unblock peers
                // we will propagate the error status to the caller later
                nvls->status_supported = UCC_ERR_NOT_SUPPORTED;
            } else {
                nvls->status_supported = UCC_OK;
            }
            // Store PID for POSIX handles
            if (!nvls->is_multinode) {
                nvls->local_handle.data.posix.pid = getpid();
                // Allow peer processes to use pidfd_getfd on this process
                // A more aggressive solution would be to modify Yama ptrace policy by
                // running `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope` or to run
                // the docker container with `--cap-add=SYS_PTRACE` and `--sysctl
                // kernel.yama.ptrace_scope=0`.
                if (prctl(PR_SET_PTRACER, PR_SET_PTRACER_ANY) != 0) {
                    tl_warn(
                        UCC_TL_TEAM_LIB(team),
                        "failed to set PR_SET_PTRACER: %s (errno=%d). "
                        "This may cause pidfd_getfd to fail on peer processes. "
                        "Consider adjusting Yama ptrace_scope settings: `echo "
                        "0 | sudo tee /proc/sys/kernel/yama/ptrace_scope` or "
                        "to run "
                        "the docker container with `--cap-add=SYS_PTRACE` and "
                        "`--sysctl kernel.yama.ptrace_scope=0`",
                        strerror(errno),
                        errno);
                }
            }
            nvls->mc_handle = mc_handle;
        }
        team->state = UCC_TL_CUDA_NVLS_STATE_SHARE_HANDLES;
        // fall through
    case UCC_TL_CUDA_NVLS_STATE_SHARE_HANDLES:
        // Share handles across ranks using unified function
        status = ucc_tl_cuda_nvls_share_handles(team, &nvls->local_handle);
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
                    team, &nvls->share_data[0], &mc_handle);
            } else {
                status = ucc_tl_cuda_nvls_import_handle_posix(
                    team, &nvls->share_data[0], &mc_handle);
            }
            if (status != UCC_OK) {
                goto cleanup;
            }
            nvls->mc_handle = mc_handle;
        }
        if (nvls->status_supported != UCC_OK) {
            // Propagate the supported status to the caller
            status = nvls->status_supported;
            goto cleanup;
        }
        team->state = UCC_TL_CUDA_NVLS_STATE_ADD_DEVICE;
        // fall through
    case UCC_TL_CUDA_NVLS_STATE_ADD_DEVICE:
    {
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
        accessDesc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        accessDesc.location.id   = nvls->device;
        accessDesc.flags         = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

        // Reserve and map unicast virtual address space
        status                   = CUDADRV_FUNC(
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
        nvls->coll_ids     = (size_t *)ucc_malloc(
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

        break;
    }
    default:
        break;
    }

    return UCC_OK;

cleanup:
    // Cleanup on error - free state-stored temporary allocations
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

ucc_status_t ucc_tl_cuda_nvls_destroy(ucc_tl_cuda_team_t *team)
{
    int device = team->nvls.device;

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
