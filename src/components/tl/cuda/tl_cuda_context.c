/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"
#include <tl_cuda_topo.h>
#include <cuda_runtime.h>
#include <cuda.h>

/**
 * Initialize CUDA transport layer context
 *
 * This function initializes a CUDA TL context which requires an active CUDA context.
 * It sets up memory pools for CUDA tasks and initializes the topology information.
 *
 * @param [in]  params      Base context initialization parameters
 * @param [in]  config      Configuration for CUDA context
 *
 * @return UCC_OK on success or error code on failure
 */
UCC_CLASS_INIT_FUNC(ucc_tl_cuda_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_cuda_context_config_t *tl_cuda_config =
        ucc_derived_of(config, ucc_tl_cuda_context_config_t);
    ucc_status_t       status;
    ucc_tl_cuda_lib_t *lib;
    int                num_devices;
    cudaError_t        cuda_st;
    CUcontext          cu_ctx;
    CUresult           cu_st;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_cuda_config->super,
                              params->context);
    lib = ucc_derived_of(self->super.super.lib, ucc_tl_cuda_lib_t);
    memcpy(&self->cfg, tl_cuda_config, sizeof(*tl_cuda_config));

    cuda_st = cudaGetDeviceCount(&num_devices);
    if (cuda_st != cudaSuccess) {
        tl_debug(self->super.super.lib,
                 "failed to get number of GPU devices: %d (%s)", cuda_st,
                 cudaGetErrorName(cuda_st));
        return UCC_ERR_NO_RESOURCE;
    } else if (num_devices == 0) {
        tl_debug(self->super.super.lib, "no GPU devices found");
        return UCC_ERR_NO_RESOURCE;
    }

    cu_st = cuCtxGetCurrent(&cu_ctx);
    if (cu_ctx == NULL || cu_st != CUDA_SUCCESS) {
        tl_debug(self->super.super.lib,
                 "cannot create CUDA TL context without active CUDA context");
        return UCC_ERR_NO_RESOURCE;
    }

    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_cuda_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_coll_task_mpool_ops, params->thread_mode,
                            "tl_cuda_req_mp");
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_cuda_req mpool");
        return status;
    }

    CUDA_CHECK_GOTO(cudaGetDevice(&self->device), free_mpool, status);

    /* Handle CUDA topology initialization based on caching configuration */
    if (lib->cfg.topo_cache_enable && lib->topo != NULL) {
        /* If topology caching is enabled and a cached topology exists,
           reuse the existing topology from the library */
        self->topo = lib->topo;
    } else {
        /* Determine where to store the topology:
           - If caching is enabled: store in lib->topo for reuse
           - If caching is disabled: store in self->topo (context-specific) */
        ucc_tl_cuda_topo_t **topo_ptr =
            lib->cfg.topo_cache_enable ? &lib->topo : &self->topo;

        /* Create new topology instance and store it in the appropriate location */
        status = ucc_tl_cuda_topo_create((const ucc_base_lib_t *)&lib->super,
                                         topo_ptr);
        if (status != UCC_OK) {
            tl_error(self->super.super.lib, "failed to initialize topology");
            goto free_mpool;
        }
        /* Update the context's topology pointer to point to the newly created topology */
        self->topo = *topo_ptr;
    }

    status = ucc_tl_cuda_topo_get_pci_id(self->device, &self->device_id);
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to get pci id for device %d, status: %s", self->device,
                 ucc_status_string(status));
        goto free_mpool;
    }

    self->ipc_cache = kh_init(tl_cuda_ep_hash);
    tl_debug(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;

free_mpool:
    ucc_mpool_cleanup(&self->req_mp, 1);
    return status;
}

ucc_status_t ucc_tl_cuda_mem_map(const ucc_base_context_t *context, /* NOLINT */
                                 int type, void *memh, void *tl_h) /* NOLINT */
{
    return UCC_ERR_NOT_IMPLEMENTED;
}

ucc_status_t ucc_tl_cuda_mem_unmap(const ucc_base_context_t *context, /* NOLINT */
                                   int type, void *tl_h) /* NOLINT */
{
    return UCC_ERR_NOT_IMPLEMENTED;
}

ucc_status_t ucc_tl_cuda_memh_pack(const ucc_base_context_t *context, /* NOLINT */
                                   int type, void *memh, void **pack_buffer) /* NOLINT */
{
    return UCC_ERR_NOT_IMPLEMENTED;
}

/**
 * @brief Cleanup function for CUDA TL context
 *
 * This function is responsible for cleaning up resources associated with a CUDA TL context.
 * It performs the following operations:
 * 1. Logs the context finalization with debug information
 * 2. Destroys the IPC cache hash table if it exists
 * 3. Cleans up topology if it's context-specific (not cached)
 * 4. Cleans up the request memory pool
 *
 * @param self Pointer to the CUDA TL context structure to be cleaned up
 */
UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_context_t)
{
    ucc_tl_cuda_lib_t *lib =
        ucc_derived_of(self->super.super.lib, ucc_tl_cuda_lib_t);

    // Log context finalization for debugging purposes
    tl_debug(self->super.super.lib, "finalizing tl context: %p", self);

    // Clean up IPC cache if it exists
    if (self->ipc_cache != NULL) {
        kh_destroy(tl_cuda_ep_hash, self->ipc_cache);
        self->ipc_cache = NULL;
    }

    // Only destroy topology if it's context-specific (not cached)
    // For cached topology, it will be destroyed when the library is cleaned up
    if (self->topo != NULL && !lib->cfg.topo_cache_enable) {
        ucc_tl_cuda_topo_destroy(self->topo);
        self->topo = NULL;
    }

    // Clean up the request memory pool with force leak check
    ucc_mpool_cleanup(&self->req_mp, 1);
}

UCC_CLASS_DEFINE(ucc_tl_cuda_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_cuda_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    ucc_base_ctx_attr_clear(attr);
    attr->topo_required = 1;
    return UCC_OK;
}
