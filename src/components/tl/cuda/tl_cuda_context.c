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
#include <string.h>

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

/**
 * @brief Map a memory region for CUDA IPC access
 *
 * This function maps a memory region for inter-process communication using
 * CUDA IPC. For export mode, it creates an IPC memory handle that can be
 * shared with other processes. For import mode, it would open a remote IPC
 * handle (not yet implemented).
 *
 * @param [in]  context  CUDA TL context
 * @param [in]  mode     Memory mapping mode (export/import)
 * @param [in]  memh     Memory handle containing address and length info
 * @param [out] tl_h     TL-specific handle to store mapping information
 *
 * @return UCC_OK on success, error code on failure
 */
ucc_status_t ucc_tl_cuda_mem_map(
    const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
    ucc_mem_map_memh_t *memh, ucc_mem_map_tl_t *tl_h)
{
    ucc_tl_cuda_context_t       *ctx = ucc_derived_of(context, ucc_tl_cuda_context_t);
    ucc_tl_cuda_memh_data_t     *m_data;
    cudaError_t                  cuda_st;
    struct cudaPointerAttributes ptr_attrs;

    /* Initialize tl_h fields */
    tl_h->tl_data = NULL;
    strncpy(tl_h->tl_name, "cuda", UCC_MEM_MAP_TL_NAME_LEN - 1);

    if (mode != UCC_MEM_MAP_MODE_EXPORT &&
        mode != UCC_MEM_MAP_MODE_IMPORT) {
        return UCC_OK;
    }

    if (mode == UCC_MEM_MAP_MODE_EXPORT) {
        /* Check if memory is CUDA device memory - cudaIpcGetMemHandle only works
         * on device memory allocated with cudaMalloc */
        cuda_st = cudaPointerGetAttributes(&ptr_attrs, memh->address);
        if (cuda_st != cudaSuccess) {
            /* Not a CUDA pointer or error querying - nothing to do for this TL */
            tl_debug(
                ctx->super.super.lib,
                "cudaPointerGetAttributes failed for %p: %s - skipping CUDA "
                "IPC",
                memh->address,
                cudaGetErrorString(cuda_st));
            cudaGetLastError(); /* Clear error state */
            return UCC_OK; /* Return OK - other TLs may handle this memory */
        }

        /* cudaMemoryTypeDevice = 2, check if it's device memory */
        if (ptr_attrs.type != cudaMemoryTypeDevice) {
            tl_debug(
                ctx->super.super.lib,
                "memory at %p is not device memory (type=%d), skipping CUDA "
                "IPC",
                memh->address,
                ptr_attrs.type);
            return UCC_OK; /* Return OK - other TLs may handle this memory */
        }
    }

    m_data = ucc_calloc(
        1, sizeof(ucc_tl_cuda_memh_data_t), "tl cuda memh data");
    if (!m_data) {
        tl_error(ctx->super.super.lib, "failed to allocate tl cuda memh data");
        return UCC_ERR_NO_MEMORY;
    }

    tl_h->tl_data = m_data;

    if (mode == UCC_MEM_MAP_MODE_EXPORT) {
        /* Resolve the allocation base: cudaIpcGetMemHandle requires the pointer
         * returned by cudaMalloc, not an interior sub-buffer pointer. */
        ucc_mem_attr_t mem_attr;
        ucc_status_t   st;

        mem_attr.field_mask   = UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                                UCC_MEM_ATTR_FIELD_ALLOC_LENGTH;
        mem_attr.alloc_length = memh->len;
        st = ucc_mc_get_mem_attr(memh->address, &mem_attr);
        if (ucc_unlikely(st != UCC_OK)) {
            tl_error(ctx->super.super.lib,
                     "failed to get mem attrs for %p", memh->address);
            ucc_free(m_data);
            tl_h->tl_data = NULL;
            return st;
        }

        cuda_st = cudaIpcGetMemHandle(&m_data->ipc_handle,
                                      mem_attr.base_address);
        if (cuda_st != cudaSuccess) {
            tl_error(
                ctx->super.super.lib,
                "cudaIpcGetMemHandle failed: %s",
                cudaGetErrorString(cuda_st));
            ucc_free(m_data);
            tl_h->tl_data = NULL;
            return UCC_ERR_NO_RESOURCE;
        }
        m_data->base_address = mem_attr.base_address;
        m_data->length       = mem_attr.alloc_length;
        m_data->offset       = (ptrdiff_t)memh->address -
                               (ptrdiff_t)mem_attr.base_address;
        tl_debug(
            ctx->super.super.lib,
            "exported CUDA IPC handle for address %p (base %p, offset %td, "
            "length %zu)",
            memh->address, m_data->base_address, m_data->offset, m_data->length);
    } else if (mode == UCC_MEM_MAP_MODE_IMPORT) {
        /* Import mode: unpack the IPC handle from the pack_buffer */
        if (memh->num_tls > 0) {
            size_t offset = 0;
            int    found  = 0;
            int    i;

            /* Search through pack_buffer for our TL's data */
            for (i = 0; i < memh->num_tls && !found; i++) {
                char   *tl_name     = PTR_OFFSET(memh->pack_buffer, offset);
                size_t *packed_size = PTR_OFFSET(
                    memh->pack_buffer, offset + UCC_MEM_MAP_TL_NAME_LEN);
                void *packed_data = PTR_OFFSET(
                    memh->pack_buffer,
                    offset + UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t));

                if (strncmp(tl_name, "cuda", UCC_MEM_MAP_TL_NAME_LEN) == 0 &&
                    *packed_size > 0) {
                    /* Format: [ipc_handle] [base_address] [length] [offset] */
                    size_t poff = 0;
                    memcpy(&m_data->ipc_handle,
                           PTR_OFFSET(packed_data, poff),
                           sizeof(cudaIpcMemHandle_t));
                    poff += sizeof(cudaIpcMemHandle_t);
                    memcpy(&m_data->base_address,
                           PTR_OFFSET(packed_data, poff), sizeof(void *));
                    poff += sizeof(void *);
                    memcpy(&m_data->length,
                           PTR_OFFSET(packed_data, poff), sizeof(size_t));
                    poff += sizeof(size_t);
                    memcpy(&m_data->offset,
                           PTR_OFFSET(packed_data, poff), sizeof(ptrdiff_t));
                    found = 1;
                    tl_debug(
                        ctx->super.super.lib,
                        "imported CUDA IPC handle for base %p offset %td "
                        "length %zu",
                        m_data->base_address, m_data->offset, m_data->length);
                }
                offset += UCC_MEM_MAP_TL_NAME_LEN + sizeof(size_t) +
                          *packed_size;
            }

            if (!found) {
                tl_debug(
                    ctx->super.super.lib,
                    "no CUDA TL data found in pack_buffer");
                ucc_free(m_data);
                tl_h->tl_data = NULL;
                return UCC_OK;
            }
        } else {
            /* No TL data to import */
            tl_debug(
                ctx->super.super.lib, "import mode without TL data, skipping");
            ucc_free(m_data);
            tl_h->tl_data = NULL;
            return UCC_OK;
        }
    }

    return UCC_OK;
}

/**
 * @brief Unmap a previously mapped CUDA memory region
 *
 * This function releases resources associated with a mapped memory region.
 * For export mode, it simply frees the TL data. For import mode, it would
 * close any opened IPC handles.
 *
 * @param [in] context  CUDA TL context
 * @param [in] mode     Memory mapping mode used during mapping
 * @param [in] tl_h     TL-specific handle to unmap
 *
 * @return UCC_OK on success, error code on failure
 */
ucc_status_t ucc_tl_cuda_mem_unmap(
    const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
    ucc_mem_map_tl_t *tl_h)
{
    ucc_tl_cuda_context_t *ctx = ucc_derived_of(context, ucc_tl_cuda_context_t);
    ucc_tl_cuda_memh_data_t *data;

    if (!tl_h || !tl_h->tl_data) {
        return UCC_OK;
    }

    data = (ucc_tl_cuda_memh_data_t *)tl_h->tl_data;

    if (mode == UCC_MEM_MAP_MODE_EXPORT) {
        /* For export mode, just free the data structure.
         * The IPC handle doesn't need explicit cleanup. */
        tl_debug(
            ctx->super.super.lib,
            "unmapping exported CUDA memory at %p",
            data->base_address);
    } else if (mode == UCC_MEM_MAP_MODE_IMPORT) {
        /* For import mode, nothing special to do as we don't open handles here */
        tl_debug(
            ctx->super.super.lib,
            "unmapping imported CUDA memory at %p",
            data->base_address);
    } else {
        tl_debug(
            ctx->super.super.lib,
            "mem_unmap mode %d not supported for CUDA TL",
            mode);
    }

    ucc_free(data);
    tl_h->tl_data = NULL;

    return UCC_OK;
}

/**
 * @brief Pack CUDA memory handle for transfer to remote processes
 *
 * This function packs the CUDA IPC memory handle into a buffer that can be
 * sent to remote processes. The remote process can then use this packed
 * data to open the IPC handle and access the memory.
 *
 * @param [in]  context      CUDA TL context
 * @param [in]  mode         Memory mapping mode
 * @param [in]  tl_h         TL-specific handle containing the IPC handle
 * @param [out] pack_buffer  Allocated buffer containing packed handle data
 *
 * @return UCC_OK on success, error code on failure
 */
ucc_status_t ucc_tl_cuda_memh_pack(
    const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
    ucc_mem_map_tl_t *tl_h, void **pack_buffer)
{
    ucc_tl_cuda_context_t   *ctx = ucc_derived_of(context, ucc_tl_cuda_context_t);
    ucc_tl_cuda_memh_data_t *data;
    void                    *buffer;
    size_t                   offset;

    /* If no TL data (e.g., non-device memory), nothing to pack - return success */
    if (!tl_h->tl_data) {
        tl_debug(
            ctx->super.super.lib,
            "no CUDA TL data to pack (non-device memory?)");
        tl_h->packed_size = 0;
        *pack_buffer      = NULL;
        return UCC_OK;
    }

    data = (ucc_tl_cuda_memh_data_t *)tl_h->tl_data;

    if (mode != UCC_MEM_MAP_MODE_EXPORT) {
        tl_debug(
            ctx->super.super.lib,
            "memh_pack only supported for export/export_offload mode, skipping");
        tl_h->packed_size = 0;
        *pack_buffer      = NULL;
        return UCC_OK;
    }

    /*
     * Pack format:
     *   - cudaIpcMemHandle_t (64 bytes)
     *   - base_address (sizeof(void *))
     *   - length (sizeof(size_t))
     *   - offset (sizeof(ptrdiff_t))  -- user pointer - base_address
     */
    tl_h->packed_size = sizeof(cudaIpcMemHandle_t) + sizeof(void *) +
                        sizeof(size_t) + sizeof(ptrdiff_t);

    buffer = ucc_malloc(tl_h->packed_size, "cuda packed memh");
    if (!buffer) {
        tl_error(
            ctx->super.super.lib,
            "failed to allocate packed buffer of size %zu",
            tl_h->packed_size);
        return UCC_ERR_NO_MEMORY;
    }

    offset = 0;
    memcpy(buffer, &data->ipc_handle, sizeof(cudaIpcMemHandle_t));
    offset += sizeof(cudaIpcMemHandle_t);

    memcpy(PTR_OFFSET(buffer, offset), &data->base_address, sizeof(void *));
    offset += sizeof(void *);

    memcpy(PTR_OFFSET(buffer, offset), &data->length, sizeof(size_t));
    offset += sizeof(size_t);

    memcpy(PTR_OFFSET(buffer, offset), &data->offset, sizeof(ptrdiff_t));

    *pack_buffer = buffer;

    tl_debug(
        ctx->super.super.lib,
        "packed CUDA IPC handle, size %zu bytes",
        tl_h->packed_size);

    return UCC_OK;
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
