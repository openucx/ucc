/**
 * Copyright (C) Mellanox Technologies Ltd. 2018.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_CUDA_IPC_H_
#define UCC_CUDA_IPC_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <cuda.h>
#include <cuda_runtime.h>


typedef struct ucc_cuda_ipc_cache        ucc_cuda_ipc_cache_t;
typedef struct ucc_cuda_ipc_cache_region ucc_cuda_ipc_cache_region_t;


struct ucc_cuda_ipc_cache_region {
    ucs_pgt_region_t        super;        /**< Base class - page table region */
    ucs_list_link_t         list;         /**< List element */
    void                    *d_ptr;
    size_t                  size;
    cudaIpcMemHandle_t      mem_handle;
    void                    *mapped_addr; /**< Local mapped address */
    uint64_t                refcount;     /**< Track inflight ops before unmapping*/
};


struct ucc_cuda_ipc_cache {
    pthread_rwlock_t      lock;       /**< protests the page table */
    ucs_pgtable_t         pgtable;    /**< Page table to hold the regions */
    char                  *name;      /**< Name */
};


ucc_status_t ucc_cuda_ipc_create_cache(ucc_cuda_ipc_cache_t **cache,
                                       const char *name);


void ucc_cuda_ipc_destroy_cache(ucc_cuda_ipc_cache_t *cache);


ucc_status_t
ucc_cuda_ipc_map_memhandle(const void *dptr, size_t size, cudaIpcMemHandle_t mem_handle, void **mapped_addr, ucc_cuda_ipc_cache_t *cache);
ucc_status_t ucc_cuda_ipc_unmap_memhandle(uintptr_t d_bptr, void *mapped_addr, ucc_cuda_ipc_cache_t *cache);
#endif
