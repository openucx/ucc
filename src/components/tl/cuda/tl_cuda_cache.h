/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_CACHE_H_
#define UCC_TL_CUDA_CACHE_H_

#include <ucs/datastruct/pgtable.h>
#include <ucs/datastruct/list.h>
#include <cuda.h>
#include <cuda_runtime.h>

typedef struct ucc_tl_cuda_cache_region {
    ucs_pgt_region_t    super;        /**< Base class - page table region */
    ucs_list_link_t     list;         /**< List element */
    void               *d_ptr;
    size_t              size;
    cudaIpcMemHandle_t  mem_handle;
    void               *mapped_addr;  /**< Local mapped address */
    uint64_t            refcount;     /**< Track inflight ops before unmapping*/
} ucc_tl_cuda_cache_region_t;

typedef struct ucc_tl_cuda_cache {
    pthread_rwlock_t  lock;       /**< protests the page table */
    ucs_pgtable_t     pgtable;    /**< Page table to hold the regions */
    char             *name;       /**< Name */
} ucc_tl_cuda_cache_t;

ucc_status_t ucc_tl_cuda_create_cache(ucc_tl_cuda_cache_t **cache,
                                      const char *name);

void ucc_tl_cuda_destroy_cache(ucc_tl_cuda_cache_t *cache);

ucc_status_t ucc_tl_cuda_map_memhandle(const void *dptr, size_t size,
                                       cudaIpcMemHandle_t mem_handle,
                                       void **mapped_addr,
                                       ucc_tl_cuda_cache_t *cache);

ucc_status_t ucc_tl_cuda_unmap_memhandle(uintptr_t d_bptr, void *mapped_addr,
                                         ucc_tl_cuda_cache_t *cache);

ucc_tl_cuda_cache_t* ucc_tl_cuda_get_cache(ucc_tl_cuda_team_t *team,
                                           ucc_rank_t rank);

#endif
