/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "cuda_runtime.h"
#include "tl_ucp_cuda_ipc.h"

#define ENABLE_CACHE 1

static ucs_pgt_dir_t *ucc_cuda_ipc_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable)
{
    void *ptr;
    int ret;

    ret = posix_memalign(&ptr, ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                             sizeof(ucs_pgt_dir_t));
    return (ret == 0) ? ptr : NULL;
}

static void ucc_cuda_ipc_cache_pgt_dir_release(const ucs_pgtable_t *pgtable,
                                               ucs_pgt_dir_t *dir)
{
    free(dir);
}


static void
ucc_cuda_ipc_cache_region_collect_callback(const ucs_pgtable_t *pgtable,
                                           ucs_pgt_region_t *pgt_region,
                                           void *arg)
{
    ucs_list_link_t *list = arg;
    ucc_cuda_ipc_cache_region_t *region;

    region = ucs_derived_of(pgt_region, ucc_cuda_ipc_cache_region_t);
    ucs_list_add_tail(list, &region->list);
}

static void ucc_cuda_ipc_cache_purge(ucc_cuda_ipc_cache_t *cache)
{
    ucc_cuda_ipc_cache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&cache->pgtable, ucc_cuda_ipc_cache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        CUDACHECK_NO_RET(cudaIpcCloseMemHandle(region->mapped_addr));
        free(region);
    }
}


ucc_status_t ucc_cuda_ipc_create_cache(ucc_cuda_ipc_cache_t **cache,
                                       const char *name)
{

    ucc_status_t status;
    ucc_cuda_ipc_cache_t *cache_desc;
    int ret;

    cache_desc = ucc_malloc(sizeof(ucc_cuda_ipc_cache_t), "ucc_cuda_ipc_cache_t");
    if (cache_desc == NULL) {
        ucs_error("failed to allocate memory for cuda_ipc cache");
        return UCC_ERR_NO_MEMORY;
    }

    ret = pthread_rwlock_init(&cache_desc->lock, NULL);
    if (ret) {
        ucs_error("pthread_rwlock_init() failed: %m");
        status = UCC_ERR_INVALID_PARAM;
        goto err;
    }

    if (UCS_OK != ucs_pgtable_init(&cache_desc->pgtable,
                                   ucc_cuda_ipc_cache_pgt_dir_alloc,
                                   ucc_cuda_ipc_cache_pgt_dir_release)) {
        goto err_destroy_rwlock;
    }

    cache_desc->name = strdup(name);
    if (cache_desc->name == NULL) {
        status = UCC_ERR_NO_MEMORY;
        goto err_destroy_rwlock;
    }

    *cache = cache_desc;
    return UCC_OK;

err_destroy_rwlock:
    pthread_rwlock_destroy(&cache_desc->lock);
err:
    free(cache_desc);
    return status;

    return UCC_OK;
}


void ucc_cuda_ipc_destroy_cache(ucc_cuda_ipc_cache_t *cache)
{
    ucc_cuda_ipc_cache_purge(cache);
    ucs_pgtable_cleanup(&cache->pgtable);
    pthread_rwlock_destroy(&cache->lock);
    free(cache->name);
    free(cache);
}

static void ucc_cuda_ipc_cache_invalidate_regions(ucc_cuda_ipc_cache_t *cache,
                                                  void *from, void *to)
{
    ucs_list_link_t region_list;
    ucs_status_t status;
    ucc_cuda_ipc_cache_region_t *region, *tmp;

    ucs_list_head_init(&region_list);
    ucs_pgtable_search_range(&cache->pgtable, (ucs_pgt_addr_t)from,
                             (ucs_pgt_addr_t)to,
                             ucc_cuda_ipc_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucs_error("failed to remove address:%p from cache (%s)",
                      (void *)region->d_ptr, ucs_status_string(status));
        }
        CUDACHECK_NO_RET(cudaIpcCloseMemHandle(region->mapped_addr));
        free(region);
    }
    ucs_trace("%s: closed memhandles in the range [%p..%p]",
              cache->name, from, to);
}

ucc_status_t
ucc_cuda_ipc_map_memhandle(const void *d_ptr, size_t size, cudaIpcMemHandle_t mem_handle,
                           void **mapped_addr, ucc_cuda_ipc_cache_t *cache)
{
#if ENABLE_CACHE
    ucs_status_t status;
    ucs_status_t ucs_status;
    ucs_pgt_region_t *pgt_region;
    ucc_cuda_ipc_cache_region_t *region;
    int ret;


    pthread_rwlock_wrlock(&cache->lock);  //todo :is lock needed
    pgt_region = ucs_pgtable_lookup(&cache->pgtable, (uintptr_t) d_ptr);
    if (pgt_region != NULL) {
        region = ucc_derived_of(pgt_region, ucc_cuda_ipc_cache_region_t);
        if (memcmp((const void *)&mem_handle, (const void *)&region->mem_handle,
                   sizeof(cudaIpcMemHandle_t)) == 0) {
            /*cache hit */
            ucs_debug("%s: cuda_ipc cache hit addr:%p size:%lu region:"
                      UCS_PGT_REGION_FMT, cache->name, d_ptr,
                      size, UCS_PGT_REGION_ARG(&region->super));

            *mapped_addr = region->mapped_addr;
            ucc_assert(region->refcount < UINT64_MAX);
            region->refcount++;
            pthread_rwlock_unlock(&cache->lock);
            return UCS_OK;
        } else {
            ucs_debug("%s: cuda_ipc cache remove stale region:"
                      UCS_PGT_REGION_FMT " new_addr:%p new_size:%lu",
                      cache->name, UCS_PGT_REGION_ARG(&region->super),
                      d_ptr, size);

            status = ucs_pgtable_remove(&cache->pgtable, &region->super);
            if (status != UCS_OK) {
                ucs_error("%s: failed to remove address:%p from cache",
                          cache->name, d_ptr);
                goto err;
            }

            /* close memhandle */
            CUDACHECK(cudaIpcCloseMemHandle(region->mapped_addr));
            free(region);
        }
    }

    CUDACHECK(cudaIpcOpenMemHandle(mapped_addr, mem_handle, cudaIpcMemLazyEnablePeerAccess));

    /*create new cache entry */
    ret = posix_memalign((void **)&region,
                          ucs_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                          sizeof(ucc_cuda_ipc_cache_region_t));
    if (ret != 0) {
        ucs_warn("failed to allocate uct_cuda_ipc_cache region");
        status = UCS_ERR_NO_MEMORY;
        goto err;
    }

    region->super.start = ucs_align_down_pow2((uintptr_t)d_ptr,
                                               UCS_PGT_ADDR_ALIGN);
    region->super.end   = ucs_align_up_pow2  ((uintptr_t)d_ptr + size,
                                               UCS_PGT_ADDR_ALIGN);
    region->d_ptr       = (void *)d_ptr;
    region->size        = size;
    region->mem_handle  = mem_handle;
    region->mapped_addr = *mapped_addr;
    region->refcount    = 1;

    ucs_status = ucs_pgtable_insert(&cache->pgtable, &region->super);
    if (ucs_status == UCS_ERR_ALREADY_EXISTS) {
        /* overlapped region means memory freed at source. remove and try insert */
        ucc_cuda_ipc_cache_invalidate_regions(cache,
                                              (void *)region->super.start,
                                              (void *)region->super.end);
        status = ucs_pgtable_insert(&cache->pgtable, &region->super);
    }

    if (ucs_status != UCS_OK) {

        ucs_error("%s: failed to insert region:"UCS_PGT_REGION_FMT" size:%lu :%s",
                  cache->name, UCS_PGT_REGION_ARG(&region->super), size,
                  ucs_status_string(ucs_status));
        free(region);
        goto err;
    }

    ucs_trace("%s: cuda_ipc cache new region:"UCS_PGT_REGION_FMT" size:%lu",
              cache->name, UCS_PGT_REGION_ARG(&region->super), size);

    status = UCC_OK;

err:
    pthread_rwlock_unlock(&cache->lock);
    return status;


#else
    CUDACHECK(cudaIpcOpenMemHandle(mapped_addr, mem_handle, cudaIpcMemLazyEnablePeerAccess));
    return UCC_OK;
#endif
}


ucc_status_t ucc_cuda_ipc_unmap_memhandle(uintptr_t d_bptr, void *mapped_addr, ucc_cuda_ipc_cache_t *cache)
{
#if ENABLE_CACHE
    ucs_pgt_region_t *pgt_region;
    ucc_cuda_ipc_cache_region_t *region;

    /* use write lock because cache maybe modified */
    pthread_rwlock_wrlock(&cache->lock);
    pgt_region = ucs_pgtable_lookup(&cache->pgtable, d_bptr);
    ucc_assert(pgt_region != NULL);
    region = ucs_derived_of(pgt_region, ucc_cuda_ipc_cache_region_t);

    ucc_assert(region->refcount >= 1);
    region->refcount--;

    pthread_rwlock_unlock(&cache->lock);
#else
   CUDACHECK(cudaIpcCloseMemHandle(mapped_addr));
#endif
   return UCC_OK;
}
