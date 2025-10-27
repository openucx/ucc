/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_cuda.h"
#include "tl_cuda_cache.h"
#include "tl_cuda_ep_hash.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "utils/arch/cuda_def.h"
#include "core/ucc_team.h"
#include <cuda_runtime.h>

#define ENABLE_CACHE 1

static ucs_pgt_dir_t*
ucc_tl_cuda_cache_pgt_dir_alloc(const ucs_pgtable_t *pgtable) //NOLINT: pgtable is unused
{
    void *ptr;
    int ret;

    ret = posix_memalign(&ptr, ucc_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                         sizeof(ucs_pgt_dir_t));
    return (ret == 0) ? ptr : NULL;
}

static void ucc_tl_cuda_cache_pgt_dir_release(const ucs_pgtable_t *pgtable, //NOLINT: pgtable is unused
                                              ucs_pgt_dir_t *dir)
{
    free(dir);
}

static void
ucc_tl_cuda_cache_region_collect_callback(const ucs_pgtable_t *pgtable, //NOLINT: pgtable is unused
                                          ucs_pgt_region_t *pgt_region,
                                          void *arg)
{
    ucs_list_link_t *list = arg;
    ucc_tl_cuda_cache_region_t *region;

    region = ucc_derived_of(pgt_region, ucc_tl_cuda_cache_region_t);
    ucs_list_add_tail(list, &region->list);
}

static void ucc_tl_cuda_cache_purge(ucc_tl_cuda_cache_t *cache)
{
    ucc_tl_cuda_cache_region_t *region, *tmp;
    ucs_list_link_t region_list;

    ucs_list_head_init(&region_list);
    ucs_pgtable_purge(&cache->pgtable, ucc_tl_cuda_cache_region_collect_callback,
                      &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        CUDA_FUNC(cudaIpcCloseMemHandle(region->mapped_addr));
        free(region);
    }
}

ucc_status_t ucc_tl_cuda_create_cache(ucc_tl_cuda_cache_t **cache,
                                      const char *name)
{
    ucc_status_t status;
    ucs_status_t ucs_st;
    ucc_tl_cuda_cache_t *cache_desc;
    int ret;

    cache_desc = ucc_malloc(sizeof(ucc_tl_cuda_cache_t), "ucc_tl_cuda_cache_t");
    if (cache_desc == NULL) {
        ucc_error("failed to allocate memory for tl_cuda cache");
        return UCC_ERR_NO_MEMORY;
    }

    ret = pthread_rwlock_init(&cache_desc->lock, NULL);
    if (ret) {
        ucc_error("pthread_rwlock_init() failed: %m");
        status = UCC_ERR_INVALID_PARAM;
        goto err;
    }

    ucs_st = ucs_pgtable_init(&cache_desc->pgtable,
                              ucc_tl_cuda_cache_pgt_dir_alloc,
                              ucc_tl_cuda_cache_pgt_dir_release);
    if (ucc_unlikely(UCS_OK != ucs_st)) {
        status = ucs_status_to_ucc_status(ucs_st);
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
}

void ucc_tl_cuda_destroy_cache(ucc_tl_cuda_cache_t *cache)
{
    ucc_tl_cuda_cache_purge(cache);
    ucs_pgtable_cleanup(&cache->pgtable);
    pthread_rwlock_destroy(&cache->lock);
    free(cache->name);
    free(cache);
}

#if ENABLE_CACHE
static void ucc_tl_cuda_cache_invalidate_regions(ucc_tl_cuda_cache_t *cache,
                                                 void *from, void *to)
{
    ucs_list_link_t region_list;
    ucs_status_t status;
    ucc_tl_cuda_cache_region_t *region, *tmp;

    ucs_list_head_init(&region_list);
    ucs_pgtable_search_range(&cache->pgtable, (ucs_pgt_addr_t)from,
                             (ucs_pgt_addr_t)to - 1,
                             ucc_tl_cuda_cache_region_collect_callback,
                             &region_list);
    ucs_list_for_each_safe(region, tmp, &region_list, list) {
        status = ucs_pgtable_remove(&cache->pgtable, &region->super);
        if (status != UCS_OK) {
            ucc_error("failed to remove address:%p from cache (%s)",
                      (void *)region->d_ptr, ucs_status_string(status));
        }
        CUDA_FUNC(cudaIpcCloseMemHandle(region->mapped_addr));
        free(region);
    }
    ucc_trace("%s: closed memhandles in the range [%p..%p]",
              cache->name, from, to);
}
#endif

ucc_status_t
ucc_tl_cuda_map_memhandle(const void *d_ptr, size_t size,
                          cudaIpcMemHandle_t mem_handle, void **mapped_addr,
                          ucc_tl_cuda_cache_t *cache)
{
    if (d_ptr == NULL || size == 0) {
        *mapped_addr = NULL;
        return UCC_OK;
    }
#if ENABLE_CACHE
    ucc_status_t status;
    ucs_status_t ucs_status;
    ucs_pgt_region_t *pgt_region;
    ucc_tl_cuda_cache_region_t *region;
    cudaError_t cuerr;
    int ret;

    pthread_rwlock_wrlock(&cache->lock);  //todo :is lock needed
    pgt_region = ucs_pgtable_lookup(&cache->pgtable, (uintptr_t) d_ptr);
    if (pgt_region != NULL) {
        region = ucc_derived_of(pgt_region, ucc_tl_cuda_cache_region_t);
        if (memcmp((const void *)&mem_handle, (const void *)&region->mem_handle,
                   sizeof(cudaIpcMemHandle_t)) == 0) {
            /*cache hit */
            ucc_debug("%s: tl_cuda cache hit addr:%p size:%lu region:"
                      UCS_PGT_REGION_FMT, cache->name, d_ptr,
                      size, UCS_PGT_REGION_ARG(&region->super));

            *mapped_addr = region->mapped_addr;
            ucc_assert(region->refcount < UINT64_MAX);
            region->refcount++;
            pthread_rwlock_unlock(&cache->lock);
            return UCC_OK;
        } else {
            ucc_debug("%s: tl_cuda cache remove stale region:"
                      UCS_PGT_REGION_FMT " new_addr:%p new_size:%lu",
                      cache->name, UCS_PGT_REGION_ARG(&region->super),
                      d_ptr, size);

            ucs_status = ucs_pgtable_remove(&cache->pgtable, &region->super);
            if (ucc_unlikely(ucs_status != UCS_OK)) {
                ucc_error("%s: failed to remove address:%p from cache",
                          cache->name, d_ptr);
                status = ucs_status_to_ucc_status(ucs_status);
                goto err;
            }

            /* close memhandle */
            cuerr = cudaIpcCloseMemHandle(region->mapped_addr);
            if (ucc_unlikely(cuerr != cudaSuccess)) {
                ucc_error("cudaIpcCloseMemHandle error %d %s", cuerr,
                          cudaGetErrorName(cuerr));
                status = UCC_ERR_NO_MESSAGE;
                goto err;
            }
            free(region);
        }
    }

    cuerr = cudaIpcOpenMemHandle(mapped_addr, mem_handle,
                                 cudaIpcMemLazyEnablePeerAccess);
    if (cuerr != cudaSuccess) {
        if (cuerr == cudaErrorAlreadyMapped) {
            ucc_tl_cuda_cache_invalidate_regions(cache,
                    (void *)ucc_align_down_pow2((uintptr_t)d_ptr,
                    UCS_PGT_ADDR_ALIGN),
                    (void *)ucc_align_up_pow2((uintptr_t)d_ptr + size,
                    UCS_PGT_ADDR_ALIGN));
            cuerr = cudaIpcOpenMemHandle(mapped_addr, mem_handle,
                                         cudaIpcMemLazyEnablePeerAccess);
            if (cuerr != cudaSuccess) {
                if (cuerr == cudaErrorAlreadyMapped) {
                    ucc_tl_cuda_cache_purge(cache);
                    cuerr = cudaIpcOpenMemHandle(mapped_addr, mem_handle,
                                                 cudaIpcMemLazyEnablePeerAccess);
                    if (ucc_unlikely(cuerr != cudaSuccess)) {
                        ucc_error("cudaIpcOpenMemHandle error %d %s", cuerr,
                                  cudaGetErrorString(cuerr));
                        status = UCC_ERR_INVALID_PARAM;
                        goto err;
                    }
                } else {
                    ucc_error("%s: failed to open ipc mem handle. addr:%p len:%lu",
                              cache->name, d_ptr, size);
                }
            }
            cudaGetLastError();
        } else {
            ucc_error("%s: failed to open ipc mem handle. addr:%p len:%lu "
                      "err: %d %s", cache->name, d_ptr, size, cuerr,
                      cudaGetErrorString(cuerr));
            status = UCC_ERR_NO_MESSAGE;
            goto err;
        }
    }

    /*create new cache entry */
    ret = posix_memalign((void **)&region,
                          ucc_max(sizeof(void *), UCS_PGT_ENTRY_MIN_ALIGN),
                          sizeof(ucc_tl_cuda_cache_region_t));
    if (ret != 0) {
        ucc_warn("failed to allocate uct_tl_cuda_cache region");
        status = UCC_ERR_NO_MEMORY;
        goto err;
    }

    region->super.start = ucc_align_down_pow2((uintptr_t)d_ptr,
                                               UCS_PGT_ADDR_ALIGN);
    region->super.end   = ucc_align_up_pow2  ((uintptr_t)d_ptr + size,
                                               UCS_PGT_ADDR_ALIGN);
    region->d_ptr       = (void *)d_ptr;
    region->size        = size;
    region->mem_handle  = mem_handle;
    region->mapped_addr = *mapped_addr;
    region->refcount    = 1;

    ucs_status = ucs_pgtable_insert(&cache->pgtable, &region->super);
    if (ucs_status == UCS_ERR_ALREADY_EXISTS) {
        /* overlapped region means memory freed at source. remove and try insert */
        ucc_tl_cuda_cache_invalidate_regions(cache,
                                              (void *)region->super.start,
                                              (void *)region->super.end);
        ucs_status = ucs_pgtable_insert(&cache->pgtable, &region->super);
    }

    if (ucs_status != UCS_OK) {
        ucc_error("%s: failed to insert region:"UCS_PGT_REGION_FMT" size:%lu :%s",
                  cache->name, UCS_PGT_REGION_ARG(&region->super), size,
                  ucs_status_string(ucs_status));
        free(region);
        status = ucs_status_to_ucc_status(ucs_status);
        goto err;
    }

    ucc_debug("%s: tl_cuda cache new region:"UCS_PGT_REGION_FMT" size:%lu",
              cache->name, UCS_PGT_REGION_ARG(&region->super), size);

    status = UCC_OK;

err:
    pthread_rwlock_unlock(&cache->lock);
    // coverity[leaked_storage:FALSE]
    return status;


#else
    CUDA_FUNC(cudaIpcOpenMemHandle(mapped_addr, mem_handle,
                                         cudaIpcMemLazyEnablePeerAccess));
    return UCC_OK;
#endif
}

ucc_status_t ucc_tl_cuda_unmap_memhandle(uintptr_t d_bptr, void *mapped_addr,
                                         ucc_tl_cuda_cache_t *cache, int force)
{

    if ((d_bptr == 0) || (mapped_addr == 0)) {
        return UCC_OK;
    }

#if ENABLE_CACHE
    ucs_pgt_region_t *pgt_region;
    ucc_tl_cuda_cache_region_t *region;

    /* use write lock because cache maybe modified */
    pthread_rwlock_wrlock(&cache->lock);
    pgt_region = ucs_pgtable_lookup(&cache->pgtable, d_bptr);

    ucc_debug("%s: tl_cuda unmap addr:%p region:"
                UCS_PGT_REGION_FMT, cache->name, (void*)d_bptr,
                UCS_PGT_REGION_ARG(pgt_region));

    ucc_assert(pgt_region != NULL);
    region = ucc_derived_of(pgt_region, ucc_tl_cuda_cache_region_t);

    ucc_assert(region->refcount >= 1);
    region->refcount--;

    if ((region->refcount == 0 ) && (force == 1)) {
        ucs_pgtable_remove(&cache->pgtable, &region->super);
        CUDA_FUNC(cudaIpcCloseMemHandle(mapped_addr));
    }

    pthread_rwlock_unlock(&cache->lock);
#else
   CUDA_FUNC(cudaIpcCloseMemHandle(mapped_addr));
#endif
   return UCC_OK;
}

ucc_tl_cuda_cache_t* ucc_tl_cuda_get_cache(ucc_tl_cuda_team_t *team,
                                           ucc_rank_t rank)
{
    ucc_tl_cuda_context_t     *ctx = UCC_TL_CUDA_TEAM_CTX(team);
    ucc_context_addr_header_t *h   = NULL;
    ucc_tl_cuda_cache_t *cache;
    ucc_status_t status;

    h = ucc_get_team_ep_header(UCC_TL_CORE_CTX(team), UCC_TL_CORE_TEAM(team),
                               rank);
    cache = tl_cuda_hash_get(ctx->ipc_cache, h->ctx_id);
    if (ucc_unlikely(NULL == cache)) {
        status =  ucc_tl_cuda_create_cache(&cache, "ipc-cache");
        if (status != UCC_OK) {
            tl_error(UCC_TL_TEAM_LIB(team), "failed to create ipc cache");
            return NULL;
        }
        tl_cuda_hash_put(ctx->ipc_cache, h->ctx_id, cache);
    }
    return cache;
}
