/**
 * Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CUDA_RESOURCES_H_
#define UCC_MC_CUDA_RESOURCES_H_

#include "components/mc/base/ucc_mc_base.h"
#include "utils/arch/cuda_def.h"
#include "utils/ucc_mpool.h"

typedef struct ucc_mc_cuda_config {
    ucc_mc_config_t super;
    size_t          mpool_elem_size;
    int             mpool_max_elems;
} ucc_mc_cuda_config_t;

typedef struct ucc_mc_cuda_resources {
    CUcontext    cu_ctx;
    cudaStream_t stream;
    ucc_mpool_t  scratch_mpool;
} ucc_mc_cuda_resources_t;

extern ucc_mc_cuda_config_t *ucc_mc_cuda_config;

ucc_status_t ucc_mc_cuda_resources_init(ucc_mc_base_t *mc,
                                        ucc_mc_cuda_resources_t *resources);

void ucc_mc_cuda_resources_cleanup(ucc_mc_cuda_resources_t *resources);

KHASH_INIT(ucc_mc_cuda_resources_hash, unsigned long long, void*, 1, \
           kh_int64_hash_func, kh_int64_hash_equal);
#define ucc_mc_cuda_resources_hash_t khash_t(ucc_mc_cuda_resources_hash)

static inline
void* mc_cuda_resources_hash_get(ucc_mc_cuda_resources_hash_t *h,
                                 unsigned long long key)
{
    khiter_t  k;
    void     *value;

    k = kh_get(ucc_mc_cuda_resources_hash, h , key);
    if (k == kh_end(h)) {
        return NULL;
    }
    value = kh_value(h, k);
    return value;
}

static inline
void mc_cuda_resources_hash_put(ucc_mc_cuda_resources_hash_t *h,
                                unsigned long long key,
                                void *value)
{
    int ret;
    khiter_t k;
    k = kh_put(ucc_mc_cuda_resources_hash, h, key, &ret);
    kh_value(h, k) = value;
}

static inline
void* mc_cuda_resources_hash_pop(ucc_mc_cuda_resources_hash_t *h)
{
    void    *resources = NULL;
    khiter_t k;

    k = kh_begin(h);
    while (k != kh_end(h)) {
        if (kh_exist(h, k)) {
            resources = kh_value(h, k);
            break;
        }
        k++;
    }

    if (resources) {
        kh_del(ucc_mc_cuda_resources_hash, h, k);
    }
    return resources;
}

#endif
