/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_cuda.h"
#include "utils/ucc_malloc.h"
#include <cuda_runtime.h>

static ucc_config_field_t ucc_mc_cuda_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_cuda_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {NULL}};

static ucc_status_t ucc_mc_cuda_init()
{
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_finalize()
{
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_alloc(void **ptr, size_t size)
{
    cudaError_t st;

    st = cudaMalloc(ptr, size);
    if (st != cudaSuccess) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to allocate %zd bytes, "
                 "cuda error %d(%s)",
                 size, st, cudaGetErrorString(st));
        return UCC_ERR_NO_MEMORY;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_free(void *ptr)
{
    cudaError_t st;

    st = cudaFree(ptr);
    if (st != cudaSuccess) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to free mem at %p, "
                 "cuda error %d(%s)",
                 ptr, st, cudaGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_type(const void *ptr,
                                         ucc_memory_type_t *mem_type)
{
    struct cudaPointerAttributes attr;
    cudaError_t                  st;

    st = cudaPointerGetAttributes(&attr, ptr);
    if (st != cudaSuccess) {
        cudaGetLastError();
        return UCC_ERR_NOT_SUPPORTED;
    }

#if CUDART_VERSION >= 10000
    switch (attr.type) {
    case cudaMemoryTypeUnregistered:
    case cudaMemoryTypeHost:
        *mem_type = UCC_MEMORY_TYPE_HOST;
        break;
    case cudaMemoryTypeDevice:
        *mem_type = UCC_MEMORY_TYPE_CUDA;
        break;
    case cudaMemoryTypeManaged:
        *mem_type = UCC_MEMORY_TYPE_CUDA_MANAGED;
        break;
    }
#else
    if (attr.memoryType == cudaMemoryTypeDevice) {
        if (attr.isManaged) {
            *mem_type = UCC_MEMORY_TYPE_CUDA_MANAGED;
        } else {
            *mem_type = UCC_MEMORY_TYPE_CUDA;
        }
    }
    else {
        *mem_type = UCC_MEMORY_TYPE_HOST;
    }
#endif

    return UCC_OK;
}

ucc_mc_cuda_t ucc_mc_cuda = {
    .super.super.name = "cuda mc",
    .super.ref_cnt    = 0,
    .super.type       = UCC_MEMORY_TYPE_CUDA,
    .super.config_table =
        {
            .name   = "CUDA memory component",
            .prefix = "MC_CUDA_",
            .table  = ucc_mc_cuda_config_table,
            .size   = sizeof(ucc_mc_cuda_config_t),
        },
    .super.init          = ucc_mc_cuda_init,
    .super.finalize      = ucc_mc_cuda_finalize,
    .super.ops.mem_type  = ucc_mc_cuda_mem_type,
    .super.ops.mem_alloc = ucc_mc_cuda_mem_alloc,
    .super.ops.mem_free  = ucc_mc_cuda_mem_free,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cuda.super.config_table,
                                &ucc_config_global_list);
