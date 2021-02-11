/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_cuda.h"
#include "utils/ucc_malloc.h"
#include <cuda_runtime.h>
#include <cuda.h>

static ucc_config_field_t ucc_mc_cuda_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_cuda_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {"REDUCE_NUM_BLOCKS", "auto",
     "Number of thread blocks to use for reduction",
     ucc_offsetof(ucc_mc_cuda_config_t, reduce_num_blocks),
     UCC_CONFIG_TYPE_ULUNITS},

    {NULL}};

static ucc_status_t ucc_mc_cuda_init()
{
    struct cudaDeviceProp prop;
    int device;

    ucc_mc_cuda_config_t *cfg = MC_CUDA_CONFIG;
    CUDACHECK(cudaGetDevice(&device));
    CUDACHECK(cudaGetDeviceProperties(&prop, device));
    cfg->reduce_num_threads = prop.maxThreadsPerBlock;
    if (cfg->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        if (prop.maxGridSize[0] < cfg->reduce_num_blocks) {
            mc_warn(&ucc_mc_cuda.super, "number of blocks is too large, "
                    "max supported %d", prop.maxGridSize[0]);
            cfg->reduce_num_blocks = prop.maxGridSize[0];
        }
    }
    CUDACHECK(cudaStreamCreate(&ucc_mc_cuda.stream));
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_finalize()
{
    CUDACHECK(cudaStreamDestroy(ucc_mc_cuda.stream));
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

static ucc_status_t ucc_mc_cuda_mem_query(const void *ptr,
                                          size_t length,
                                          ucc_mem_attr_t *mem_attr)
{
    struct cudaPointerAttributes attr;
    cudaError_t                  st;
    CUresult                     cu_err;
    ucc_memory_type_t            mem_type;
    void                         *base_address;
    size_t                       alloc_length;

    if (!(mem_attr->field_mask & (UCC_MEM_ATTR_FIELD_MEM_TYPE     |
                                  UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                                  UCC_MEM_ATTR_FIELD_ALLOC_LENGTH))) {
        return UCC_OK;
    }

    if (ptr == 0) {
        mem_type = UCC_MEMORY_TYPE_HOST;
    } else {
        if (mem_attr->field_mask & UCC_MEM_ATTR_FIELD_MEM_TYPE) {
            st = cudaPointerGetAttributes(&attr, ptr);
            if (st != cudaSuccess) {
                cudaGetLastError();
                return UCC_ERR_NOT_SUPPORTED;
            }
#if CUDART_VERSION >= 10000
            switch (attr.type) {
            case cudaMemoryTypeHost:
                mem_type = UCC_MEMORY_TYPE_HOST;
                break;
            case cudaMemoryTypeDevice:
                mem_type = UCC_MEMORY_TYPE_CUDA;
                break;
            case cudaMemoryTypeManaged:
                mem_type = UCC_MEMORY_TYPE_CUDA_MANAGED;
                break;
            default:
                return UCC_ERR_NOT_SUPPORTED;
            }
#else
            if (attr.memoryType == cudaMemoryTypeDevice) {
                if (attr.isManaged) {
                    mem_type = UCC_MEMORY_TYPE_CUDA_MANAGED;
                } else {
                    mem_type = UCC_MEMORY_TYPE_CUDA;
                }
            }
            else if (attr.memoryType == cudaMemoryTypeHost) {
                mem_type = UCC_MEMORY_TYPE_HOST;
            } else {
                return UCC_ERR_NOT_SUPPORTED;
            }
#endif
            mem_attr->mem_type = mem_type;
        }

        if (mem_attr->field_mask & (UCC_MEM_ATTR_FIELD_ALLOC_LENGTH |
                                    UCC_MEM_ATTR_FIELD_BASE_ADDRESS)) {
            cu_err = cuMemGetAddressRange((CUdeviceptr*)&base_address,
                    &alloc_length, (CUdeviceptr)ptr);
            if (cu_err != CUDA_SUCCESS) {
                mc_error(&ucc_mc_cuda.super,
                         "cuMemGetAddressRange(%p) error: %d(%s)",
                          ptr, cu_err, cudaGetErrorString(st));
                return UCC_ERR_NOT_SUPPORTED;
            }

            mem_attr->base_address = base_address;
            mem_attr->alloc_length = alloc_length;
        }
    }

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
    .super.ops.mem_query = ucc_mc_cuda_mem_query,
    .super.ops.mem_alloc = ucc_mc_cuda_mem_alloc,
    .super.ops.mem_free  = ucc_mc_cuda_mem_free,
    .super.ops.reduce    = ucc_mc_cuda_reduce,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cuda.super.config_table,
                                &ucc_config_global_list);
