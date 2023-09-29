/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "mc_cuda.h"
#include "utils/ucc_malloc.h"
#include "utils/arch/cpu.h"
#include <cuda_runtime.h>
#include <cuda.h>

static ucc_config_field_t ucc_mc_cuda_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_cuda_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {"MPOOL_ELEM_SIZE", "1Mb", "The size of each element in mc cuda mpool",
     ucc_offsetof(ucc_mc_cuda_config_t, mpool_elem_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"MPOOL_MAX_ELEMS", "8", "The max amount of elements in mc cuda mpool",
     ucc_offsetof(ucc_mc_cuda_config_t, mpool_max_elems), UCC_CONFIG_TYPE_UINT},

    {NULL}

};

static ucc_status_t ucc_mc_cuda_flush_not_supported()
{
    mc_error(&ucc_mc_cuda.super, "consistency api is not supported");
    return UCC_ERR_NOT_SUPPORTED;
}

#if CUDA_VERSION >= 11030
static ucc_status_t ucc_mc_cuda_flush_no_op()
{
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_flush_to_owner()
{
    CUDADRV_FUNC(cuFlushGPUDirectRDMAWrites(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX,
                                            CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER));
    return UCC_OK;
}
#endif

static ucc_status_t ucc_mc_cuda_init(const ucc_mc_params_t *mc_params)
{
    int         num_devices, driver_ver;
    cudaError_t cuda_st;

    ucc_mc_cuda_config = ucc_derived_of(ucc_mc_cuda.super.config,
                                        ucc_mc_cuda_config_t);
    ucc_strncpy_safe(ucc_mc_cuda.super.config->log_component.name,
                     ucc_mc_cuda.super.super.name,
                     sizeof(ucc_mc_cuda.super.config->log_component.name));
    ucc_mc_cuda.thread_mode = mc_params->thread_mode;
    cuda_st = cudaGetDeviceCount(&num_devices);
    if ((cuda_st != cudaSuccess) || (num_devices == 0)) {
        mc_debug(&ucc_mc_cuda.super, "cuda devices are not found");
        return UCC_ERR_NO_RESOURCE;
    }
    CUDADRV_FUNC(cuDriverGetVersion(&driver_ver));
    mc_debug(&ucc_mc_cuda.super, "driver version %d", driver_ver);

#if CUDA_VERSION >= 11030
    if (driver_ver >= 11030) {
        CUdevice    cu_dev;
        CUresult    cu_st;
        int         attr;
        const char *cu_err_st_str;

        cu_st = cuCtxGetDevice(&cu_dev);
        if (cu_st != CUDA_SUCCESS){
            cuGetErrorString(cu_st, &cu_err_st_str);
            mc_debug(&ucc_mc_cuda.super, "cuCtxGetDevice() failed: %s",
                     cu_err_st_str);
        } else {
            attr = 0;
            CUDADRV_FUNC(cuDeviceGetAttribute(&attr,
                         CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS,
                         cu_dev));
            if (attr & CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST) {
                CUDADRV_FUNC(cuDeviceGetAttribute(&attr,
                             CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING,
                             cu_dev));
                if (CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER > attr) {
                    ucc_mc_cuda.super.ops.flush = ucc_mc_cuda_flush_to_owner;
                } else {
                    ucc_mc_cuda.super.ops.flush = ucc_mc_cuda_flush_no_op;
                }
            } else {
                mc_debug(&ucc_mc_cuda.super, "consistency api is not supported");
            }
        }
    } else {
        mc_debug(&ucc_mc_cuda.super,
                 "cuFlushGPUDirectRDMAWrites is not supported "
                 "with driver version %d", driver_ver);
    }
#endif
    ucc_mc_cuda.resources_hash = kh_init(ucc_mc_cuda_resources_hash);
    // lock assures single mpool initiation when multiple threads concurrently execute
    // different collective operations thus concurrently entering init function.
    ucc_spinlock_init(&ucc_mc_cuda.init_spinlock, 0);

    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_get_attr(ucc_mc_attr_t *mc_attr)
{
    if (mc_attr->field_mask & UCC_MC_ATTR_FIELD_THREAD_MODE) {
        mc_attr->thread_mode = ucc_mc_cuda.thread_mode;
    }
    if (mc_attr->field_mask & UCC_MC_ATTR_FIELD_FAST_ALLOC_SIZE) {
        if (MC_CUDA_CONFIG->mpool_max_elems > 0) {
            mc_attr->fast_alloc_size = MC_CUDA_CONFIG->mpool_elem_size;
        } else {
            mc_attr->fast_alloc_size = 0;
        }
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_alloc(ucc_mc_buffer_header_t **h_ptr,
                                          size_t                   size,
                                          ucc_memory_type_t        mt)
{
    cudaError_t             st;
    ucc_mc_buffer_header_t *h;

    h = ucc_malloc(sizeof(ucc_mc_buffer_header_t), "mc cuda");
    if (ucc_unlikely(!h)) {
        mc_error(&ucc_mc_cuda.super, "failed to allocate %zd bytes",
                 sizeof(ucc_mc_buffer_header_t));
        return UCC_ERR_NO_MEMORY;
    }
    st = (mt == UCC_MEMORY_TYPE_CUDA) ? cudaMalloc(&h->addr, size) :
                                        cudaMallocManaged(&h->addr, size,
                                                          cudaMemAttachGlobal);
    if (ucc_unlikely(st != cudaSuccess)) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super, "failed to allocate %zd bytes, "
                 "cuda error %d(%s)",
                 size, st, cudaGetErrorString(st));
        ucc_free(h);
        return UCC_ERR_NO_MEMORY;
    }

    h->from_pool = 0;
    h->mt        = UCC_MEMORY_TYPE_CUDA;
    *h_ptr       = h;
    mc_trace(&ucc_mc_cuda.super, "allocated %ld bytes with %s", size,
             ucc_memory_type_names[mt]);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_pool_alloc(ucc_mc_buffer_header_t **h_ptr,
                                               size_t                   size,
                                               ucc_memory_type_t        mt)
{
    ucc_mc_buffer_header_t  *h = NULL;
    ucc_mc_cuda_resources_t *resources;
    ucc_status_t             status;

    if ((size <= MC_CUDA_CONFIG->mpool_elem_size) &&
        (mt != UCC_MEMORY_TYPE_CUDA_MANAGED)) {
        status = ucc_mc_cuda_get_resources(&resources);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }

        h = (ucc_mc_buffer_header_t *)ucc_mpool_get(&resources->scratch_mpool);
    }

    if (!h) {
        // Slow path
        return ucc_mc_cuda_mem_alloc(h_ptr, size, mt);
    }

    if (ucc_unlikely(!h->addr)){
        return UCC_ERR_NO_MEMORY;
    }
    *h_ptr = h;
    mc_trace(&ucc_mc_cuda.super, "allocated %ld bytes from cuda mpool", size);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_free(ucc_mc_buffer_header_t *h_ptr)
{
    cudaError_t st;
    st = cudaFree(h_ptr->addr);
    if (ucc_unlikely(st != cudaSuccess)) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to free mem at %p, "
                 "cuda error %d(%s)",
                 h_ptr, st, cudaGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    ucc_free(h_ptr);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_pool_free(ucc_mc_buffer_header_t *h_ptr)
{
    if (!h_ptr->from_pool) {
        return ucc_mc_cuda_mem_free(h_ptr);
    }
    ucc_mpool_put(h_ptr);
    return UCC_OK;
}

static ucc_status_t
ucc_mc_cuda_mem_pool_alloc_with_init(ucc_mc_buffer_header_t **h_ptr,
                                     size_t size,
                                     ucc_memory_type_t mt)
{
    if (MC_CUDA_CONFIG->mpool_max_elems == 0) {
        ucc_mc_cuda.super.ops.mem_alloc = ucc_mc_cuda_mem_alloc;
        ucc_mc_cuda.super.ops.mem_free  = ucc_mc_cuda_mem_free;
        return ucc_mc_cuda_mem_alloc(h_ptr, size, mt);
    } else {
        ucc_mc_cuda.super.ops.mem_alloc = ucc_mc_cuda_mem_pool_alloc;
        ucc_mc_cuda.super.ops.mem_free  = ucc_mc_cuda_mem_pool_free;
        return ucc_mc_cuda_mem_pool_alloc(h_ptr, size, mt);
    }
}

static ucc_status_t ucc_mc_cuda_memcpy(void *dst, const void *src, size_t len,
                                       ucc_memory_type_t dst_mem,
                                       ucc_memory_type_t src_mem)
{
    ucc_status_t status;
    ucc_mc_cuda_resources_t *resources;

    ucc_assert(dst_mem == UCC_MEMORY_TYPE_CUDA ||
               src_mem == UCC_MEMORY_TYPE_CUDA ||
               dst_mem == UCC_MEMORY_TYPE_CUDA_MANAGED ||
               src_mem == UCC_MEMORY_TYPE_CUDA_MANAGED);

    status = ucc_mc_cuda_get_resources(&resources);
    if (ucc_unlikely(status) != UCC_OK) {
        return status;
    }

    status = CUDA_FUNC(cudaMemcpyAsync(dst, src, len, cudaMemcpyDefault,
                                       resources->stream));
    if (ucc_unlikely(status != UCC_OK)) {
        mc_error(&ucc_mc_cuda.super,
                 "failed to launch cudaMemcpyAsync, dst %p, src %p, len %zd",
                 dst, src, len);
        return status;
    }

    status = CUDA_FUNC(cudaStreamSynchronize(resources->stream));

    return status;
}

ucc_status_t ucc_mc_cuda_memset(void *ptr, int val, size_t len)
{
    ucc_status_t status;
    ucc_mc_cuda_resources_t *resources;

    status = ucc_mc_cuda_get_resources(&resources);
    if (ucc_unlikely(status) != UCC_OK) {
        return status;
    }

    status = CUDA_FUNC(cudaMemsetAsync(ptr, val, len, resources->stream));
    if (ucc_unlikely(status != UCC_OK)) {
        mc_error(&ucc_mc_cuda.super,
                 "failed to launch cudaMemsetAsync, dst %p, len %zd",
                 ptr, len);
        return status;
    }

    status = CUDA_FUNC(cudaStreamSynchronize(resources->stream));

    return status;
}

static ucc_status_t ucc_mc_cuda_mem_query(const void *ptr,
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
        } else if (attr.memoryType == cudaMemoryTypeHost) {
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
            mc_debug(&ucc_mc_cuda.super,
                     "cuMemGetAddressRange(%p) error: %d", ptr, cu_err);
            return UCC_ERR_NOT_SUPPORTED;
        }

        mem_attr->base_address = base_address;
        mem_attr->alloc_length = alloc_length;
    }

    return UCC_OK;
}

ucc_status_t ucc_mc_cuda_get_resources(ucc_mc_cuda_resources_t **resources)
{
    CUcontext cu_ctx;
    unsigned long long int cu_ctx_id;
    ucc_status_t status;

    status = CUDADRV_FUNC(cuCtxGetCurrent(&cu_ctx));
    if (ucc_unlikely(status != UCC_OK)) {
        mc_error(&ucc_mc_cuda.super, "failed to get current CUDA context");
        return status;
    }

#if CUDA_VERSION < 12000
    cu_ctx_id = 1;
#else
    status = CUDADRV_FUNC(cuCtxGetId(cu_ctx, &cu_ctx_id));
    if (ucc_unlikely(status != UCC_OK)) {
        mc_error(&ucc_mc_cuda.super, "failed to get currect CUDA context ID");
    }
#endif

    *resources = mc_cuda_resources_hash_get(ucc_mc_cuda.resources_hash,
                                            cu_ctx_id);
    if (ucc_unlikely(*resources == NULL)) {
        ucc_spin_lock(&ucc_mc_cuda.init_spinlock);
        *resources = mc_cuda_resources_hash_get(ucc_mc_cuda.resources_hash,
                                                cu_ctx_id);
        if (*resources == NULL) {
            *resources = ucc_malloc(sizeof(ucc_mc_cuda_resources_t),
                                    "mc cuda resources");
            if (*resources == NULL) {
                mc_error(&ucc_mc_cuda.super,
                         "failed to allocate %zd bytes for resources",
                         sizeof(ucc_mc_cuda_resources_t));
                ucc_spin_unlock(&ucc_mc_cuda.init_spinlock);
                return UCC_ERR_NO_MEMORY;
            }
            status = ucc_mc_cuda_resources_init(&ucc_mc_cuda.super,
                                                *resources);
            if (status != UCC_OK) {
                ucc_free(*resources);
                ucc_spin_unlock(&ucc_mc_cuda.init_spinlock);
                return status;
            }
            mc_cuda_resources_hash_put(ucc_mc_cuda.resources_hash, cu_ctx_id,
                                       *resources);
        }
        ucc_spin_unlock(&ucc_mc_cuda.init_spinlock);
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_finalize()
{
    ucc_mc_cuda_resources_t *resources;

    resources = mc_cuda_resources_hash_pop(ucc_mc_cuda.resources_hash);
    while (resources) {
        ucc_mc_cuda_resources_cleanup(resources);
        resources = mc_cuda_resources_hash_pop(ucc_mc_cuda.resources_hash);
    }

    ucc_mc_cuda.super.ops.mem_alloc = ucc_mc_cuda_mem_pool_alloc_with_init;
    ucc_spinlock_destroy(&ucc_mc_cuda.init_spinlock);
    return UCC_OK;
}

ucc_mc_cuda_t ucc_mc_cuda = {
    .super.super.name             = "cuda mc",
    .super.ref_cnt                = 0,
    .super.ee_type                = UCC_EE_CUDA_STREAM,
    .super.type                   = UCC_MEMORY_TYPE_CUDA,
    .super.init                   = ucc_mc_cuda_init,
    .super.get_attr               = ucc_mc_cuda_get_attr,
    .super.finalize               = ucc_mc_cuda_finalize,
    .super.ops.mem_query          = ucc_mc_cuda_mem_query,
    .super.ops.mem_alloc          = ucc_mc_cuda_mem_pool_alloc_with_init,
    .super.ops.mem_free           = ucc_mc_cuda_mem_pool_free,
    .super.ops.memcpy             = ucc_mc_cuda_memcpy,
    .super.ops.memset             = ucc_mc_cuda_memset,
    .super.ops.flush              = ucc_mc_cuda_flush_not_supported,
    .super.config_table =
        {
            .name   = "CUDA memory component",
            .prefix = "MC_CUDA_",
            .table  = ucc_mc_cuda_config_table,
            .size   = sizeof(ucc_mc_cuda_config_t),
        },
};

ucc_mc_cuda_config_t *ucc_mc_cuda_config;

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cuda.super.config_table,
                                &ucc_config_global_list);
