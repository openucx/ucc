/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

    {"REDUCE_NUM_BLOCKS", "auto",
     "Number of thread blocks to use for reduction",
     ucc_offsetof(ucc_mc_cuda_config_t, reduce_num_blocks),
     UCC_CONFIG_TYPE_ULUNITS},

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
    ucc_mc_cuda_config_t *cfg = MC_CUDA_CONFIG;
    struct cudaDeviceProp prop;
    int device, num_devices, driver_ver;
    cudaError_t cuda_st;

    ucc_mc_cuda.stream             = NULL;
    ucc_mc_cuda.stream_initialized = 0;
    ucc_strncpy_safe(ucc_mc_cuda.super.config->log_component.name,
                     ucc_mc_cuda.super.super.name,
                     sizeof(ucc_mc_cuda.super.config->log_component.name));
    ucc_mc_cuda.thread_mode = mc_params->thread_mode;
    cuda_st = cudaGetDeviceCount(&num_devices);
    if ((cuda_st != cudaSuccess) || (num_devices == 0)) {
        mc_info(&ucc_mc_cuda.super, "cuda devices are not found");
        return UCC_ERR_NO_RESOURCE;
    }
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    cfg->reduce_num_threads = prop.maxThreadsPerBlock;
    if (cfg->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        if (prop.maxGridSize[0] < cfg->reduce_num_blocks) {
            mc_warn(&ucc_mc_cuda.super, "number of blocks is too large, "
                    "max supported %d", prop.maxGridSize[0]);
            cfg->reduce_num_blocks = prop.maxGridSize[0];
        }
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
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_alloc(ucc_mc_buffer_header_t **h_ptr,
                                          size_t                   size)
{
    cudaError_t             st;
    ucc_mc_buffer_header_t *h =
        ucc_malloc(sizeof(ucc_mc_buffer_header_t), "mc cuda");
    if (ucc_unlikely(!h)) {
        mc_error(&ucc_mc_cuda.super, "failed to allocate %zd bytes",
                 sizeof(ucc_mc_buffer_header_t));
        return UCC_ERR_NO_MEMORY;
    }
    st = cudaMalloc(&h->addr, size);
    if (ucc_unlikely(st != cudaSuccess)) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to allocate %zd bytes, "
                 "cuda error %d(%s)",
                 size, st, cudaGetErrorString(st));
        ucc_free(h);
        return UCC_ERR_NO_MEMORY;
    }
    h->from_pool = 0;
    h->mt        = UCC_MEMORY_TYPE_CUDA;
    *h_ptr       = h;
    mc_trace(&ucc_mc_cuda.super, "allocated %ld bytes with cudaMalloc", size);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_mem_pool_alloc(ucc_mc_buffer_header_t **h_ptr,
                                               size_t                   size)
{
    ucc_mc_buffer_header_t *h = NULL;
    if (size <= MC_CUDA_CONFIG->mpool_elem_size) {
        h = (ucc_mc_buffer_header_t *)ucc_mpool_get(&ucc_mc_cuda.mpool);
    }
    if (!h) {
        // Slow path
        return ucc_mc_cuda_mem_alloc(h_ptr, size);
    }
    if (ucc_unlikely(!h->addr)){
        return UCC_ERR_NO_MEMORY;
    }
    *h_ptr = h;
    mc_trace(&ucc_mc_cuda.super, "allocated %ld bytes from cuda mpool", size);
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_chunk_alloc(ucc_mpool_t *mp, //NOLINT
                                            size_t *size_p,
                                            void **chunk_p)
{
    *chunk_p = ucc_malloc(*size_p, "mc cuda");
    if (!*chunk_p) {
        mc_error(&ucc_mc_cuda.super, "failed to allocate %zd bytes", *size_p);
        return UCC_ERR_NO_MEMORY;
    }

    return UCC_OK;
}

static void ucc_mc_cuda_chunk_init(ucc_mpool_t *mp, //NOLINT
                                   void *obj, void *chunk) //NOLINT
{
    ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *)obj;
    cudaError_t st = cudaMalloc(&h->addr, MC_CUDA_CONFIG->mpool_elem_size);
    if (st != cudaSuccess) {
        // h->addr will be 0 so ucc_mc_cuda_mem_alloc_pool function will
        // return UCC_ERR_NO_MEMORY. As such mc_error message is suffice.
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to allocate %zd bytes, "
                 "cuda error %d(%s)",
                 MC_CUDA_CONFIG->mpool_elem_size, st, cudaGetErrorString(st));
    }
    h->from_pool = 1;
    h->mt        = UCC_MEMORY_TYPE_CUDA;
}

static void ucc_mc_cuda_chunk_release(ucc_mpool_t *mp, void *chunk) //NOLINT
{
    ucc_free(chunk);
}

static void ucc_mc_cuda_chunk_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *)obj;
    cudaError_t             st;
    st = cudaFree(h->addr);
    if (st != cudaSuccess) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to free mem at %p, "
                 "cuda error %d(%s)",
                 obj, st, cudaGetErrorString(st));
    }
}

static ucc_mpool_ops_t ucc_mc_ops = {.chunk_alloc   = ucc_mc_cuda_chunk_alloc,
                                     .chunk_release = ucc_mc_cuda_chunk_release,
                                     .obj_init      = ucc_mc_cuda_chunk_init,
                                     .obj_cleanup = ucc_mc_cuda_chunk_cleanup};

static ucc_status_t ucc_mc_cuda_mem_free(ucc_mc_buffer_header_t *h_ptr)
{
    cudaError_t st;
    st = cudaFree(h_ptr->addr);
    if (ucc_unlikely(st != cudaSuccess)) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to free mem at %p, "
                 "cuda error %d(%s)",
                 h_ptr->addr, st, cudaGetErrorString(st));
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
                                     size_t                   size)
{
    // lock assures single mpool initiation when multiple threads concurrently execute
    // different collective operations thus concurrently entering init function.
    ucc_spin_lock(&ucc_mc_cuda.init_spinlock);

    if (MC_CUDA_CONFIG->mpool_max_elems == 0) {
        ucc_mc_cuda.super.ops.mem_alloc = ucc_mc_cuda_mem_alloc;
        ucc_mc_cuda.super.ops.mem_free  = ucc_mc_cuda_mem_free;
        ucc_spin_unlock(&ucc_mc_cuda.init_spinlock);
        return ucc_mc_cuda_mem_alloc(h_ptr, size);
    }

    if (!ucc_mc_cuda.mpool_init_flag) {
        ucc_status_t status = ucc_mpool_init(
            &ucc_mc_cuda.mpool, 0, sizeof(ucc_mc_buffer_header_t), 0,
            UCC_CACHE_LINE_SIZE, 1, MC_CUDA_CONFIG->mpool_max_elems,
            &ucc_mc_ops, ucc_mc_cuda.thread_mode, "mc cuda mpool buffers");
        if (status != UCC_OK) {
            ucc_spin_unlock(&ucc_mc_cuda.init_spinlock);
            return status;
        }
        ucc_mc_cuda.super.ops.mem_alloc = ucc_mc_cuda_mem_pool_alloc;
        ucc_mc_cuda.mpool_init_flag     = 1;
    }
    ucc_spin_unlock(&ucc_mc_cuda.init_spinlock);
    return ucc_mc_cuda_mem_pool_alloc(h_ptr, size);
}

static ucc_status_t ucc_mc_cuda_memcpy(void *dst, const void *src, size_t len,
                                       ucc_memory_type_t dst_mem,
                                       ucc_memory_type_t src_mem)
{
    cudaError_t    st;
    ucc_assert(dst_mem == UCC_MEMORY_TYPE_CUDA ||
               src_mem == UCC_MEMORY_TYPE_CUDA);

    UCC_MC_CUDA_INIT_STREAM();
    st = cudaMemcpyAsync(dst, src, len, cudaMemcpyDefault, ucc_mc_cuda.stream);
    if (ucc_unlikely(st != cudaSuccess)) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to launch cudaMemcpyAsync,  dst %p, src %p, len %zd "
                 "cuda error %d(%s)",
                 dst, src, len, st, cudaGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    st = cudaStreamSynchronize(ucc_mc_cuda.stream);
    if (ucc_unlikely(st != cudaSuccess)) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to synchronize mc_cuda.stream "
                 "cuda error %d(%s)",
                 st, cudaGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
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
                     "cuMemGetAddressRange(%p) error: %d(%s)",
                      ptr, cu_err, cudaGetErrorString(st));
            return UCC_ERR_NOT_SUPPORTED;
        }

        mem_attr->base_address = base_address;
        mem_attr->alloc_length = alloc_length;
    }

    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_finalize()
{
    if (ucc_mc_cuda.stream != NULL) {
        CUDA_CHECK(cudaStreamDestroy(ucc_mc_cuda.stream));
        ucc_mc_cuda.stream = NULL;
    }
    if (ucc_mc_cuda.mpool_init_flag) {
        ucc_mpool_cleanup(&ucc_mc_cuda.mpool, 1);
        ucc_mc_cuda.mpool_init_flag     = 0;
        ucc_mc_cuda.super.ops.mem_alloc = ucc_mc_cuda_mem_pool_alloc_with_init;
    }
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
    .super.ops.reduce             = ucc_mc_cuda_reduce,
    .super.ops.reduce_multi       = ucc_mc_cuda_reduce_multi,
    .super.ops.reduce_multi_alpha = ucc_mc_cuda_reduce_multi_alpha,
    .super.ops.memcpy             = ucc_mc_cuda_memcpy,
    .super.ops.flush              = ucc_mc_cuda_flush_not_supported,
    .super.config_table =
        {
            .name   = "CUDA memory component",
            .prefix = "MC_CUDA_",
            .table  = ucc_mc_cuda_config_table,
            .size   = sizeof(ucc_mc_cuda_config_t),
        },
    .mpool_init_flag               = 0,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cuda.super.config_table,
                                &ucc_config_global_list);
