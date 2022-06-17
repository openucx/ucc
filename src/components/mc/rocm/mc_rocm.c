/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_rocm.h"
#include "utils/ucc_malloc.h"
#include "utils/arch/cpu.h"
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

static ucc_config_field_t ucc_mc_rocm_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_rocm_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {"REDUCE_NUM_BLOCKS", "auto",
     "Number of thread blocks to use for reduction",
     ucc_offsetof(ucc_mc_rocm_config_t, reduce_num_blocks),
     UCC_CONFIG_TYPE_ULUNITS},

    {"MPOOL_ELEM_SIZE", "1Mb", "The size of each element in mc rocm mpool",
     ucc_offsetof(ucc_mc_rocm_config_t, mpool_elem_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"MPOOL_MAX_ELEMS", "8", "The max amount of elements in mc rocm mpool",
     ucc_offsetof(ucc_mc_rocm_config_t, mpool_max_elems), UCC_CONFIG_TYPE_UINT},

    {NULL}
};


static ucc_status_t ucc_mc_rocm_init(const ucc_mc_params_t *mc_params)
{
    ucc_mc_rocm_config_t *cfg = MC_ROCM_CONFIG;
    struct hipDeviceProp_t prop;
    int device, num_devices;
    hipError_t rocm_st;

    ucc_mc_rocm.stream             = NULL;
    ucc_mc_rocm.stream_initialized = 0;
    ucc_strncpy_safe(ucc_mc_rocm.super.config->log_component.name,
                     ucc_mc_rocm.super.super.name,
                     sizeof(ucc_mc_rocm.super.config->log_component.name));
    ucc_mc_rocm.thread_mode = mc_params->thread_mode;
    rocm_st = hipGetDeviceCount(&num_devices);
    if ((rocm_st != hipSuccess) || (num_devices == 0)) {
        mc_info(&ucc_mc_rocm.super, "rocm devices are not found");
        return hip_error_to_ucc_status(rocm_st);
    }
    ROCMCHECK(hipGetDevice(&device));
    ROCMCHECK(hipGetDeviceProperties(&prop, device));
    cfg->reduce_num_threads = prop.maxThreadsPerBlock;
    if (cfg->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        if (prop.maxGridSize[0] < cfg->reduce_num_blocks) {
            mc_warn(&ucc_mc_rocm.super, "number of blocks is too large, "
                    "max supported %d", prop.maxGridSize[0]);
            cfg->reduce_num_blocks = prop.maxGridSize[0];
        }
    }

    // lock assures single mpool initiation when multiple threads concurrently execute
    // different collective operations thus concurrently entering init function.
    ucc_spinlock_init(&ucc_mc_rocm.init_spinlock, 0);

    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_get_attr(ucc_mc_attr_t *mc_attr)
{
    if (mc_attr->field_mask & UCC_MC_ATTR_FIELD_THREAD_MODE) {
        mc_attr->thread_mode = ucc_mc_rocm.thread_mode;
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_alloc(ucc_mc_buffer_header_t **h_ptr,
                                          size_t                   size)
{
    hipError_t st;

    ucc_mc_buffer_header_t *h = ucc_malloc(sizeof(ucc_mc_buffer_header_t), "mc rocm");
    if (ucc_unlikely(!h)) {
      mc_error(&ucc_mc_rocm.super, "failed to allocate %zd bytes",
               sizeof(ucc_mc_buffer_header_t));
    }
    st = hipMalloc(&h->addr, size);
    if (ucc_unlikely(st != hipSuccess)) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to allocate %zd bytes, "
                 "rocm error %d(%s)",
                 size, st, hipGetErrorString(st));
        ucc_free(h);
        return hip_error_to_ucc_status(st);
    }
    h->from_pool = 0;
    h->mt        = UCC_MEMORY_TYPE_ROCM;
    *h_ptr       = h;
    mc_trace(&ucc_mc_rocm.super, "allocated %ld bytes with hipMalloc", size);
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_pool_alloc(ucc_mc_buffer_header_t **h_ptr,
                                               size_t                   size)
{
    ucc_mc_buffer_header_t *h = NULL;

    if (size <= MC_ROCM_CONFIG->mpool_elem_size) {
        h = (ucc_mc_buffer_header_t *)ucc_mpool_get(&ucc_mc_rocm.mpool);
    }
    if (!h) {
        // Slow path
        return ucc_mc_rocm_mem_alloc(h_ptr, size);
    }
    if (ucc_unlikely(!h->addr)){
        return UCC_ERR_NO_MEMORY;
    }
    *h_ptr = h;
    mc_trace(&ucc_mc_rocm.super, "allocated %ld bytes from rocm mpool", size);
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_chunk_alloc(ucc_mpool_t *mp,
                                            size_t *size_p,
                                            void **chunk_p)
{
    *chunk_p = ucc_malloc(*size_p, "mc rocm");
    if (!*chunk_p) {
        mc_error(&ucc_mc_rocm.super, "failed to allocate %zd bytes", *size_p);
        return UCC_ERR_NO_MEMORY;
    }

    return UCC_OK;
}

static void ucc_mc_rocm_chunk_init(ucc_mpool_t *mp,
                                   void *obj, void *chunk)
{
    ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *)obj;
    hipError_t st             = hipMalloc(&h->addr, MC_ROCM_CONFIG->mpool_elem_size);

    if (st != hipSuccess) {
        // h->addr will be 0 so ucc_mc_rocm_mem_alloc_pool function will
        // return UCC_ERR_NO_MEMORY. As such mc_error message is suffice.
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to allocate %zd bytes, "
                 "rocm error %d(%s)",
                 MC_ROCM_CONFIG->mpool_elem_size, st, hipGetErrorString(st));
    }
    h->from_pool = 1;
    h->mt        = UCC_MEMORY_TYPE_ROCM;
}

static void ucc_mc_rocm_chunk_release(ucc_mpool_t *mp, void *chunk)
{
    ucc_free(chunk);
}

static void ucc_mc_rocm_chunk_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *)obj;
    hipError_t st;

    st = hipFree(h->addr);
    if (st != hipSuccess) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to free mem at %p, "
                 "rocm error %d(%s)",
                 obj, st, hipGetErrorString(st));
    }
}

static ucc_mpool_ops_t ucc_mc_ops = {.chunk_alloc   = ucc_mc_rocm_chunk_alloc,
                                     .chunk_release = ucc_mc_rocm_chunk_release,
                                     .obj_init      = ucc_mc_rocm_chunk_init,
                                     .obj_cleanup   = ucc_mc_rocm_chunk_cleanup};

static ucc_status_t ucc_mc_rocm_mem_free(ucc_mc_buffer_header_t *h_ptr)
{
    hipError_t st;

    st = hipFree(h_ptr->addr);
    if (ucc_unlikely(st != hipSuccess)) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to free mem at %p, "
                 "hip error %d(%s)",
                 h_ptr->addr, st, hipGetErrorString(st));
        return hip_error_to_ucc_status(st);
    }
    ucc_free(h_ptr);
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_pool_free(ucc_mc_buffer_header_t *h_ptr)
{
    if (!h_ptr->from_pool) {
        return ucc_mc_rocm_mem_free(h_ptr);
    }
    ucc_mpool_put(h_ptr);
    return UCC_OK;
}

static ucc_status_t
ucc_mc_rocm_mem_pool_alloc_with_init(ucc_mc_buffer_header_t **h_ptr,
                                     size_t                   size)
{
    // lock assures single mpool initiation when multiple threads concurrently execute
    // different collective operations thus concurrently entering init function.
    ucc_spin_lock(&ucc_mc_rocm.init_spinlock);

    if (MC_ROCM_CONFIG->mpool_max_elems == 0) {
        ucc_mc_rocm.super.ops.mem_alloc = ucc_mc_rocm_mem_alloc;
        ucc_mc_rocm.super.ops.mem_free  = ucc_mc_rocm_mem_free;
        ucc_spin_unlock(&ucc_mc_rocm.init_spinlock);
        return ucc_mc_rocm_mem_alloc(h_ptr, size);
    }

    if (!ucc_mc_rocm.mpool_init_flag) {
        ucc_status_t status = ucc_mpool_init(
            &ucc_mc_rocm.mpool, 0, sizeof(ucc_mc_buffer_header_t), 0,
            UCC_CACHE_LINE_SIZE, 1, MC_ROCM_CONFIG->mpool_max_elems,
            &ucc_mc_ops, ucc_mc_rocm.thread_mode, "mc rocm mpool buffers");
        if (status != UCC_OK) {
            ucc_spin_unlock(&ucc_mc_rocm.init_spinlock);
            return status;
        }
        ucc_mc_rocm.super.ops.mem_alloc = ucc_mc_rocm_mem_pool_alloc;
        ucc_mc_rocm.mpool_init_flag     = 1;
    }
    ucc_spin_unlock(&ucc_mc_rocm.init_spinlock);
    return ucc_mc_rocm_mem_pool_alloc(h_ptr, size);
}

static ucc_status_t ucc_mc_rocm_memcpy(void *dst, const void *src, size_t len,
                                       ucc_memory_type_t dst_mem,
                                       ucc_memory_type_t src_mem)
{
    hipError_t st;

    ucc_assert(dst_mem == UCC_MEMORY_TYPE_ROCM ||
               src_mem == UCC_MEMORY_TYPE_ROCM);

    UCC_MC_ROCM_INIT_STREAM();
    st = hipMemcpyAsync(dst, src, len, hipMemcpyDefault, ucc_mc_rocm.stream);
    if (ucc_unlikely(st != hipSuccess)) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to launch hipMemcpyAsync,  dst %p, src %p, len %zd "
                 "hip error %d(%s)",
                 dst, src, len, st, hipGetErrorString(st));
        return hip_error_to_ucc_status(st);
    }
    st = hipStreamSynchronize(ucc_mc_rocm.stream);
    if (ucc_unlikely(st != hipSuccess)) {
        hipGetLastError();
        mc_error(&ucc_mc_rocm.super,
                 "failed to synchronize mc_rocm.stream "
                 "hip error %d(%s)",
                 st, hipGetErrorString(st));
        return hip_error_to_ucc_status(st);
    }
    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_mem_query(const void *ptr,
                                          ucc_mem_attr_t *mem_attr)
{
    struct hipPointerAttribute_t attr;
    hipError_t                   st;
    ucc_memory_type_t            mem_type;
    void                        *base_address;
    size_t                       alloc_length;

    if (!(mem_attr->field_mask & (UCC_MEM_ATTR_FIELD_MEM_TYPE     |
                                  UCC_MEM_ATTR_FIELD_BASE_ADDRESS |
                                  UCC_MEM_ATTR_FIELD_ALLOC_LENGTH))) {
        return UCC_OK;
    }

    if (mem_attr->field_mask & UCC_MEM_ATTR_FIELD_MEM_TYPE) {
        st = hipPointerGetAttributes(&attr, ptr);
        if (st != hipSuccess) {
            hipGetLastError();
            return UCC_ERR_NOT_SUPPORTED;
        }
        switch (attr.memoryType) {
        case hipMemoryTypeHost:
            mem_type = (attr.isManaged ? UCC_MEMORY_TYPE_ROCM_MANAGED : UCC_MEMORY_TYPE_HOST);
            break;
        case hipMemoryTypeDevice:
            mem_type = UCC_MEMORY_TYPE_ROCM;
            break;
        default:
            return UCC_ERR_NOT_SUPPORTED;
        }
        mem_attr->mem_type = mem_type;
    }

    if (mem_attr->field_mask & (UCC_MEM_ATTR_FIELD_ALLOC_LENGTH |
                                UCC_MEM_ATTR_FIELD_BASE_ADDRESS)) {
      st = hipMemGetAddressRange((hipDeviceptr_t*)&base_address,
                                      &alloc_length, (hipDeviceptr_t)ptr);
      if (st != hipSuccess) {
        mc_error(&ucc_mc_rocm.super,
                 "hipMemGetAddressRange(%p) error: %d(%s)",
                 ptr, st, hipGetErrorString(st));
        return hip_error_to_ucc_status(st);
      }

      mem_attr->base_address = base_address;
      mem_attr->alloc_length = alloc_length;
    }

    return UCC_OK;
}

static ucc_status_t ucc_mc_rocm_finalize()
{
    if (ucc_mc_rocm.stream != NULL) {
        ROCMCHECK(hipStreamDestroy(ucc_mc_rocm.stream));
        ucc_mc_rocm.stream = NULL;
    }
    if (ucc_mc_rocm.mpool_init_flag) {
        ucc_mpool_cleanup(&ucc_mc_rocm.mpool, 1);
        ucc_mc_rocm.mpool_init_flag     = 0;
        ucc_mc_rocm.super.ops.mem_alloc = ucc_mc_rocm_mem_pool_alloc_with_init;
    }
    ucc_spinlock_destroy(&ucc_mc_rocm.init_spinlock);
    return UCC_OK;
}

ucc_mc_rocm_t ucc_mc_rocm = {
    .super.super.name             = "rocm mc",
    .super.ref_cnt                = 0,
    .super.ee_type                = UCC_EE_ROCM_STREAM,
    .super.type                   = UCC_MEMORY_TYPE_ROCM,
    .super.init                   = ucc_mc_rocm_init,
    .super.get_attr               = ucc_mc_rocm_get_attr,
    .super.finalize               = ucc_mc_rocm_finalize,
    .super.ops.mem_query          = ucc_mc_rocm_mem_query,
    .super.ops.mem_alloc          = ucc_mc_rocm_mem_pool_alloc_with_init,
    .super.ops.mem_free           = ucc_mc_rocm_mem_pool_free,
    .super.ops.memcpy             = ucc_mc_rocm_memcpy,
    .super.ops.reduce             = ucc_mc_rocm_reduce,
    .super.ops.reduce_multi       = ucc_mc_rocm_reduce_multi,
    .super.ops.reduce_multi_alpha = ucc_mc_rocm_reduce_multi_alpha,
    .super.config_table =
        {
            .name   = "ROCM memory component",
            .prefix = "MC_ROCM_",
            .table  = ucc_mc_rocm_config_table,
            .size   = sizeof(ucc_mc_rocm_config_t),
        },
    .mpool_init_flag               = 0,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_rocm.super.config_table,
                                &ucc_config_global_list);
