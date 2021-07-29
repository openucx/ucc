/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "mc_cuda.h"
#include "utils/ucc_malloc.h"
#include <cuda_runtime.h>
#include <cuda.h>

static const char *stream_task_modes[] = {
    [UCC_MC_CUDA_TASK_KERNEL]  = "kernel",
    [UCC_MC_CUDA_TASK_MEM_OPS] = "driver",
    [UCC_MC_CUDA_TASK_AUTO]    = "auto",
    [UCC_MC_CUDA_TASK_LAST]    = NULL
};

static const char *task_stream_types[] = {
    [UCC_MC_CUDA_USER_STREAM]      = "user",
    [UCC_MC_CUDA_INTERNAL_STREAM]  = "ucc",
    [UCC_MC_CUDA_TASK_STREAM_LAST] = NULL
};

static ucc_config_field_t ucc_mc_cuda_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_mc_cuda_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_mc_config_table)},

    {"REDUCE_NUM_BLOCKS", "auto",
     "Number of thread blocks to use for reduction",
     ucc_offsetof(ucc_mc_cuda_config_t, reduce_num_blocks),
     UCC_CONFIG_TYPE_ULUNITS},

    {"STREAM_TASK_MODE", "auto",
     "Mechanism to create stream dependency\n"
     "kernel - use waiting kernel\n"
     "driver - use driver MEM_OPS\n"
     "auto   - runtime automatically chooses best one",
     ucc_offsetof(ucc_mc_cuda_config_t, strm_task_mode),
     UCC_CONFIG_TYPE_ENUM(stream_task_modes)},

    {"TASK_STREAM", "user",
     "Stream for cuda task\n"
     "user - user stream provided in execution engine context\n"
     "ucc  - ucc library internal stream",
     ucc_offsetof(ucc_mc_cuda_config_t, task_strm_type),
     UCC_CONFIG_TYPE_ENUM(task_stream_types)},

    {"STREAM_BLOCKING_WAIT", "1",
     "Stream is blocked until collective operation is done",
     ucc_offsetof(ucc_mc_cuda_config_t, stream_blocking_wait),
     UCC_CONFIG_TYPE_UINT},

    {"MPOOL_ELEM_SIZE", "1Mb", "The size of each element in mc cuda mpool",
     ucc_offsetof(ucc_mc_cuda_config_t, mpool_elem_size),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"MPOOL_MAX_ELEMS", "8", "The max amount of elements in mc cuda mpool",
     ucc_offsetof(ucc_mc_cuda_config_t, mpool_max_elems), UCC_CONFIG_TYPE_UINT},

    {NULL}

};

static ucc_status_t ucc_mc_cuda_stream_req_mpool_chunk_malloc(ucc_mpool_t *mp,
                                                              size_t *size_p,
                                                              void ** chunk_p)
{
    ucc_status_t status;

    status = CUDA_FUNC(cudaHostAlloc((void**)chunk_p, *size_p,
                       cudaHostAllocMapped));
    return status;
}

static void ucc_mc_cuda_stream_req_mpool_chunk_free(ucc_mpool_t *mp,
                                                    void *       chunk)
{
    cudaFreeHost(chunk);
}

static void ucc_mc_cuda_stream_req_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mc_cuda_stream_request_t *req = (ucc_mc_cuda_stream_request_t*) obj;

    CUDA_FUNC(cudaHostGetDevicePointer(
                  (void**)(&req->dev_status), (void *)&req->status, 0));
}

static ucc_mpool_ops_t ucc_mc_cuda_stream_req_mpool_ops = {
    .chunk_alloc   = ucc_mc_cuda_stream_req_mpool_chunk_malloc,
    .chunk_release = ucc_mc_cuda_stream_req_mpool_chunk_free,
    .obj_init      = ucc_mc_cuda_stream_req_init,
    .obj_cleanup   = NULL
};

static void ucc_mc_cuda_event_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mc_cuda_event_t *base = (ucc_mc_cuda_event_t *) obj;

    if (ucc_unlikely(
            cudaSuccess !=
            cudaEventCreateWithFlags(&base->event, cudaEventDisableTiming))) {
        mc_error(&ucc_mc_cuda.super, "cudaEventCreateWithFlags Failed");
    }
}

static void ucc_mc_cuda_event_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_mc_cuda_event_t *base = (ucc_mc_cuda_event_t *) obj;
    if (ucc_unlikely(cudaSuccess != cudaEventDestroy(base->event))) {
        mc_error(&ucc_mc_cuda.super, "cudaEventDestroy Failed");
    }
}

static ucc_mpool_ops_t ucc_mc_cuda_event_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_mc_cuda_event_init,
    .obj_cleanup   = ucc_mc_cuda_event_cleanup,
};

ucc_status_t ucc_mc_cuda_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 cudaStream_t stream);

static ucc_status_t ucc_mc_cuda_post_driver_stream_task(uint32_t *status,
                                                        int blocking_wait,
                                                        cudaStream_t stream)
{
    CUdeviceptr status_ptr  = (CUdeviceptr)status;

    if (blocking_wait) {
        CUDADRV_FUNC(cuStreamWriteValue32(stream, status_ptr,
                                          UCC_MC_CUDA_TASK_STARTED, 0));
        CUDADRV_FUNC(cuStreamWaitValue32(stream, status_ptr,
                                         UCC_MC_CUDA_TASK_COMPLETED,
                                         CU_STREAM_WAIT_VALUE_EQ));
    }
    CUDADRV_FUNC(cuStreamWriteValue32(stream, status_ptr,
                                      UCC_MC_CUDA_TASK_COMPLETED_ACK, 0));
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_init(const ucc_mc_params_t *mc_params)
{
    ucc_mc_cuda_config_t *cfg = MC_CUDA_CONFIG;
    struct cudaDeviceProp prop;
    ucc_status_t status;
    int device, num_devices, mem_ops_attr;
    CUdevice cu_dev;
    CUresult cu_st;
    cudaError_t cuda_st;
    const char *cu_err_st_str;

    ucc_mc_cuda.stream = NULL;
    ucc_strncpy_safe(ucc_mc_cuda.super.config->log_component.name,
                     ucc_mc_cuda.super.super.name,
                     sizeof(ucc_mc_cuda.super.config->log_component.name));
    ucc_mc_cuda.thread_mode = mc_params->thread_mode;
    cuda_st = cudaGetDeviceCount(&num_devices);
    if ((cuda_st != cudaSuccess) || (num_devices == 0)) {
        mc_info(&ucc_mc_cuda.super, "cuda devices are not found");
        return UCC_ERR_NO_RESOURCE;
    }
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

    /*create event pool */
    status = ucc_mpool_init(&ucc_mc_cuda.events, 0, sizeof(ucc_mc_cuda_event_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_mc_cuda_event_mpool_ops, UCC_THREAD_MULTIPLE,
                            "CUDA Event Objects");
    if (status != UCC_OK) {
        mc_error(&ucc_mc_cuda.super, "Error to create event pool");
        return status;
    }

    /* create request pool */
    status = ucc_mpool_init(
        &ucc_mc_cuda.strm_reqs, 0, sizeof(ucc_mc_cuda_stream_request_t), 0,
        UCC_CACHE_LINE_SIZE, 16, UINT_MAX, &ucc_mc_cuda_stream_req_mpool_ops,
        UCC_THREAD_MULTIPLE, "CUDA Event Objects");
    if (status != UCC_OK) {
        mc_error(&ucc_mc_cuda.super, "Error to create event pool");
        return status;
    }

    if (cfg->strm_task_mode == UCC_MC_CUDA_TASK_KERNEL) {
        ucc_mc_cuda.strm_task_mode = UCC_MC_CUDA_TASK_KERNEL;
        ucc_mc_cuda.post_strm_task = ucc_mc_cuda_post_kernel_stream_task;
    } else {
        ucc_mc_cuda.strm_task_mode = UCC_MC_CUDA_TASK_MEM_OPS;
        ucc_mc_cuda.post_strm_task = ucc_mc_cuda_post_driver_stream_task;

        cu_st = cuCtxGetDevice(&cu_dev);
        if (cu_st != CUDA_SUCCESS){
            cuGetErrorString(cu_st, &cu_err_st_str);
            mc_debug(&ucc_mc_cuda.super, "cuCtxGetDevice() failed: %s",
                     cu_err_st_str);
            mem_ops_attr = 0;
        } else {
            CUDADRV_FUNC(cuDeviceGetAttribute(&mem_ops_attr,
                        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                        cu_dev));
        }

        if (cfg->strm_task_mode == UCC_MC_CUDA_TASK_AUTO) {
            if (mem_ops_attr == 0) {
                mc_info(&ucc_mc_cuda.super,
                        "CUDA MEM OPS are not supported or disabled");
                ucc_mc_cuda.strm_task_mode = UCC_MC_CUDA_TASK_KERNEL;
                ucc_mc_cuda.post_strm_task = ucc_mc_cuda_post_kernel_stream_task;
            }
        } else if (mem_ops_attr == 0) {
            mc_error(&ucc_mc_cuda.super,
                     "CUDA MEM OPS are not supported or disabled");
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    ucc_mc_cuda.task_strm_type = cfg->task_strm_type;
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


ucc_status_t ucc_ee_cuda_task_post(void *ee_stream, void **ee_req)
{
    ucc_mc_cuda_stream_request_t *req;
    ucc_mc_cuda_event_t *cuda_event;
    ucc_status_t status;
    ucc_mc_cuda_config_t *cfg = MC_CUDA_CONFIG;

    UCC_MC_CUDA_INIT_STREAM();
    req = ucc_mpool_get(&ucc_mc_cuda.strm_reqs);
    ucc_assert(req);
    req->status = UCC_MC_CUDA_TASK_POSTED;
    req->stream = (cudaStream_t)ee_stream;

    if (ucc_mc_cuda.task_strm_type == UCC_MC_CUDA_USER_STREAM) {
        status = ucc_mc_cuda.post_strm_task(req->dev_status,
                                            cfg->stream_blocking_wait,
                                            req->stream);
        if (ucc_unlikely(status != UCC_OK)) {
            goto free_req;
        }
    } else {
        cuda_event = ucc_mpool_get(&ucc_mc_cuda.events);
        ucc_assert(cuda_event);
        CUDACHECK(cudaEventRecord(cuda_event->event, req->stream));
        CUDACHECK(cudaStreamWaitEvent(ucc_mc_cuda.stream, cuda_event->event, 0));
        status = ucc_mc_cuda.post_strm_task(req->dev_status,
                                            cfg->stream_blocking_wait,
                                            ucc_mc_cuda.stream);
        if (ucc_unlikely(status != UCC_OK)) {
            goto free_event;
        }
        CUDACHECK(cudaEventRecord(cuda_event->event, ucc_mc_cuda.stream));
        CUDACHECK(cudaStreamWaitEvent(req->stream, cuda_event->event, 0));
        ucc_mpool_put(cuda_event);
    }

    *ee_req = (void *) req;

    mc_info(&ucc_mc_cuda.super, "CUDA stream task posted on \"%s\" stream. req:%p",
            task_stream_types[ucc_mc_cuda.task_strm_type], req);

    return UCC_OK;

free_event:
    ucc_mpool_put(cuda_event);
free_req:
    ucc_mpool_put(req);
    return status;
}

ucc_status_t ucc_ee_cuda_task_query(void *ee_req)
{
    ucc_mc_cuda_stream_request_t *req = ee_req;

    /* ee task might be only in POSTED, STARTED or COMPLETED_ACK state
       COMPLETED state is used by ucc_ee_cuda_task_end function to request
       stream unblock*/
    ucc_assert(req->status != UCC_MC_CUDA_TASK_COMPLETED);
    if (req->status == UCC_MC_CUDA_TASK_POSTED) {
        return UCC_INPROGRESS;
    }
    mc_info(&ucc_mc_cuda.super, "CUDA stream task started. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ee_cuda_task_end(void *ee_req)
{
    ucc_mc_cuda_stream_request_t *req = ee_req;
    volatile ucc_mc_task_status_t *st = &req->status;

    /* can be safely ended only if it's in STARTED or COMPLETED_ACK state */
    ucc_assert((*st != UCC_MC_CUDA_TASK_POSTED) &&
               (*st != UCC_MC_CUDA_TASK_COMPLETED));
    if (*st == UCC_MC_CUDA_TASK_STARTED) {
        *st = UCC_MC_CUDA_TASK_COMPLETED;
        while(*st != UCC_MC_CUDA_TASK_COMPLETED_ACK) { }
    }
    ucc_mpool_put(req);
    mc_info(&ucc_mc_cuda.super, "CUDA stream task done. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ee_cuda_create_event(void **event)
{
    ucc_mc_cuda_event_t *cuda_event;

    cuda_event = ucc_mpool_get(&ucc_mc_cuda.events);
    ucc_assert(cuda_event);
    *event = cuda_event;
    return UCC_OK;
}

ucc_status_t ucc_ee_cuda_destroy_event(void *event)
{
    ucc_mc_cuda_event_t *cuda_event = event;

    ucc_mpool_put(cuda_event);
    return UCC_OK;
}

ucc_status_t ucc_ee_cuda_event_post(void *ee_context, void *event)
{
    cudaStream_t stream = (cudaStream_t )ee_context;
    ucc_mc_cuda_event_t *cuda_event = event;

    CUDACHECK(cudaEventRecord(cuda_event->event, stream));
    return UCC_OK;
}

ucc_status_t ucc_ee_cuda_event_test(void *event)
{
    cudaError_t cu_err;
    ucc_mc_cuda_event_t *cuda_event = event;

    cu_err = cudaEventQuery(cuda_event->event);
    return cuda_error_to_ucc_status(cu_err);
}

static ucc_status_t ucc_mc_cuda_finalize()
{
    if (ucc_mc_cuda.stream != NULL) {
        CUDACHECK(cudaStreamDestroy(ucc_mc_cuda.stream));
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
    .super.super.name       = "cuda mc",
    .super.ref_cnt          = 0,
    .super.ee_type          = UCC_EE_CUDA_STREAM,
    .super.type             = UCC_MEMORY_TYPE_CUDA,
    .super.init             = ucc_mc_cuda_init,
    .super.get_attr         = ucc_mc_cuda_get_attr,
    .super.finalize         = ucc_mc_cuda_finalize,
    .super.ops.mem_query    = ucc_mc_cuda_mem_query,
    .super.ops.mem_alloc    = ucc_mc_cuda_mem_pool_alloc_with_init,
    .super.ops.mem_free     = ucc_mc_cuda_mem_pool_free,
    .super.ops.reduce       = ucc_mc_cuda_reduce,
    .super.ops.reduce_multi = ucc_mc_cuda_reduce_multi,
    .super.ops.memcpy       = ucc_mc_cuda_memcpy,
    .super.config_table =
        {
            .name   = "CUDA memory component",
            .prefix = "MC_CUDA_",
            .table  = ucc_mc_cuda_config_table,
            .size   = sizeof(ucc_mc_cuda_config_t),
        },
    .super.ee_ops.ee_task_post     = ucc_ee_cuda_task_post,
    .super.ee_ops.ee_task_query    = ucc_ee_cuda_task_query,
    .super.ee_ops.ee_task_end      = ucc_ee_cuda_task_end,
    .super.ee_ops.ee_create_event  = ucc_ee_cuda_create_event,
    .super.ee_ops.ee_destroy_event = ucc_ee_cuda_destroy_event,
    .super.ee_ops.ee_event_post    = ucc_ee_cuda_event_post,
    .super.ee_ops.ee_event_test    = ucc_ee_cuda_event_test,
    .mpool_init_flag               = 0,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cuda.super.config_table,
                                &ucc_config_global_list);
