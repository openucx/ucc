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
};

static const char *task_stream_types[] = {
    [UCC_MC_CUDA_USER_STREAM]     = "user",
    [UCC_MC_CUDA_INTERNAL_STREAM] = "ucc",
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

    {NULL}
};

static ucs_status_t
ucc_mc_cuda_stream_req_mpool_chunk_malloc(ucs_mpool_t *mp, size_t *size_p, void **chunk_p)
{
    ucc_status_t status;

    status = CUDA_FUNC(cudaHostAlloc((void**)chunk_p, *size_p, cudaHostAllocMapped));
    if(status != UCC_OK)  {
        return UCS_ERR_NO_MEMORY;
    }
    return UCS_OK;
}

static void ucc_mc_cuda_stream_req_mpool_chunk_free(ucs_mpool_t *mp, void *chunk)
{
    cudaFreeHost(chunk);
}

static void ucc_mc_cuda_stream_req_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mc_cuda_stream_request_t *req = (ucc_mc_cuda_stream_request_t*) obj;

    CUDA_FUNC(cudaHostGetDevicePointer(
                  (void**)(&req->dev_status), (void *)&req->status, 0));
}

static ucs_mpool_ops_t ucc_mc_cuda_stream_req_mpool_ops = {
    .chunk_alloc   = ucc_mc_cuda_stream_req_mpool_chunk_malloc,
    .chunk_release = ucc_mc_cuda_stream_req_mpool_chunk_free,
    .obj_init      = ucc_mc_cuda_stream_req_init,
    .obj_cleanup   = NULL
};

static void ucc_mc_cuda_event_init(ucs_mpool_t *mp, void *obj, void *chunk)
{
    ucc_mc_cuda_event_t *base = (ucc_mc_cuda_event_t *) obj;

    if (cudaSuccess != cudaEventCreateWithFlags(&base->event,
                                                cudaEventDisableTiming)) {
        mc_error(&ucc_mc_cuda.super, "cudaEventCreateWithFlags Failed");
    }
}

static void ucc_mc_cuda_event_cleanup(ucs_mpool_t *mp, void *obj)
{
    ucc_mc_cuda_event_t *base = (ucc_mc_cuda_event_t *) obj;
    if (cudaSuccess != cudaEventDestroy(base->event)) {
        mc_error(&ucc_mc_cuda.super, "cudaEventDestroy Failed");
    }
}

static ucs_mpool_ops_t ucc_mc_cuda_event_mpool_ops = {
    .chunk_alloc   = ucs_mpool_hugetlb_malloc,
    .chunk_release = ucs_mpool_hugetlb_free,
    .obj_init      = ucc_mc_cuda_event_init,
    .obj_cleanup   = ucc_mc_cuda_event_cleanup,
};

//TODO implement cuda kernel
static ucc_status_t ucc_mc_cuda_post_kernel_stream_task(uint32_t *status,
                                                        cudaStream_t stream)
{
    return UCC_ERR_NOT_IMPLEMENTED;
}

static ucc_status_t ucc_mc_cuda_post_driver_stream_task(uint32_t *status,
                                                        cudaStream_t stream)
{
    CUdeviceptr status_ptr  = (CUdeviceptr)status;

    CUDADRV_FUNC(cuStreamWriteValue32(stream, status_ptr,
                                      UCC_MC_CUDA_TASK_STARTED, 0));
    CUDADRV_FUNC(cuStreamWaitValue32(stream, status_ptr, UCC_OK,
                                     CU_STREAM_WAIT_VALUE_EQ));
    return UCC_OK;
}

static ucc_status_t ucc_mc_cuda_init()
{
    struct cudaDeviceProp prop;
    ucs_status_t status;
    int device;
    CUdevice cu_dev;
    int mem_ops_attr;

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

    CUDACHECK(cudaStreamCreateWithFlags(&ucc_mc_cuda.stream, cudaStreamNonBlocking));

    /*create event pool */
    status = ucs_mpool_init(&ucc_mc_cuda.events, 0, sizeof(ucc_mc_cuda_event_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_mc_cuda_event_mpool_ops, "CUDA Event Objects");
    if (status != UCS_OK) {
        mc_error(&ucc_mc_cuda.super, "Error to create event pool");
        return ucs_status_to_ucc_status(status);
    }

    /* create request pool */
    status = ucs_mpool_init(&ucc_mc_cuda.strm_reqs, 0, sizeof(ucc_mc_cuda_stream_request_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_mc_cuda_stream_req_mpool_ops, "CUDA Event Objects");
    if (status != UCS_OK) {
        mc_error(&ucc_mc_cuda.super, "Error to create event pool");
        return ucs_status_to_ucc_status(status);
    }

    if (cfg->strm_task_mode == UCC_MC_CUDA_TASK_KERNEL) {
        ucc_mc_cuda.strm_task_mode = UCC_MC_CUDA_TASK_KERNEL;
        ucc_mc_cuda.post_strm_task = ucc_mc_cuda_post_kernel_stream_task;
    } else {
        ucc_mc_cuda.strm_task_mode = UCC_MC_CUDA_TASK_MEM_OPS;
        ucc_mc_cuda.post_strm_task = ucc_mc_cuda_post_driver_stream_task;

        CUDADRV_FUNC(cuCtxGetDevice(&cu_dev));
        CUDADRV_FUNC(cuDeviceGetAttribute(&mem_ops_attr,
                    CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                    cu_dev));

        if (cfg->strm_task_mode == UCC_MC_CUDA_TASK_AUTO) {
            if (mem_ops_attr == 0) {
                mc_warn(&ucc_mc_cuda.super, "CUDA MEM OPS are not supported or disabled");
                ucc_mc_cuda.strm_task_mode = UCC_MC_CUDA_TASK_KERNEL;
                ucc_mc_cuda.post_strm_task = ucc_mc_cuda_post_kernel_stream_task;
            }
        } else if (mem_ops_attr == 0) {
            mc_error(&ucc_mc_cuda.super, "CUDA MEM OPS are not supported or disabled");
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    ucc_mc_cuda.task_strm_type = cfg->task_strm_type;

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

    mc_debug(&ucc_mc_cuda.super, "ucc_mc_cuda_mem_alloc size:%ld", size);
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

static ucc_status_t ucc_mc_cuda_memcpy(void *dst, const void *src, size_t len,
                                       ucc_memory_type_t dst_mem,
                                       ucc_memory_type_t src_mem)
{
    cudaError_t    st;
    ucc_assert(dst_mem == UCC_MEMORY_TYPE_CUDA ||
               src_mem == UCC_MEMORY_TYPE_CUDA);

    st = cudaMemcpyAsync(dst, src, len, cudaMemcpyDefault, ucc_mc_cuda.stream);
    if (st != cudaSuccess) {
        cudaGetLastError();
        mc_error(&ucc_mc_cuda.super,
                 "failed to launch cudaMemcpyAsync,  dst %p, src %p, len %zd "
                 "cuda error %d(%s)",
                 dst, src, len, st, cudaGetErrorString(st));
        return UCC_ERR_NO_MESSAGE;
    }
    st = cudaStreamSynchronize(ucc_mc_cuda.stream);
    if (st != cudaSuccess) {
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


ucc_status_t ucc_ee_cuda_task_post(void *ee_stream, void **ee_req)
{
    ucc_mc_cuda_stream_request_t *req;
    ucc_mc_cuda_event_t *cuda_event;
    ucc_status_t status;

    req = ucs_mpool_get(&ucc_mc_cuda.strm_reqs);
    ucc_assert(req);
    req->status = UCC_MC_CUDA_TASK_POSTED;
    req->stream = (cudaStream_t)ee_stream;

    if (ucc_mc_cuda.task_strm_type == UCC_MC_CUDA_USER_STREAM) {
        status = ucc_mc_cuda.post_strm_task(req->dev_status, req->stream);
        if (status != UCC_OK) {
            goto free_req;
        }
    } else {
        cuda_event = ucs_mpool_get(&ucc_mc_cuda.events);
        ucc_assert(cuda_event);
        CUDACHECK(cudaEventRecord(cuda_event->event, req->stream));
        CUDACHECK(cudaStreamWaitEvent(ucc_mc_cuda.stream, cuda_event->event, 0));
        status = ucc_mc_cuda.post_strm_task(req->dev_status, ucc_mc_cuda.stream);
        if (status != UCC_OK) {
            goto free_event;
        }
        CUDACHECK(cudaEventRecord(cuda_event->event, ucc_mc_cuda.stream));
        CUDACHECK(cudaStreamWaitEvent(req->stream, cuda_event->event, 0));
        ucs_mpool_put(cuda_event);
    }

    *ee_req = (void *) req;

    mc_info(&ucc_mc_cuda.super, "CUDA stream task posted on \"%s\" stream. req:%p",
            task_stream_types[ucc_mc_cuda.task_strm_type], req);

    return UCC_OK;

free_event:
    ucs_mpool_put(cuda_event);
free_req:
    ucs_mpool_put(req);
    return status;
}

ucc_status_t ucc_ee_cuda_task_query(void *ee_req)
{
    ucc_mc_cuda_stream_request_t *req = ee_req;

    if (req->status != UCC_MC_CUDA_TASK_STARTED) {
        return UCC_INPROGRESS;
    }
    mc_info(&ucc_mc_cuda.super, "CUDA stream task started. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ee_cuda_task_end(void *ee_req)
{
    ucc_mc_cuda_stream_request_t *req = ee_req;

    req->status = UCC_OK;

    mc_info(&ucc_mc_cuda.super, "CUDA stream task done. req:%p", req);
    ucs_mpool_put(req);
    return UCC_OK;
}

ucc_status_t ucc_ee_cuda_create_event(void **event)
{
    ucc_mc_cuda_event_t *cuda_event;

    cuda_event = ucs_mpool_get(&ucc_mc_cuda.events);
    ucc_assert(cuda_event);
    *event = cuda_event;
    return UCC_OK;
}

ucc_status_t ucc_ee_cuda_destroy_event(void *event)
{
    ucc_mc_cuda_event_t *cuda_event = event;

    ucs_mpool_put(cuda_event);
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

ucc_mc_cuda_t ucc_mc_cuda = {
    .super.super.name       = "cuda mc",
    .super.ref_cnt          = 0,
    .super.type             = UCC_MEMORY_TYPE_CUDA,
    .super.init             = ucc_mc_cuda_init,
    .super.finalize         = ucc_mc_cuda_finalize,
    .super.ops.mem_query    = ucc_mc_cuda_mem_query,
    .super.ops.mem_alloc    = ucc_mc_cuda_mem_alloc,
    .super.ops.mem_free     = ucc_mc_cuda_mem_free,
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
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_mc_cuda.super.config_table,
                                &ucc_config_global_list);
