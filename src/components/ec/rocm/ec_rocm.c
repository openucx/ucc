/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ec_rocm.h"
#include "ec_rocm_executor.h"
#include "utils/ucc_malloc.h"
#include "utils/arch/cpu.h"
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

static const char *stream_task_modes[] = {
    [UCC_EC_ROCM_TASK_KERNEL]  = "kernel",
    [UCC_EC_ROCM_TASK_MEM_OPS] = "driver",
    [UCC_EC_ROCM_TASK_AUTO]    = "auto",
    [UCC_EC_ROCM_TASK_LAST]    = NULL
};

static const char *task_stream_types[] = {
    [UCC_EC_ROCM_USER_STREAM]      = "user",
    [UCC_EC_ROCM_INTERNAL_STREAM]  = "ucc",
    [UCC_EC_ROCM_TASK_STREAM_LAST] = NULL
};

static ucc_config_field_t ucc_ec_rocm_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_ec_rocm_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_ec_config_table)},

    {"STREAM_TASK_MODE", "auto",
     "Mechanism to create stream dependency\n"
     "kernel - use waiting kernel\n"
     "driver - use driver MEM_OPS\n"
     "auto   - runtime automatically chooses best one",
     ucc_offsetof(ucc_ec_rocm_config_t, strm_task_mode),
     UCC_CONFIG_TYPE_ENUM(stream_task_modes)},

    {"TASK_STREAM", "user",
     "Stream for rocm task\n"
     "user - user stream provided in execution engine context\n"
     "ucc  - ucc library internal stream",
     ucc_offsetof(ucc_ec_rocm_config_t, task_strm_type),
     UCC_CONFIG_TYPE_ENUM(task_stream_types)},

    {"STREAM_BLOCKING_WAIT", "1",
     "Stream is blocked until collective operation is done",
     ucc_offsetof(ucc_ec_rocm_config_t, stream_blocking_wait),
     UCC_CONFIG_TYPE_UINT},

    {"EXEC_NUM_WORKERS", "1",
     "Number of thread blocks to use for rocm executor",
     ucc_offsetof(ucc_ec_rocm_config_t, exec_num_workers),
     UCC_CONFIG_TYPE_ULUNITS},

    {"EXEC_NUM_THREADS", "512",
     "Number of thread per block to use for rocm executor",
     ucc_offsetof(ucc_ec_rocm_config_t, exec_num_threads),
     UCC_CONFIG_TYPE_ULUNITS},

    {"EXEC_MAX_TASKS", "128",
     "Maximum number of outstanding tasks per executor",
     ucc_offsetof(ucc_ec_rocm_config_t, exec_max_tasks),
     UCC_CONFIG_TYPE_ULUNITS},

    {"EXEC_NUM_STREAMS", "16",
     "Number of streams used by interruptible executor",
     ucc_offsetof(ucc_ec_rocm_config_t, exec_num_streams),
     UCC_CONFIG_TYPE_ULUNITS},

     {"REDUCE_NUM_BLOCKS", "auto",
     "Number of thread blocks to use for reduction in interruptible mode",
     ucc_offsetof(ucc_ec_rocm_config_t, reduce_num_blocks),
     UCC_CONFIG_TYPE_ULUNITS},

    {NULL}

};

static ucc_status_t ucc_ec_rocm_ee_executor_mpool_chunk_malloc(ucc_mpool_t *mp,
                                                               size_t *size_p,
                                                               void ** chunk_p)
{
    return ROCM_FUNC(hipHostMalloc((void**)chunk_p, *size_p,
                                   hipHostMallocMapped));
}

static void ucc_ec_rocm_ee_executor_mpool_chunk_free(ucc_mpool_t *mp,
                                                     void *chunk)
{
    ROCM_FUNC(hipHostFree(chunk));
}

static void ucc_ec_rocm_executor_chunk_init(ucc_mpool_t *mp, void *obj,
                                            void *chunk)
{
    ucc_ec_rocm_executor_t *eee       = (ucc_ec_rocm_executor_t*) obj;
    int                     max_tasks = EC_ROCM_CONFIG->exec_max_tasks;

    ROCM_FUNC(hipHostGetDevicePointer(
                  (void**)(&eee->dev_state), (void *)&eee->state, 0));
    ROCM_FUNC(hipHostGetDevicePointer(
                  (void**)(&eee->dev_pidx), (void *)&eee->pidx, 0));
    ROCM_FUNC(hipMalloc((void**)&eee->dev_cidx, sizeof(*eee->dev_cidx)));
    ROCM_FUNC(hipHostMalloc((void**)&eee->tasks,
                             max_tasks * sizeof(ucc_ee_executor_task_t),
                             hipHostMallocMapped));
    ROCM_FUNC(hipHostGetDevicePointer(
                  (void**)(&eee->dev_tasks), (void *)eee->tasks, 0));
    if (ucc_ec_rocm.thread_mode == UCC_THREAD_MULTIPLE) {
        ucc_spinlock_init(&eee->tasks_lock, 0);
    }
}

static void ucc_ec_rocm_executor_chunk_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_ec_rocm_executor_t *eee = (ucc_ec_rocm_executor_t*) obj;

    ROCM_FUNC(hipFree((void*)eee->dev_cidx));
    ROCM_FUNC(hipHostFree((void*)eee->tasks));
    if (ucc_ec_rocm.thread_mode == UCC_THREAD_MULTIPLE) {
        ucc_spinlock_destroy(&eee->tasks_lock);
    }
}


static ucc_mpool_ops_t ucc_ec_rocm_ee_executor_mpool_ops = {
    .chunk_alloc   = ucc_ec_rocm_ee_executor_mpool_chunk_malloc,
    .chunk_release = ucc_ec_rocm_ee_executor_mpool_chunk_free,
    .obj_init      = ucc_ec_rocm_executor_chunk_init,
    .obj_cleanup   = ucc_ec_rocm_executor_chunk_cleanup,
};


static ucc_status_t ucc_ec_rocm_stream_req_mpool_chunk_malloc(ucc_mpool_t *mp,
                                                              size_t *size_p,
                                                              void ** chunk_p)
{
    ucc_status_t status;

    status = ROCM_FUNC(hipHostMalloc((void**)chunk_p, *size_p,
                       hipHostMallocMapped));
    return status;
}

static void ucc_ec_rocm_stream_req_mpool_chunk_free(ucc_mpool_t *mp,
                                                    void *       chunk)
{
    hipHostFree(chunk);
}

static void ucc_ec_rocm_stream_req_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    ucc_ec_rocm_stream_request_t *req = (ucc_ec_rocm_stream_request_t*) obj;

    ROCM_FUNC(hipHostGetDevicePointer(
                  (void**)(&req->dev_status), (void *)&req->status, 0));
}

static ucc_mpool_ops_t ucc_ec_rocm_stream_req_mpool_ops = {
    .chunk_alloc   = ucc_ec_rocm_stream_req_mpool_chunk_malloc,
    .chunk_release = ucc_ec_rocm_stream_req_mpool_chunk_free,
    .obj_init      = ucc_ec_rocm_stream_req_init,
    .obj_cleanup   = NULL
};

static void ucc_ec_rocm_event_init(ucc_mpool_t *mp, void *obj, void *chunk)
{
    ucc_ec_rocm_event_t *base = (ucc_ec_rocm_event_t *) obj;

    if (ucc_unlikely(
          hipSuccess !=
          hipEventCreateWithFlags(&base->event, hipEventDisableTiming))) {
      ec_error(&ucc_ec_rocm.super, "hipEventCreateWithFlags Failed");
    }
}

static void ucc_ec_rocm_event_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_ec_rocm_event_t *base = (ucc_ec_rocm_event_t *) obj;

    if (ucc_unlikely(hipSuccess != hipEventDestroy(base->event))) {
        ec_error(&ucc_ec_rocm.super, "hipEventDestroy Failed");
    }
}

static ucc_mpool_ops_t ucc_ec_rocm_event_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_ec_rocm_event_init,
    .obj_cleanup   = ucc_ec_rocm_event_cleanup,
};

ucc_status_t ucc_ec_rocm_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 hipStream_t stream);

static ucc_status_t ucc_ec_rocm_post_driver_stream_task(uint32_t *status,
                                                        int blocking_wait,
                                                        hipStream_t stream)
{
    hipDeviceptr_t status_ptr  = (hipDeviceptr_t)status;

    if (blocking_wait) {
        ROCM_FUNC(hipStreamWriteValue32(stream, status_ptr,
                                        UCC_EC_ROCM_TASK_STARTED, 0));
        ROCM_FUNC(hipStreamWaitValue32(stream, status_ptr,
                                       UCC_EC_ROCM_TASK_COMPLETED,
                                       hipStreamWaitValueEq, 0xFFFFFFFF));
    }
    ROCM_FUNC(hipStreamWriteValue32(stream, status_ptr,
                                    UCC_EC_ROCM_TASK_COMPLETED_ACK, 0));
    return UCC_OK;
}

static ucc_status_t ucc_ec_rocm_init(const ucc_ec_params_t *ec_params)
{
    ucc_ec_rocm_config_t *cfg = EC_ROCM_CONFIG;
    ucc_status_t          status;
    int                   device, num_devices;
    int                   attr=0;
    hipError_t            rocm_st;
    hipDevice_t           hip_dev;
    const char           *hip_err_st_str;
    hipDeviceProp_t       prop;

    ucc_ec_rocm.stream                   = NULL;
    ucc_ec_rocm.stream_initialized       = 0;
    ucc_ec_rocm.exec_streams_initialized = 0;
    ucc_strncpy_safe(ucc_ec_rocm.super.config->log_component.name,
                     ucc_ec_rocm.super.super.name,
                     sizeof(ucc_ec_rocm.super.config->log_component.name));
    ucc_ec_rocm.thread_mode = ec_params->thread_mode;
    rocm_st = hipGetDeviceCount(&num_devices);
    if ((rocm_st != hipSuccess) || (num_devices == 0)) {
        ec_info(&ucc_ec_rocm.super, "rocm devices are not found");
        return UCC_ERR_NO_RESOURCE;
    }
    ROCMCHECK(hipGetDevice(&device));

    ROCMCHECK(hipGetDeviceProperties(&prop, device));
    cfg->reduce_num_threads = prop.maxThreadsPerBlock;

    if (cfg->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        if (prop.maxGridSize[0] < cfg->reduce_num_blocks) {
            ec_warn(&ucc_ec_rocm.super,
                    "number of blocks is too large, max supported %d",
                    prop.maxGridSize[0]);
            cfg->reduce_num_blocks = prop.maxGridSize[0];
        }
    } else {
        cfg->reduce_num_blocks = prop.maxGridSize[0];
    }

    /*create event pool */
    ucc_ec_rocm.exec_streams = ucc_calloc(cfg->exec_num_streams,
                                          sizeof(hipStream_t),
                                          "ec rocm streams");
    if (!ucc_ec_rocm.exec_streams) {
        ec_error(&ucc_ec_rocm.super, "failed to allocate streams array");
        return UCC_ERR_NO_MEMORY;
    }
    status = ucc_mpool_init(&ucc_ec_rocm.events, 0, sizeof(ucc_ec_rocm_event_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_ec_rocm_event_mpool_ops, UCC_THREAD_MULTIPLE,
                            "ROCM Event Objects");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_rocm.super, "Error to create event pool");
        return status;
    }

    /* create request pool */
    status = ucc_mpool_init(
        &ucc_ec_rocm.strm_reqs, 0, sizeof(ucc_ec_rocm_stream_request_t), 0,
        UCC_CACHE_LINE_SIZE, 16, UINT_MAX, &ucc_ec_rocm_stream_req_mpool_ops,
        UCC_THREAD_MULTIPLE, "ROCM Event Objects");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_rocm.super, "Error to create event pool");
        return status;
    }

    status = ucc_mpool_init(
        &ucc_ec_rocm.executors, 0, sizeof(ucc_ec_rocm_executor_t), 0,
        UCC_CACHE_LINE_SIZE, 16, UINT_MAX, &ucc_ec_rocm_ee_executor_mpool_ops,
        UCC_THREAD_MULTIPLE, "EE executor Objects");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_rocm.super, "failed to create executors pool");
        return status;
    }

    status = ucc_mpool_init(
        &ucc_ec_rocm.executor_interruptible_tasks, 0,
        sizeof(ucc_ec_rocm_executor_interruptible_task_t), 0, UCC_CACHE_LINE_SIZE,
        16, UINT_MAX, NULL, UCC_THREAD_MULTIPLE,
        "interruptible executor tasks");
    
    if (cfg->strm_task_mode == UCC_EC_ROCM_TASK_KERNEL) {
        ucc_ec_rocm.strm_task_mode = UCC_EC_ROCM_TASK_KERNEL;
        ucc_ec_rocm.post_strm_task = ucc_ec_rocm_post_kernel_stream_task;
    } else {
        ucc_ec_rocm.strm_task_mode = UCC_EC_ROCM_TASK_MEM_OPS;
        ucc_ec_rocm.post_strm_task = ucc_ec_rocm_post_driver_stream_task;

        rocm_st = hipCtxGetDevice(&hip_dev);
        if (rocm_st != hipSuccess){
            hip_err_st_str = hipGetErrorString(rocm_st);
            ec_debug(&ucc_ec_rocm.super, "hipCtxGetDevice() failed: %s",
                     hip_err_st_str);
            attr = 0;
        } else {
            ROCM_FUNC(hipDeviceGetAttribute(&attr,
		      hipDeviceAttributeCanUseStreamWaitValue,
                      hip_dev));
        }
        if (cfg->strm_task_mode == UCC_EC_ROCM_TASK_AUTO) {
            if (attr == 0) {
                ec_info(&ucc_ec_rocm.super,
                        "ROCm MEM OPS are not supported or disabled");
                ucc_ec_rocm.strm_task_mode = UCC_EC_ROCM_TASK_KERNEL;
                ucc_ec_rocm.post_strm_task = ucc_ec_rocm_post_kernel_stream_task;
            }
        } else if (attr == 0) {
            ec_error(&ucc_ec_rocm.super,
                     "ROCm MEM OPS are not supported or disabled");
            return UCC_ERR_NOT_SUPPORTED;
        }
    }

    ucc_ec_rocm.task_strm_type = cfg->task_strm_type;
    ucc_spinlock_init(&ucc_ec_rocm.init_spinlock, 0);
    return UCC_OK;
}

static ucc_status_t ucc_ec_rocm_get_attr(ucc_ec_attr_t *ec_attr)
{
    if (ec_attr->field_mask & UCC_EC_ATTR_FIELD_THREAD_MODE) {
        ec_attr->thread_mode = ucc_ec_rocm.thread_mode;
    }
    return UCC_OK;
}

ucc_status_t ucc_ec_rocm_task_post(void *ee_stream, void **ee_req)
{
    ucc_ec_rocm_config_t         *cfg = EC_ROCM_CONFIG;
    ucc_ec_rocm_stream_request_t *req;
    ucc_ec_rocm_event_t          *rocm_event;
    ucc_status_t                  status = UCC_OK;

    UCC_EC_ROCM_INIT_STREAM();
    req = ucc_mpool_get(&ucc_ec_rocm.strm_reqs);
    if (ucc_unlikely(!req)) {
        ec_error(&ucc_ec_rocm.super, "Failed to allocate stream request");
	return UCC_ERR_NO_MEMORY;
    }
    req->status = UCC_EC_ROCM_TASK_POSTED;
    req->stream = (hipStream_t)ee_stream;

    if (ucc_ec_rocm.task_strm_type == UCC_EC_ROCM_USER_STREAM) {
        status = ucc_ec_rocm.post_strm_task(req->dev_status,
                                            cfg->stream_blocking_wait,
                                            req->stream);
        if (status != UCC_OK) {
            goto free_req;
        }
    } else {
        rocm_event = ucc_mpool_get(&ucc_ec_rocm.events);
        if (ucc_unlikely(!rocm_event)) {
	    ec_error(&ucc_ec_rocm.super, "Failed to allocate rocm event");
	    goto free_event;
	}
        ROCMCHECK(hipEventRecord(rocm_event->event, req->stream));
        ROCMCHECK(hipStreamWaitEvent(ucc_ec_rocm.stream, rocm_event->event, 0));
        status = ucc_ec_rocm.post_strm_task(req->dev_status,
                                            cfg->stream_blocking_wait,
                                            ucc_ec_rocm.stream);
        if (ucc_unlikely(status != UCC_OK)) {
            goto free_req;
        }
        ROCMCHECK(hipEventRecord(rocm_event->event, ucc_ec_rocm.stream));
        ROCMCHECK(hipStreamWaitEvent(req->stream, rocm_event->event, 0));
        ucc_mpool_put(rocm_event);
    }

    *ee_req = (void *) req;

    ec_info(&ucc_ec_rocm.super, "ROCM stream task posted on \"%s\" stream. req:%p",
            task_stream_types[ucc_ec_rocm.task_strm_type], req);

    return UCC_OK;

free_event:
    ucc_mpool_put(rocm_event);
free_req:
    ucc_mpool_put(req);
    return status;
}

ucc_status_t ucc_ec_rocm_task_query(void *ee_req)
{
    ucc_ec_rocm_stream_request_t *req = ee_req;

    /* ee task might be only in POSTED, STARTED or COMPLETED_ACK state
       COMPLETED state is used by ucc_ee_rocm_task_end function to request
       stream unblock*/
    ucc_assert(req->status != UCC_EC_ROCM_TASK_COMPLETED);
    if (req->status == UCC_EC_ROCM_TASK_POSTED) {
        return UCC_INPROGRESS;
    }
    ec_info(&ucc_ec_rocm.super, "ROCM stream task started. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ec_rocm_task_end(void *ee_req)
{
    ucc_ec_rocm_stream_request_t *req = ee_req;
    volatile ucc_ec_task_status_t *st = &req->status;

    /* can be safely ended only if it's in STARTED or COMPLETED_ACK state */
    ucc_assert((*st != UCC_EC_ROCM_TASK_POSTED) &&
               (*st != UCC_EC_ROCM_TASK_COMPLETED));
    if (*st == UCC_EC_ROCM_TASK_STARTED) {
        *st = UCC_EC_ROCM_TASK_COMPLETED;
        while(*st != UCC_EC_ROCM_TASK_COMPLETED_ACK) { }
    }
    ucc_mpool_put(req);
    ec_info(&ucc_ec_rocm.super, "ROCM stream task done. req:%p", req);
    return UCC_OK;
}

ucc_status_t ucc_ec_rocm_event_create(void **event)
{
    ucc_ec_rocm_event_t *rocm_event;

    rocm_event = ucc_mpool_get(&ucc_ec_rocm.events);
    if (ucc_unlikely(!rocm_event)) {
	ec_error(&ucc_ec_rocm.super, "Failed to allocate rocm event");
	return UCC_ERR_NO_MEMORY;
    }
    *event = rocm_event;
    return UCC_OK;
}

ucc_status_t ucc_ec_rocm_event_destroy(void *event)
{
    ucc_ec_rocm_event_t *rocm_event = event;

    ucc_mpool_put(rocm_event);
    return UCC_OK;
}

ucc_status_t ucc_ec_rocm_event_post(void *ee_context, void *event)
{
    hipStream_t stream              = (hipStream_t) ee_context;
    ucc_ec_rocm_event_t *rocm_event = event;

    ROCMCHECK(hipEventRecord(rocm_event->event, stream));
    return UCC_OK;
}

ucc_status_t ucc_ec_rocm_event_test(void *event)
{
    ucc_ec_rocm_event_t *rocm_event = event;
    hipError_t hip_err;

    hip_err = hipEventQuery(rocm_event->event);
    if (ucc_unlikely((hip_err != hipSuccess) &&
                     (hip_err != hipErrorNotReady))) {
        ROCMCHECK(hip_err);
    }
    return hip_error_to_ucc_status(hip_err);
}

static ucc_status_t ucc_ec_rocm_finalize()
{
    int i;

    if (ucc_ec_rocm.stream != NULL) {
        ROCMCHECK(hipStreamDestroy(ucc_ec_rocm.stream));
        ucc_ec_rocm.stream = NULL;
    }
    if (ucc_ec_rocm.exec_streams_initialized) {
        for (i = 0; i < EC_ROCM_CONFIG->exec_num_streams; i++) {
            ROCM_FUNC(hipStreamDestroy(ucc_ec_rocm.exec_streams[i]));
        }
        ucc_ec_rocm.exec_streams_initialized = 0;
    }

    ucc_mpool_cleanup(&ucc_ec_rocm.events, 1);
    ucc_mpool_cleanup(&ucc_ec_rocm.strm_reqs, 1);
    ucc_mpool_cleanup(&ucc_ec_rocm.executors, 1);
    ucc_free(ucc_ec_rocm.exec_streams);
    return UCC_OK;
}

ucc_ec_rocm_t ucc_ec_rocm = {
    .super.super.name             = "rocm ec",
    .super.ref_cnt                = 0,
    .super.type                   = UCC_EE_ROCM_STREAM,
    .super.init                   = ucc_ec_rocm_init,
    .super.get_attr               = ucc_ec_rocm_get_attr,
    .super.finalize               = ucc_ec_rocm_finalize,
    .super.config_table =
        {
            .name   = "ROCM execution component",
            .prefix = "EC_ROCM_",
            .table  = ucc_ec_rocm_config_table,
            .size   = sizeof(ucc_ec_rocm_config_t),
        },
    .super.ops.task_post              = ucc_ec_rocm_task_post,
    .super.ops.task_query             = ucc_ec_rocm_task_query,
    .super.ops.task_end               = ucc_ec_rocm_task_end,
    .super.ops.create_event           = ucc_ec_rocm_event_create,
    .super.ops.destroy_event          = ucc_ec_rocm_event_destroy,
    .super.ops.event_post             = ucc_ec_rocm_event_post,
    .super.ops.event_test             = ucc_ec_rocm_event_test,
    .super.executor_ops.init          = ucc_rocm_executor_init,
    .super.executor_ops.start         = ucc_rocm_executor_start,
    .super.executor_ops.status        = ucc_rocm_executor_status,
    .super.executor_ops.stop          = ucc_rocm_executor_stop,
    .super.executor_ops.task_post     = ucc_rocm_executor_task_post,
    .super.executor_ops.task_test     = ucc_rocm_executor_task_test,
    .super.executor_ops.task_finalize = ucc_rocm_executor_task_finalize,
    .super.executor_ops.finalize      = ucc_rocm_executor_finalize,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_ec_rocm.super.config_table,
                                &ucc_config_global_list);
