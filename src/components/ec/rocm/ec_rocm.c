/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ec_rocm.h"
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

    {NULL}
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

static ucc_status_t ucc_ec_rocm_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 hipStream_t stream)
{
    return UCC_ERR_NOT_IMPLEMENTED;
}

static ucc_status_t ucc_ec_rocm_init(const ucc_ec_params_t *ec_params)
{
    ucc_ec_rocm_config_t *cfg = EC_ROCM_CONFIG;
    ucc_status_t status;
    int device, num_devices;
    hipError_t rocm_st;

    ucc_ec_rocm.stream             = NULL;
    ucc_ec_rocm.stream_initialized = 0;
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

    /*create event pool */
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

    if (cfg->strm_task_mode == UCC_EC_ROCM_TASK_KERNEL) {
        ucc_ec_rocm.strm_task_mode = UCC_EC_ROCM_TASK_KERNEL;
        ucc_ec_rocm.post_strm_task = ucc_ec_rocm_post_kernel_stream_task;
    } else {
        if (cfg->strm_task_mode == UCC_EC_ROCM_TASK_AUTO) {
            ucc_ec_rocm.strm_task_mode = UCC_EC_ROCM_TASK_KERNEL;
            ucc_ec_rocm.post_strm_task = ucc_ec_rocm_post_kernel_stream_task;
        } else {
            ec_error(&ucc_ec_rocm.super, "ROCM MEM OPS are not supported");
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
    ucc_ec_rocm_config_t *cfg = EC_ROCM_CONFIG;
    ucc_ec_rocm_stream_request_t *req;
    ucc_ec_rocm_event_t *rocm_event;
    ucc_status_t status;

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

ucc_status_t ucc_ec_rocm_create_event(void **event)
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

ucc_status_t ucc_ec_rocm_destroy_event(void *event)
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
    if (ucc_ec_rocm.stream != NULL) {
        ROCMCHECK(hipStreamDestroy(ucc_ec_rocm.stream));
        ucc_ec_rocm.stream = NULL;
    }
    ucc_spinlock_destroy(&ucc_ec_rocm.init_spinlock);
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
    .super.ops.task_post     = ucc_ec_rocm_task_post,
    .super.ops.task_query    = ucc_ec_rocm_task_query,
    .super.ops.task_end      = ucc_ec_rocm_task_end,
    .super.ops.create_event  = ucc_ec_rocm_create_event,
    .super.ops.destroy_event = ucc_ec_rocm_destroy_event,
    .super.ops.event_post    = ucc_ec_rocm_event_post,
    .super.ops.event_test    = ucc_ec_rocm_event_test,
};

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_ec_rocm.super.config_table,
                                &ucc_config_global_list);
