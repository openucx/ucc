/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cuda.h"
#include "ec_cuda_executor.h"
#include "utils/ucc_malloc.h"
#include "utils/arch/cpu.h"
#include <cuda_runtime.h>
#include <cuda.h>

static const char *stream_task_modes[] = {
    [UCC_EC_CUDA_TASK_KERNEL]  = "kernel",
    [UCC_EC_CUDA_TASK_MEM_OPS] = "driver",
    [UCC_EC_CUDA_TASK_AUTO]    = "auto",
    [UCC_EC_CUDA_TASK_LAST]    = NULL
};

static ucc_config_field_t ucc_ec_cuda_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_ec_cuda_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_ec_config_table)},

    {"STREAM_TASK_MODE", "auto",
     "Mechanism to create stream dependency\n"
     "kernel - use waiting kernel\n"
     "driver - use driver MEM_OPS\n"
     "auto   - runtime automatically chooses best one",
     ucc_offsetof(ucc_ec_cuda_config_t, strm_task_mode),
     UCC_CONFIG_TYPE_ENUM(stream_task_modes)},

    {"EXEC_NUM_WORKERS", "1",
     "Number of thread blocks to use for cuda persistent executor",
     ucc_offsetof(ucc_ec_cuda_config_t, exec_num_workers),
     UCC_CONFIG_TYPE_ULUNITS},

    {"EXEC_NUM_THREADS", "auto",
     "Number of threads per block to use for cuda persistent executor",
     ucc_offsetof(ucc_ec_cuda_config_t, exec_num_threads),
     UCC_CONFIG_TYPE_ULUNITS},

    {"EXEC_MAX_TASKS", "128",
     "Maximum number of outstanding tasks per executor",
     ucc_offsetof(ucc_ec_cuda_config_t, exec_max_tasks),
     UCC_CONFIG_TYPE_ULUNITS},

    {"EXEC_NUM_STREAMS", "16",
     "Number of streams used by interruptible executor",
     ucc_offsetof(ucc_ec_cuda_config_t, exec_num_streams),
     UCC_CONFIG_TYPE_ULUNITS},

    {"EXEC_COPY_LARGE_THRESH", "1M",
     "Single memcopy size to switch from kernel copy to cudaMemcpy",
     ucc_offsetof(ucc_ec_cuda_config_t, exec_copy_thresh),
     UCC_CONFIG_TYPE_MEMUNITS},

    {"REDUCE_NUM_BLOCKS", "auto",
     "Number of thread blocks to use for reduction in interruptible mode",
     ucc_offsetof(ucc_ec_cuda_config_t, reduce_num_blocks),
     UCC_CONFIG_TYPE_ULUNITS},

    {"REDUCE_NUM_THREADS", "auto",
     "Number of threads per block to use for reduction in interruptible "
     "executor",
     ucc_offsetof(ucc_ec_cuda_config_t, reduce_num_threads),
     UCC_CONFIG_TYPE_ULUNITS},

    {"USE_COOPERATIVE_LAUNCH", "0",
     "whether to use cooperative launch in persistent kernel executor",
     ucc_offsetof(ucc_ec_cuda_config_t, use_cooperative_launch),
     UCC_CONFIG_TYPE_BOOL},

    {NULL}

};

static ucc_status_t ucc_ec_cuda_init(const ucc_ec_params_t *ec_params)
{
    ucc_ec_cuda_config_t *cfg                  = EC_CUDA_CONFIG;
    int                   supports_coop_launch = 0;
    int                   device, num_devices;
    cudaError_t           cuda_st;
    struct cudaDeviceProp prop;

    ucc_ec_cuda_config = ucc_derived_of(ucc_ec_cuda.super.config,
                                        ucc_ec_cuda_config_t);
    ucc_ec_cuda.exec_streams_initialized = 0;
    ucc_strncpy_safe(ucc_ec_cuda.super.config->log_component.name,
                     ucc_ec_cuda.super.super.name,
                     sizeof(ucc_ec_cuda.super.config->log_component.name));
    ucc_ec_cuda.thread_mode = ec_params->thread_mode;
    cuda_st = cudaGetDeviceCount(&num_devices);
    if ((cuda_st != cudaSuccess) || (num_devices == 0)) {
        ec_debug(&ucc_ec_cuda.super, "CUDA devices are not found");
        return UCC_ERR_NO_RESOURCE;
    }
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (cfg->reduce_num_blocks != UCC_ULUNITS_AUTO) {
        if (prop.maxGridSize[0] < cfg->reduce_num_blocks) {
            ec_warn(&ucc_ec_cuda.super,
                    "number of blocks is too large, max supported is %d",
                    prop.maxGridSize[0]);
            cfg->reduce_num_blocks = prop.maxGridSize[0];
        }
    } else {
        cfg->reduce_num_blocks = prop.maxGridSize[0];
    }

    if (cfg->exec_num_streams < 1) {
        ec_warn(&ucc_ec_cuda.super,
                "number of streams is too small, min supported 1");
        cfg->exec_num_streams = 1;
    }

    if (cfg->strm_task_mode == UCC_EC_CUDA_TASK_KERNEL) {
        ucc_ec_cuda.strm_task_mode = UCC_EC_CUDA_TASK_KERNEL;
    } else {
        ucc_ec_cuda.strm_task_mode = UCC_EC_CUDA_TASK_MEM_OPS;
#if CUDA_VERSION < 12000
        CUresult cu_st;
        CUdevice cu_dev;
        int attr;
        cu_st = cuCtxGetDevice(&cu_dev);
        if (cu_st != CUDA_SUCCESS){
            const char *cu_err_st_str;
            cuGetErrorString(cu_st, &cu_err_st_str);
            ec_debug(&ucc_ec_cuda.super, "cuCtxGetDevice() failed: %s",
                     cu_err_st_str);
            attr = 0;
        } else {
            CUDADRV_FUNC(cuDeviceGetAttribute(&attr,
                        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                        cu_dev));
        }

        if (cfg->strm_task_mode == UCC_EC_CUDA_TASK_AUTO) {
            if (attr == 0) {
                ec_debug(&ucc_ec_cuda.super,
                         "CUDA MEM OPS are not supported or disabled");
                ucc_ec_cuda.strm_task_mode = UCC_EC_CUDA_TASK_KERNEL;
            }
        } else if (attr == 0) {
            ec_error(&ucc_ec_cuda.super,
                     "CUDA MEM OPS are not supported or disabled");
            return UCC_ERR_NOT_SUPPORTED;
        }
#endif
    }

    if (cfg->use_cooperative_launch == 1) {
        cudaDeviceGetAttribute(&supports_coop_launch,
                               cudaDevAttrCooperativeLaunch, device);
        if (!supports_coop_launch) {
            cfg->use_cooperative_launch = 0;
            ec_warn(&ucc_ec_cuda.super,
                    "CUDA cooperative groups are not supported. "
                    "Fall back to non cooperative launch.");
        }
    }

    ucc_ec_cuda.resources_hash = kh_init(ucc_ec_cuda_resources_hash);
    ucc_spinlock_init(&ucc_ec_cuda.init_spinlock, 0);
    return UCC_OK;
}

static ucc_status_t ucc_ec_cuda_get_attr(ucc_ec_attr_t *ec_attr)
{
    if (ec_attr->field_mask & UCC_EC_ATTR_FIELD_THREAD_MODE) {
        ec_attr->thread_mode = ucc_ec_cuda.thread_mode;
    }
    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_event_create(void **event)
{
    ucc_ec_cuda_event_t     *cuda_event;
    ucc_ec_cuda_resources_t *resources;
    ucc_status_t             status;

    status = ucc_ec_cuda_get_resources(&resources);
    if (ucc_unlikely(status != UCC_OK)) {
        return status;
    }
    cuda_event = ucc_mpool_get(&resources->events);
    if (ucc_unlikely(!cuda_event)) {
        ec_error(&ucc_ec_cuda.super, "failed to get event from mpool");
        return UCC_ERR_NO_MEMORY;
    }

    *event = cuda_event;
    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_event_destroy(void *event)
{
    ucc_ec_cuda_event_t *cuda_event = event;

    ucc_mpool_put(cuda_event);
    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_event_post(void *ee_context, void *event)
{
    cudaStream_t         stream     = (cudaStream_t )ee_context;
    ucc_ec_cuda_event_t *cuda_event = event;

    CUDA_CHECK(cudaEventRecord(cuda_event->event, stream));
    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_event_test(void *event)
{
    ucc_ec_cuda_event_t *cuda_event = event;
    cudaError_t cu_err;

    cu_err = cudaEventQuery(cuda_event->event);

    if (ucc_unlikely((cu_err != cudaSuccess) &&
                     (cu_err != cudaErrorNotReady))) {
        CUDA_CHECK(cu_err);
    }
    return cuda_error_to_ucc_status(cu_err);
}

static ucc_status_t ucc_ec_cuda_finalize()
{
    ucc_ec_cuda_resources_t *resources;

    resources = ec_cuda_resources_hash_pop(ucc_ec_cuda.resources_hash);
    while (resources) {
        ucc_ec_cuda_resources_cleanup(resources);
        resources = ec_cuda_resources_hash_pop(ucc_ec_cuda.resources_hash);
    }

    ucc_spinlock_destroy(&ucc_ec_cuda.init_spinlock);

    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_get_resources(ucc_ec_cuda_resources_t **resources)
{
    CUcontext cu_ctx;
    unsigned long long int cu_ctx_id;
    ucc_status_t status;

    status = CUDADRV_FUNC(cuCtxGetCurrent(&cu_ctx));
    if (ucc_unlikely(status != UCC_OK)) {
        ec_error(&ucc_ec_cuda.super, "failed to get current CUDA context");
        return status;
    }

#if CUDA_VERSION < 12000
    cu_ctx_id = 1;
#else
    status = CUDADRV_FUNC(cuCtxGetId(cu_ctx, &cu_ctx_id));
    if (ucc_unlikely(status != UCC_OK)) {
        /* worakround for pytorch, progress thread doesn't have cuda context for GPU 0*/
        cu_ctx_id = 0x12345;
        ec_debug(&ucc_ec_cuda.super, "failed to get currect CUDA context ID");
    }
#endif

    *resources = ec_cuda_resources_hash_get(ucc_ec_cuda.resources_hash,
                                            cu_ctx_id);
    if (ucc_unlikely(*resources == NULL)) {
        ucc_spin_lock(&ucc_ec_cuda.init_spinlock);
        *resources = ec_cuda_resources_hash_get(ucc_ec_cuda.resources_hash,
                                                cu_ctx_id);
        if (*resources == NULL) {
            *resources = ucc_malloc(sizeof(ucc_ec_cuda_resources_t),
                                    "ec cuda resources");
            if (*resources == NULL) {
                ec_error(&ucc_ec_cuda.super,
                         "failed to allocate %zd bytes for resources",
                         sizeof(ucc_ec_cuda_resources_t));
                ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);
                return UCC_ERR_NO_MEMORY;
            }
            status = ucc_ec_cuda_resources_init(&ucc_ec_cuda.super,
                                                *resources);
            if (status != UCC_OK) {
                ucc_free(*resources);
                ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);
                return status;
            }
            ec_cuda_resources_hash_put(ucc_ec_cuda.resources_hash, cu_ctx_id,
                                       *resources);
        }
        ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);
    }
    return UCC_OK;
}

ucc_ec_cuda_t ucc_ec_cuda = {
    .super.super.name                 = "cuda ec",
    .super.ref_cnt                    = 0,
    .super.type                       = UCC_EE_CUDA_STREAM,
    .super.init                       = ucc_ec_cuda_init,
    .super.get_attr                   = ucc_ec_cuda_get_attr,
    .super.finalize                   = ucc_ec_cuda_finalize,
    .super.config_table =
        {
            .name   = "CUDA execution component",
            .prefix = "EC_CUDA_",
            .table  = ucc_ec_cuda_config_table,
            .size   = sizeof(ucc_ec_cuda_config_t),
        },
    .super.ops.create_event           = ucc_ec_cuda_event_create,
    .super.ops.destroy_event          = ucc_ec_cuda_event_destroy,
    .super.ops.event_post             = ucc_ec_cuda_event_post,
    .super.ops.event_test             = ucc_ec_cuda_event_test,
    .super.executor_ops.init          = ucc_cuda_executor_init,
    .super.executor_ops.start         = ucc_cuda_executor_start,
    .super.executor_ops.status        = ucc_cuda_executor_status,
    .super.executor_ops.stop          = ucc_cuda_executor_stop,
    .super.executor_ops.task_post     = ucc_cuda_executor_task_post,
    .super.executor_ops.task_test     = ucc_cuda_executor_task_test,
    .super.executor_ops.task_finalize = ucc_cuda_executor_task_finalize,
    .super.executor_ops.finalize      = ucc_cuda_executor_finalize,
};

ucc_ec_cuda_config_t *ucc_ec_cuda_config;

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_ec_cuda.super.config_table,
                                &ucc_config_global_list);
