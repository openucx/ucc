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

    {"EXEC_NUM_THREADS", "512",
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

static ucc_status_t ucc_ec_cuda_ee_executor_mpool_chunk_malloc(ucc_mpool_t *mp, //NOLINT: mp is unused
                                                               size_t *size_p,
                                                               void ** chunk_p)
{
    return CUDA_FUNC(cudaHostAlloc((void**)chunk_p, *size_p,
                                   cudaHostAllocMapped));
}

static void ucc_ec_cuda_ee_executor_mpool_chunk_free(ucc_mpool_t *mp, //NOLINT: mp is unused
                                                     void *chunk)
{
    CUDA_FUNC(cudaFreeHost(chunk));
}

static void ucc_ec_cuda_executor_chunk_init(ucc_mpool_t *mp, void *obj, //NOLINT: mp is unused
                                            void *chunk) //NOLINT: chunk is unused
{
    ucc_ec_cuda_executor_t *eee       = (ucc_ec_cuda_executor_t*) obj;
    int                     max_tasks = EC_CUDA_CONFIG->exec_max_tasks;

    CUDA_FUNC(cudaHostGetDevicePointer(
                  (void**)(&eee->dev_state), (void *)&eee->state, 0));
    CUDA_FUNC(cudaHostGetDevicePointer(
                  (void**)(&eee->dev_pidx), (void *)&eee->pidx, 0));
    CUDA_FUNC(cudaMalloc((void**)&eee->dev_cidx, sizeof(*eee->dev_cidx)));
    CUDA_FUNC(cudaHostAlloc((void**)&eee->tasks,
                            max_tasks * MAX_SUBTASKS *
                            sizeof(ucc_ee_executor_task_args_t),
                            cudaHostAllocMapped));
    CUDA_FUNC(cudaHostGetDevicePointer(
                  (void**)(&eee->dev_tasks), (void *)eee->tasks, 0));
    if (ucc_ec_cuda.thread_mode == UCC_THREAD_MULTIPLE) {
        ucc_spinlock_init(&eee->tasks_lock, 0);
    }
}

static void ucc_ec_cuda_executor_chunk_cleanup(ucc_mpool_t *mp, void *obj) //NOLINT: mp is unused
{
    ucc_ec_cuda_executor_t *eee = (ucc_ec_cuda_executor_t*) obj;

    CUDA_FUNC(cudaFree((void*)eee->dev_cidx));
    CUDA_FUNC(cudaFreeHost((void*)eee->tasks));
    if (ucc_ec_cuda.thread_mode == UCC_THREAD_MULTIPLE) {
        ucc_spinlock_destroy(&eee->tasks_lock);
    }
}


static ucc_mpool_ops_t ucc_ec_cuda_ee_executor_mpool_ops = {
    .chunk_alloc   = ucc_ec_cuda_ee_executor_mpool_chunk_malloc,
    .chunk_release = ucc_ec_cuda_ee_executor_mpool_chunk_free,
    .obj_init      = ucc_ec_cuda_executor_chunk_init,
    .obj_cleanup   = ucc_ec_cuda_executor_chunk_cleanup,
};

static void ucc_ec_cuda_event_init(ucc_mpool_t *mp, void *obj, void *chunk) //NOLINT: mp is unused
{
    ucc_ec_cuda_event_t *base = (ucc_ec_cuda_event_t *) obj;

    CUDA_FUNC(cudaEventCreateWithFlags(&base->event, cudaEventDisableTiming));
}

static void ucc_ec_cuda_event_cleanup(ucc_mpool_t *mp, void *obj) //NOLINT: mp is unused
{
    ucc_ec_cuda_event_t *base = (ucc_ec_cuda_event_t *) obj;

    CUDA_FUNC(cudaEventDestroy(base->event));
}

static ucc_mpool_ops_t ucc_ec_cuda_event_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_ec_cuda_event_init,
    .obj_cleanup   = ucc_ec_cuda_event_cleanup,
};

static void ucc_ec_cuda_graph_init(ucc_mpool_t *mp, void *obj, void *chunk) //NOLINT: mp is unused
{
    ucc_ec_cuda_executor_interruptible_task_t *task =
         (ucc_ec_cuda_executor_interruptible_task_t *) obj;
    cudaGraphNode_t memcpy_node;
    int i;

    CUDA_FUNC(cudaGraphCreate(&task->graph, 0));
    for (i = 0; i < UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS; i++) {
        CUDA_FUNC(
            cudaGraphAddMemcpyNode1D(&memcpy_node, task->graph, NULL, 0,
                                     (void*)1, (void*)1, 1, cudaMemcpyDefault));
    }

    CUDA_FUNC(
        cudaGraphInstantiateWithFlags(&task->graph_exec, task->graph, 0));
}

static void ucc_ec_cuda_graph_cleanup(ucc_mpool_t *mp, void *obj) //NOLINT: mp is unused
{
    ucc_ec_cuda_executor_interruptible_task_t *task =
         (ucc_ec_cuda_executor_interruptible_task_t *) obj;

    CUDA_FUNC(cudaGraphExecDestroy(task->graph_exec));
    CUDA_FUNC(cudaGraphDestroy(task->graph));
}

static ucc_mpool_ops_t ucc_ec_cuda_interruptible_task_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_ec_cuda_graph_init,
    .obj_cleanup   = ucc_ec_cuda_graph_cleanup,
};

static inline void ucc_ec_cuda_set_threads_nbr(int *nt, int maxThreadsPerBlock)
{
    if (*nt != UCC_ULUNITS_AUTO) {
        if (maxThreadsPerBlock < *nt) {
            ec_warn(
                &ucc_ec_cuda.super,
                "number of threads per block is too large, max supported is %d",
                maxThreadsPerBlock);
        } else if ((*nt % WARP_SIZE) != 0) {
            ec_warn(&ucc_ec_cuda.super,
                    "number of threads per block must be divisible by "
                    "WARP_SIZE(=%d)",
                    WARP_SIZE);
        } else {
            return;
        }
    }

    *nt = (maxThreadsPerBlock / WARP_SIZE) * WARP_SIZE;
}

static ucc_status_t ucc_ec_cuda_init(const ucc_ec_params_t *ec_params)
{
    ucc_ec_cuda_config_t *cfg = EC_CUDA_CONFIG;
    ucc_status_t          status;
    int                   device, num_devices;
    cudaError_t           cuda_st;
    struct cudaDeviceProp prop;
    int                   supportsCoopLaunch = 0;

    ucc_ec_cuda.stream                   = NULL;
    ucc_ec_cuda.stream_initialized       = 0;
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

    ucc_ec_cuda_set_threads_nbr((int *)&cfg->exec_num_threads,
                                prop.maxThreadsPerBlock);
    ucc_ec_cuda_set_threads_nbr(&cfg->reduce_num_threads,
                                prop.maxThreadsPerBlock);

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

    /*create event pool */
    ucc_ec_cuda.exec_streams = ucc_calloc(cfg->exec_num_streams,
                                          sizeof(cudaStream_t),
                                          "ec cuda streams");
    if (!ucc_ec_cuda.exec_streams) {
        ec_error(&ucc_ec_cuda.super, "failed to allocate streams array");
        return UCC_ERR_NO_MEMORY;
    }
    status = ucc_mpool_init(&ucc_ec_cuda.events, 0, sizeof(ucc_ec_cuda_event_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_ec_cuda_event_mpool_ops, UCC_THREAD_MULTIPLE,
                            "CUDA Event Objects");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_cuda.super, "failed to create event pool");
        return status;
    }

    status = ucc_mpool_init(
        &ucc_ec_cuda.executors, 0, sizeof(ucc_ec_cuda_executor_t), 0,
        UCC_CACHE_LINE_SIZE, 16, UINT_MAX, &ucc_ec_cuda_ee_executor_mpool_ops,
        UCC_THREAD_MULTIPLE, "EE executor Objects");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_cuda.super, "failed to create executors pool");
        return status;
    }

    status = ucc_mpool_init(
        &ucc_ec_cuda.executor_interruptible_tasks, 0,
        sizeof(ucc_ec_cuda_executor_interruptible_task_t), 0, UCC_CACHE_LINE_SIZE,
        16, UINT_MAX, &ucc_ec_cuda_interruptible_task_mpool_ops,
        UCC_THREAD_MULTIPLE, "interruptible executor tasks");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_cuda.super, "failed to create interruptible tasks pool");
        return status;
    }

    status = ucc_mpool_init(
        &ucc_ec_cuda.executor_persistent_tasks, 0,
        sizeof(ucc_ec_cuda_executor_persistent_task_t), 0, UCC_CACHE_LINE_SIZE,
        16, UINT_MAX, NULL, UCC_THREAD_MULTIPLE,
        "persistent executor tasks");
    if (status != UCC_OK) {
        ec_error(&ucc_ec_cuda.super, "failed to create persistent tasks pool");
        return status;
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
        cudaDeviceGetAttribute(&supportsCoopLaunch,
                               cudaDevAttrCooperativeLaunch, device);
        if (!supportsCoopLaunch) {
            cfg->use_cooperative_launch = 0;
            ec_warn(&ucc_ec_cuda.super,
                     "CUDA cooperative groups are not supported. "
                     "Fall back to non cooperative launch.");
        }
    }

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
    ucc_ec_cuda_event_t *cuda_event;

    cuda_event = ucc_mpool_get(&ucc_ec_cuda.events);
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
    cudaError_t cu_err;
    ucc_ec_cuda_event_t *cuda_event = event;

    cu_err = cudaEventQuery(cuda_event->event);

    if (ucc_unlikely((cu_err != cudaSuccess) &&
                     (cu_err != cudaErrorNotReady))) {
        CUDA_CHECK(cu_err);
    }
    return cuda_error_to_ucc_status(cu_err);
}

static ucc_status_t ucc_ec_cuda_finalize()
{
    int i;

    if (ucc_ec_cuda.stream_initialized) {
        CUDA_FUNC(cudaStreamDestroy(ucc_ec_cuda.stream));
        ucc_ec_cuda.stream_initialized = 0;
    }

    if (ucc_ec_cuda.exec_streams_initialized) {
        for (i = 0; i < EC_CUDA_CONFIG->exec_num_streams; i++) {
            CUDA_FUNC(cudaStreamDestroy(ucc_ec_cuda.exec_streams[i]));
        }
        ucc_ec_cuda.exec_streams_initialized = 0;
    }

    ucc_mpool_cleanup(&ucc_ec_cuda.events, 1);
    ucc_mpool_cleanup(&ucc_ec_cuda.executors, 1);
    ucc_mpool_cleanup(&ucc_ec_cuda.executor_interruptible_tasks, 1);
    ucc_mpool_cleanup(&ucc_ec_cuda.executor_persistent_tasks, 1);
    ucc_free(ucc_ec_cuda.exec_streams);

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

UCC_CONFIG_REGISTER_TABLE_ENTRY(&ucc_ec_cuda.super.config_table,
                                &ucc_config_global_list);
