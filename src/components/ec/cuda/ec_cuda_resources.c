#include "ec_cuda_resources.h"
#include "components/ec/ucc_ec_log.h"
#include "utils/ucc_malloc.h"

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
    int                     max_tasks = ucc_ec_cuda_config->exec_max_tasks;

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
    ucc_spinlock_init(&eee->tasks_lock, 0);
}

static void ucc_ec_cuda_executor_chunk_cleanup(ucc_mpool_t *mp, void *obj) //NOLINT: mp is unused
{
    ucc_ec_cuda_executor_t *eee = (ucc_ec_cuda_executor_t*) obj;

    CUDA_FUNC(cudaFree((void*)eee->dev_cidx));
    CUDA_FUNC(cudaFreeHost((void*)eee->tasks));
    ucc_spinlock_destroy(&eee->tasks_lock);
}

static ucc_mpool_ops_t ucc_ec_cuda_ee_executor_mpool_ops = {
    .chunk_alloc   = ucc_ec_cuda_ee_executor_mpool_chunk_malloc,
    .chunk_release = ucc_ec_cuda_ee_executor_mpool_chunk_free,
    .obj_init      = ucc_ec_cuda_executor_chunk_init,
    .obj_cleanup   = ucc_ec_cuda_executor_chunk_cleanup,
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

ucc_status_t ucc_ec_cuda_executor_kernel_calc_max_threads(int *max);

static void ucc_ec_cuda_set_threads_nbr(
    ucc_ec_base_t *ec, int *nt, int maxThreadsPerBlock, int is_reduce)
{
    ucc_status_t status;

    if (*nt != UCC_ULUNITS_AUTO) {
        if (maxThreadsPerBlock < *nt) {
            ec_warn(
                ec,
                "number of threads per block is too large, max supported is %d",
                maxThreadsPerBlock);
        } else if ((*nt % WARP_SIZE) != 0) {
            ec_warn(
                ec,
                "number of threads per block must be divisible by "
                "WARP_SIZE(=%d)",
                WARP_SIZE);
        }
    } else {
        *nt = (maxThreadsPerBlock / WARP_SIZE) * WARP_SIZE;

        if (!is_reduce) {
            // Pass max threads per block, lowering it if necessary
            // based on kernel occupancy requirements
            status = ucc_ec_cuda_executor_kernel_calc_max_threads(nt);
            if (status != UCC_OK) {
                ec_error(
                    ec,
                    "Error while calculating max threads: %s",
                    ucc_status_string(status));
            }
        }
    }
}

ucc_status_t ucc_ec_cuda_resources_init(ucc_ec_base_t *ec,
                                        ucc_ec_cuda_resources_t *resources)
{
    ucc_status_t status;
    int num_streams;
    int max_threads_per_block;
    CUdevice device;

    CUDADRV_CHECK(cuCtxGetCurrent(&resources->cu_ctx));
    resources->num_threads_reduce = ucc_ec_cuda_config->reduce_num_threads;
    resources->num_blocks_reduce  = ucc_ec_cuda_config->reduce_num_blocks;
    resources->num_threads_exec   = ucc_ec_cuda_config->exec_num_threads;
    resources->num_blocks_exec    = ucc_ec_cuda_config->exec_num_workers;

    CUDADRV_CHECK(cuCtxGetDevice(&device));
    CUDADRV_CHECK(cuDeviceGetAttribute(
        &max_threads_per_block,
        CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK,
        device));

    ucc_ec_cuda_set_threads_nbr(
        ec, &resources->num_threads_reduce, max_threads_per_block, 1);

    ucc_ec_cuda_set_threads_nbr(
        ec, &resources->num_threads_exec, max_threads_per_block, 0);

    status = ucc_mpool_init(
        &resources->events,
        0,
        sizeof(ucc_ec_cuda_event_t),
        0,
        UCC_CACHE_LINE_SIZE,
        16,
        UINT_MAX,
        &ucc_ec_cuda_event_mpool_ops,
        UCC_THREAD_MULTIPLE,
        "CUDA Event Objects");
    if (status != UCC_OK) {
        ec_error(ec, "failed to create CUDA events pool");
        goto exit_err;
    }

    status = ucc_mpool_init(&resources->executors, 0,
                            sizeof(ucc_ec_cuda_executor_t), 0,
                            UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_ec_cuda_ee_executor_mpool_ops,
                            UCC_THREAD_MULTIPLE, "CUDA EE executor objects");
    if (status != UCC_OK) {
        ec_error(ec, "failed to create executors pool");
        goto free_events_mpool;
    }

    status = ucc_mpool_init(&resources->executor_interruptible_tasks, 0,
                            sizeof(ucc_ec_cuda_executor_interruptible_task_t),
                            0, UCC_CACHE_LINE_SIZE, 16, UINT_MAX,
                            &ucc_ec_cuda_interruptible_task_mpool_ops,
                            UCC_THREAD_MULTIPLE, "interruptible executor tasks");
    if (status != UCC_OK) {
        ec_error(ec, "failed to create interruptible tasks pool");
        goto free_executors_mpool;
    }

    status = ucc_mpool_init(&resources->executor_persistent_tasks, 0,
                            sizeof(ucc_ec_cuda_executor_persistent_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 16, UINT_MAX, NULL,
                            UCC_THREAD_MULTIPLE, "persistent executor tasks");
    if (status != UCC_OK) {
        ec_error(ec, "failed to create persistent tasks pool");
        goto free_interruptible_tasks_mpool;
    }

    num_streams = ucc_ec_cuda_config->exec_num_streams;
    resources->exec_streams = ucc_calloc(num_streams, sizeof(cudaStream_t),
                                         "ec cuda streams");
    if (!resources->exec_streams) {
        ec_error(ec, "failed to allocate %zd bytes for executor streams",
                 sizeof(cudaStream_t) * num_streams);
        status = UCC_ERR_NO_MEMORY;
        goto free_persistent_tasks_mpool;
    }

    ec_debug(
        ec,
        "initialized cuda resources: cuCtx=%p, num_threads_reduce=%d, "
        "num_blocks_reduce=%d, num_threads_exec=%d, num_blocks_exec=%d",
        resources->cu_ctx,
        resources->num_threads_reduce,
        resources->num_blocks_reduce,
        resources->num_threads_exec,
        resources->num_blocks_exec);

    return UCC_OK;

free_persistent_tasks_mpool:
    ucc_mpool_cleanup(&resources->executor_persistent_tasks, 0);
free_interruptible_tasks_mpool:
    ucc_mpool_cleanup(&resources->executor_persistent_tasks, 0);
free_executors_mpool:
    ucc_mpool_cleanup(&resources->executors, 0);
free_events_mpool:
    ucc_mpool_cleanup(&resources->events, 0);
exit_err:
    return status;
}

void ucc_ec_cuda_resources_cleanup(ucc_ec_cuda_resources_t *resources)
{
    int i;
    CUcontext tmp_context;
#if CUDA_VERSION >= 12000
    CUresult status;
    unsigned long long int cu_ctx_id;

    status = cuCtxGetId(resources->cu_ctx, &cu_ctx_id);
    if (ucc_unlikely(status != CUDA_SUCCESS)) {
        // ctx is not available, can be due to cudaDeviceReset
        return;
    }
#endif
    cuCtxPushCurrent(resources->cu_ctx);
    for (i = 0; i < ucc_ec_cuda_config->exec_num_streams; i++) {
        if (resources->exec_streams[i] != NULL) {
            CUDA_FUNC(cudaStreamDestroy(resources->exec_streams[i]));
        }
    }
    ucc_mpool_cleanup(&resources->events, 1);
    ucc_mpool_cleanup(&resources->executors, 1);
    ucc_mpool_cleanup(&resources->executor_interruptible_tasks, 1);
    ucc_mpool_cleanup(&resources->executor_persistent_tasks, 1);

    ucc_free(resources->exec_streams);
    cuCtxPopCurrent(&tmp_context);
}
