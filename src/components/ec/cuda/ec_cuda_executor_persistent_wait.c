/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ec_cuda_executor.h"

ucc_status_t
ucc_ec_cuda_post_kernel_stream_task(ucc_ec_cuda_executor_state_t *state,
                                    cudaStream_t stream);

static ucc_status_t
ucc_ec_cuda_post_driver_stream_task(ucc_ec_cuda_executor_state_t *state,
                                    cudaStream_t stream)
{
    CUdeviceptr              state_ptr       = (CUdeviceptr)state;
    CUstreamBatchMemOpParams batch_memops[3] = {};

    batch_memops[0].operation          = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    batch_memops[0].writeValue.address = state_ptr;
    batch_memops[0].writeValue.value   = UCC_EC_CUDA_EXECUTOR_STARTED;
    batch_memops[0].writeValue.flags   = 0;

    batch_memops[1].operation         = CU_STREAM_MEM_OP_WAIT_VALUE_32;
    batch_memops[1].waitValue.address = state_ptr;
    batch_memops[1].waitValue.value   = UCC_EC_CUDA_EXECUTOR_SHUTDOWN;
    batch_memops[1].waitValue.flags   = CU_STREAM_WAIT_VALUE_EQ;

    batch_memops[2].operation          = CU_STREAM_MEM_OP_WRITE_VALUE_32;
    batch_memops[2].writeValue.address = state_ptr;
    batch_memops[2].writeValue.value   = UCC_EC_CUDA_EXECUTOR_SHUTDOWN_ACK;
    batch_memops[2].writeValue.flags   = 0;

    CUDADRV_FUNC(cuStreamBatchMemOp(stream, 3, batch_memops, 0));
    return UCC_OK;
}

ucc_status_t ucc_cuda_executor_persistent_wait_start(ucc_ee_executor_t *executor,
                                                     void *ee_context)
{
    ucc_ec_cuda_executor_t *eee    = ucc_derived_of(executor,
                                                    ucc_ec_cuda_executor_t);
    cudaStream_t            stream = (cudaStream_t)ee_context;

    eee->super.ee_context = ee_context;
    eee->state            = UCC_EC_CUDA_EXECUTOR_POSTED;
    eee->mode             = UCC_EC_CUDA_EXECUTOR_MODE_PERSISTENT;

    ucc_memory_cpu_store_fence();
    if (ucc_ec_cuda.strm_task_mode == UCC_EC_CUDA_TASK_KERNEL) {
        return ucc_ec_cuda_post_kernel_stream_task(eee->dev_state, stream);
    } else {
        return ucc_ec_cuda_post_driver_stream_task(eee->dev_state, stream);
    }
}

ucc_status_t ucc_cuda_executor_persistent_wait_stop(ucc_ee_executor_t *executor)
{
    ucc_ec_cuda_executor_t *eee = ucc_derived_of(executor,
                                                 ucc_ec_cuda_executor_t);
    volatile ucc_ec_cuda_executor_state_t *st = &eee->state;

    ec_debug(&ucc_ec_cuda.super, "executor stop, eee: %p", eee);
    ucc_assert((*st != UCC_EC_CUDA_EXECUTOR_POSTED) &&
               (*st != UCC_EC_CUDA_EXECUTOR_SHUTDOWN));
    *st = UCC_EC_CUDA_EXECUTOR_SHUTDOWN;
    while(*st != UCC_EC_CUDA_EXECUTOR_SHUTDOWN_ACK) { }
    eee->super.ee_context = NULL;
    eee->state = UCC_EC_CUDA_EXECUTOR_INITIALIZED;

    return UCC_OK;
}
