/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "../mc_cuda.h"
#ifdef __cplusplus
}
#endif

__global__ void wait_kernel(volatile uint32_t *status) {
    ucc_status_t st;
    *status = UCC_MC_CUDA_TASK_STARTED;
    do {
        st = (ucc_status_t)*status;
    } while(st != UCC_MC_CUDA_TASK_COMPLETED);
    *status = UCC_MC_CUDA_TASK_COMPLETED_ACK;
    return;
}

__global__ void wait_kernel_nb(volatile uint32_t *status) {
    *status = UCC_MC_CUDA_TASK_COMPLETED_ACK;
    return;
}

__global__ void enqueue_kernel(volatile uint32_t *status) {
    *status = UCC_MC_CUDA_TASK_STARTED;
    return;
}

__global__ void sync_kernel(volatile uint32_t *status) {
    ucc_status_t st;
    do {
        st = (ucc_status_t)*status;
    } while(st != UCC_MC_CUDA_TASK_COMPLETED);
    *status = UCC_MC_CUDA_TASK_COMPLETED_ACK;
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_mc_cuda_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 cudaStream_t stream)
{
    if (blocking_wait) {
        wait_kernel<<<1, 1, 0, stream>>>(status);
    } else {
        wait_kernel_nb<<<1, 1, 0, stream>>>(status);
    }
    CUDACHECK(cudaGetLastError());
    return UCC_OK;
}

ucc_status_t ucc_mc_cuda_enqueue_kernel_stream_task(uint32_t *status,
                                                    cudaStream_t stream)
{
    enqueue_kernel<<<1, 1, 0, stream>>>(status);
    CUDACHECK(cudaGetLastError());
    return UCC_OK;
}

ucc_status_t ucc_mc_cuda_sync_kernel_stream_task(uint32_t *status,
                                                 cudaStream_t stream)
{
    sync_kernel<<<1, 1, 0, stream>>>(status);
    CUDACHECK(cudaGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
