/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "../ec_cuda.h"
#ifdef __cplusplus
}
#endif

__global__ void wait_kernel(volatile uint32_t *status) {
    ucc_status_t st;
    *status = UCC_EC_CUDA_TASK_STARTED;
    do {
        st = (ucc_status_t)*status;
    } while(st != UCC_EC_CUDA_TASK_COMPLETED);
    *status = UCC_EC_CUDA_TASK_COMPLETED_ACK;
    return;
}

__global__ void wait_kernel_nb(volatile uint32_t *status) {
    *status = UCC_EC_CUDA_TASK_COMPLETED_ACK;
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_ec_cuda_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 cudaStream_t stream)
{
    if (blocking_wait) {
        wait_kernel<<<1, 1, 0, stream>>>(status);
    } else {
        wait_kernel_nb<<<1, 1, 0, stream>>>(status);
    }
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
