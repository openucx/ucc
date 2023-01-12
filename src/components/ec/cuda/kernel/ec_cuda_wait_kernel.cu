/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UINT32_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif
#include "../ec_cuda.h"
#ifdef __cplusplus
}
#endif

__global__ void wait_kernel(volatile ucc_ec_cuda_executor_state_t *state) {
    ucc_ec_cuda_executor_state_t st;

    *state = UCC_EC_CUDA_EXECUTOR_STARTED;
    do {
        st = *state;
    } while (st != UCC_EC_CUDA_EXECUTOR_SHUTDOWN);
    *state = UCC_EC_CUDA_EXECUTOR_SHUTDOWN_ACK;
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t
ucc_ec_cuda_post_kernel_stream_task(ucc_ec_cuda_executor_state_t *state,
                                    cudaStream_t stream)
{
    wait_kernel<<<1, 1, 0, stream>>>(state);
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
