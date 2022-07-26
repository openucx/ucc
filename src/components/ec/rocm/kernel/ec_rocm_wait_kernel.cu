/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ec_rocm.h"

__global__ void wait_kernel(volatile uint32_t *status) {
    ucc_status_t st;
    *status = UCC_EC_ROCM_TASK_STARTED;
    do {
        st = (ucc_status_t)*status;
    } while(st != UCC_EC_ROCM_TASK_COMPLETED);
    *status = UCC_EC_ROCM_TASK_COMPLETED_ACK;
    return;
}

__global__ void wait_kernel_nb(volatile uint32_t *status) {
    *status = UCC_EC_ROCM_TASK_COMPLETED_ACK;
    return;
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_ec_rocm_post_kernel_stream_task(uint32_t *status,
                                                 int blocking_wait,
                                                 hipStream_t stream)
{
    if (blocking_wait) {
        wait_kernel<<<1, 1, 0, stream>>>(status);
    } else {
        wait_kernel_nb<<<1, 1, 0, stream>>>(status);
    }
    ROCMCHECK(hipGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
