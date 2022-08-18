/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CUDA_H_
#define UCC_MC_CUDA_H_

#include <cuComplex.h>
#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc_log.h"
#include "utils/ucc_mpool.h"
#include "utils/arch/cuda_def.h"
#include <cuda_runtime.h>

typedef struct ucc_mc_cuda_config {
    ucc_mc_config_t                super;
    size_t                         mpool_elem_size;
    int                            mpool_max_elems;
} ucc_mc_cuda_config_t;

typedef struct ucc_mc_cuda {
    ucc_mc_base_t                  super;
    int                            stream_initialized;
    cudaStream_t                   stream;
    ucc_mpool_t                    events;
    ucc_mpool_t                    strm_reqs;
    ucc_mpool_t                    mpool;
    int                            mpool_init_flag;
    ucc_spinlock_t                 init_spinlock;
    ucc_thread_mode_t              thread_mode;
} ucc_mc_cuda_t;

extern ucc_mc_cuda_t ucc_mc_cuda;

#define MC_CUDA_CONFIG                                                         \
    (ucc_derived_of(ucc_mc_cuda.super.config, ucc_mc_cuda_config_t))

#define UCC_MC_CUDA_INIT_STREAM() do {                                         \
    if (!ucc_mc_cuda.stream_initialized) {                                     \
        cudaError_t cuda_st = cudaSuccess;                                     \
        ucc_spin_lock(&ucc_mc_cuda.init_spinlock);                             \
        if (!ucc_mc_cuda.stream_initialized) {                                 \
            cuda_st = cudaStreamCreateWithFlags(&ucc_mc_cuda.stream,           \
                                                cudaStreamNonBlocking);        \
            ucc_mc_cuda.stream_initialized = 1;                                \
        }                                                                      \
        ucc_spin_unlock(&ucc_mc_cuda.init_spinlock);                           \
        if (ucc_unlikely(cudaSuccess != cuda_st)) {                     \
            return cuda_error_to_ucc_status(cuda_st);                   \
        }                                                               \
    }                                                                          \
} while(0)

#endif
