/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CUDA_H_
#define UCC_MC_CUDA_H_

#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc_log.h"
#include <cuda_runtime.h>

typedef struct ucc_mc_cuda_config {
    ucc_mc_config_t super;
    unsigned long   reduce_num_blocks;
    int             reduce_num_threads;
} ucc_mc_cuda_config_t;

typedef struct ucc_mc_cuda {
    ucc_mc_base_t super;
    cudaStream_t  stream;
} ucc_mc_cuda_t;


ucc_status_t ucc_mc_cuda_reduce(const void *src1, const void *src2,
                                void *dst, size_t count, ucc_datatype_t dt,
                                ucc_reduction_op_t op);

extern ucc_mc_cuda_t ucc_mc_cuda;
#define CUDACHECK(cmd) do {                                                    \
        cudaError_t e = cmd;                                                   \
        if(e != cudaSuccess) {                                                 \
            mc_error(&ucc_mc_cuda.super, "cuda failed with ret:%d(%s)", e,     \
                     cudaGetErrorString(e));                                   \
            return UCC_ERR_NO_MESSAGE;                                         \
        }                                                                      \
} while(0)

#define MC_CUDA_CONFIG                                                         \
    (ucc_derived_of(ucc_mc_cuda.super.config, ucc_mc_cuda_config_t))

#endif
