/**
 * Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CUDA_H_
#define UCC_MC_CUDA_H_

#include <cuda_runtime.h>
#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc_log.h"
#include "utils/ucc_mpool.h"
#include "utils/arch/cuda_def.h"
#include "mc_cuda_resources.h"

typedef struct ucc_mc_cuda {
    ucc_mc_base_t                  super;
    ucc_spinlock_t                 init_spinlock;
    ucc_thread_mode_t              thread_mode;
    ucc_mc_cuda_resources_hash_t  *resources_hash;
} ucc_mc_cuda_t;

extern ucc_mc_cuda_t ucc_mc_cuda;

#define MC_CUDA_CONFIG                                                         \
    (ucc_derived_of(ucc_mc_cuda.super.config, ucc_mc_cuda_config_t))

ucc_status_t ucc_mc_cuda_get_resources(ucc_mc_cuda_resources_t **resources);

ucc_status_t ucc_mc_cuda_memset(void *ptr, int val, size_t len);

#endif
