/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CUDA_H_
#define UCC_MC_CUDA_H_

#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc_log.h"

typedef struct ucc_mc_cuda_config {
    ucc_mc_config_t super;
} ucc_mc_cuda_config_t;

typedef struct ucc_mc_cuda {
    ucc_mc_base_t super;
} ucc_mc_cuda_t;

extern ucc_mc_cuda_t ucc_mc_cuda;
#endif
