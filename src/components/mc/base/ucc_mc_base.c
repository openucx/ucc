/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_mc_base.h"

ucc_config_field_t ucc_mc_config_table[] = {
    {"LOG_LEVEL", "warn",
     "UCC logging level. Messages with a level higher or equal to the "
     "selected will be printed.\n"
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_mc_config_t, log_component), UCC_CONFIG_TYPE_LOG_COMP},

    {NULL}};

const char *ucc_memory_type_names[] = {
    [UCC_MEMORY_TYPE_HOST]         = "host",
    [UCC_MEMORY_TYPE_CUDA]         = "cuda",
    [UCC_MEMORY_TYPE_CUDA_MANAGED] = "cuda-managed",
    [UCC_MEMORY_TYPE_ROCM]         = "rocm",
    [UCC_MEMORY_TYPE_ROCM_MANAGED] = "rocm-managed",
    [UCC_MEMORY_TYPE_LAST]         = "unknown"};


const char *ucc_ee_type_names[] = {
    [UCC_EE_CUDA_STREAM] = "cuda stream",
    [UCC_EE_CPU_THREAD]  = "cpu thread"};
