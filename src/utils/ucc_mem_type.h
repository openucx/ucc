/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MEM_TYPE_H_
#define UCC_MEM_TYPE_H_

#include "ucc/api/ucc.h"

/*
 * Internal extensions of ucc_memory_type_t used to represent special cases.
 * These are intentionally *not* part of the public ucc_memory_type_t enum.
 */
#define UCC_MEMORY_TYPE_ASYMMETRIC                                             \
    ((ucc_memory_type_t)((int)UCC_MEMORY_TYPE_LAST + 1))

#define UCC_MEMORY_TYPE_NOT_APPLY                                              \
    ((ucc_memory_type_t)((int)UCC_MEMORY_TYPE_LAST + 2))

#define UCC_MEM_TYPE_MASK_FULL (UCC_BIT(UCC_MEMORY_TYPE_HOST) |                \
                                UCC_BIT(UCC_MEMORY_TYPE_CUDA) |                \
                                UCC_BIT(UCC_MEMORY_TYPE_CUDA_MANAGED) |        \
                                UCC_BIT(UCC_MEMORY_TYPE_ROCM) |                \
                                UCC_BIT(UCC_MEMORY_TYPE_ROCM_MANAGED))


#endif

