/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_ATOMIC_H_
#define UCC_ATOMIC_H_

#include "config.h"
#include <ucs/arch/atomic.h>

#define ucc_atomic_add32          ucs_atomic_add32
#define ucc_atomic_fadd32         ucs_atomic_fadd32
#define ucc_atomic_sub32          ucs_atomic_sub32
#define ucc_atomic_add64          ucs_atomic_add64
#define ucc_atomic_sub64          ucs_atomic_sub64
#define ucc_atomic_cswap8         ucs_atomic_cswap8
#define ucc_atomic_cswap64        ucs_atomic_cswap64
#define ucc_atomic_bool_cswap8    ucs_atomic_bool_cswap8
#define ucc_atomic_bool_cswap64   ucs_atomic_bool_cswap64
#endif
