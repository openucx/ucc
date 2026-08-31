/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef KNOMIAL_H_
#define KNOMIAL_H_

#include "utils/ucc_datastruct.h"
#include <limits.h>

typedef uint16_t ucc_kn_radix_t;
#define UCC_KN_MAX_RADIX_PHASES (sizeof(ucc_rank_t) * CHAR_BIT - 1)

typedef struct ucc_kn_radix_seq {
    ucc_kn_radix_t radices[UCC_KN_MAX_RADIX_PHASES];
    uint8_t        n_radices;
} ucc_kn_radix_seq_t;

#endif
