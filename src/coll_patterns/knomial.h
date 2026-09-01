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
    const ucc_kn_radix_t *radices; /* borrowed when n_radices > 1 */
    ucc_kn_radix_t        radix;   /* inline when n_radices == 1 */
    uint8_t               n_radices;
} ucc_kn_radix_seq_t;

static inline ucc_kn_radix_seq_t
ucc_kn_radix_seq_from_radix(ucc_kn_radix_t radix)
{
    ucc_kn_radix_seq_t seq = {0};

    seq.radix      = radix;
    seq.n_radices = 1;
    return seq;
}

static inline ucc_kn_radix_t
ucc_kn_radix_seq_get(const ucc_kn_radix_seq_t *seq, uint8_t index)
{
    return seq->n_radices == 1 ? seq->radix : seq->radices[index];
}

static inline int
ucc_kn_radix_seq_is_valid(const ucc_kn_radix_seq_t *seq, ucc_rank_t team_size)
{
    ucc_rank_t product = 1;
    uint8_t    i;

    if (seq->n_radices == 0) {
        return 1;
    }
    if (seq->n_radices > UCC_KN_MAX_RADIX_PHASES ||
        (seq->n_radices > 1 && !seq->radices)) {
        return 0;
    }
    if (seq->n_radices == 1) {
        return seq->radix >= 2;
    }
    for (i = 0; i < seq->n_radices; i++) {
        if (seq->radices[i] < 2 ||
            product > team_size / seq->radices[i]) {
            return 0;
        }
        product *= seq->radices[i];
    }
    return product == team_size;
}

#endif
