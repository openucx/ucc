/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef ALLGATHER_KNOMIAL_SELECT_H_
#define ALLGATHER_KNOMIAL_SELECT_H_

#include "coll_patterns/knomial.h"
#include <stddef.h>

#define UCC_TL_UCP_ALLGATHER_KN_LARGE_MSG_SIZE (1ull << 30)

/* Returns 1 when an exact sequence is found. A mixed result borrows radices;
 * a uniform result is stored inline. */
int ucc_tl_ucp_allgather_knomial_select_radix_seq(
    ucc_rank_t team_size, size_t msg_size, ucc_kn_radix_t *radices,
    ucc_kn_radix_seq_t *radix_seq);

#endif
