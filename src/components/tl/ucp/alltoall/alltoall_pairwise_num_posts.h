/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALL_PAIRWISE_NUM_POSTS_H_
#define ALLTOALL_PAIRWISE_NUM_POSTS_H_

#include "utils/ucc_math.h"

#define UCC_TL_UCP_ALLTOALL_TOTAL_SMALL 66000
#define UCC_TL_UCP_ALLTOALL_PEER_64K    (64 * 1024)
#define UCC_TL_UCP_ALLTOALL_PEER_1M     (1 * 1024 * 1024)
#define UCC_TL_UCP_ALLTOALL_PEER_4M     (4 * 1024 * 1024)
#define UCC_TL_UCP_ALLTOALL_PEER_8M     (8 * 1024 * 1024)

/*
 * Preserve the existing tiny-message and small-team behavior, then use coarse,
 * benchmark-informed per-peer bands to limit outstanding sends and receives.
 */
static inline ucc_rank_t
ucc_tl_ucp_alltoall_pairwise_auto_num_posts(ucc_rank_t tsize,
                                             size_t total_size,
                                             size_t peer_size)
{
    if (total_size <= UCC_TL_UCP_ALLTOALL_TOTAL_SMALL || tsize <= 8) {
        return tsize;
    }

    if (peer_size <= UCC_TL_UCP_ALLTOALL_PEER_64K) {
        return ucc_min(tsize, 32);
    }

    if (peer_size <= UCC_TL_UCP_ALLTOALL_PEER_1M) {
        return ucc_min(tsize, 16);
    }

    if (peer_size <= UCC_TL_UCP_ALLTOALL_PEER_4M) {
        return tsize <= 16 ? tsize : 8;
    }

    if (peer_size <= UCC_TL_UCP_ALLTOALL_PEER_8M) {
        return ucc_min(tsize, 4);
    }

    return tsize <= 32 ? 4 : 1;
}

#endif
