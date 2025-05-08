/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef BRUCK_ALLTOALL_H_
#define BRUCK_ALLTOALL_H_

#include "utils/ucc_math.h"

#define GET_NEXT_BRUCK_NUM(_num, _radix, _pow) \
    ((((_num) + 1) % (_pow))?((_num) + 1):(((_num) + 1) + (_pow) * ((_radix) - 1)))

#define GET_PREV_BRUCK_NUM(_num, _radix, _pow) \
    (((_num) % (_pow))?((_num) - 1):(((_num) - 1) - (_pow) * ((_radix) - 1)))

static inline ucc_rank_t get_bruck_step_start(uint32_t pow, uint32_t d)
{
    return pow * d;
}

static inline ucc_rank_t get_bruck_step_finish(ucc_rank_t n, uint32_t radix,
                                               uint32_t d, uint32_t pow)
{
    ucc_assume(pow > 0 && radix > 0);
    return ucc_min(n + pow - 1 - (n - d * pow) % (pow * radix), n);
}

static inline ucc_rank_t get_bruck_recv_peer(ucc_rank_t trank, ucc_rank_t tsize,
                                             ucc_rank_t step, uint32_t digit)
{
    return (trank - step * digit + tsize * digit) % tsize;
}

static inline ucc_rank_t get_bruck_send_peer(ucc_rank_t trank, ucc_rank_t tsize,
                                             ucc_rank_t step, uint32_t digit)
{
    return (trank + step * digit) % tsize;
}

#endif
