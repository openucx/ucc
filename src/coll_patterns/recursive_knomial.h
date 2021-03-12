/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef RECURSIVE_KNOMIAL_H_
#define RECURSIVE_KNOMIAL_H_

#define UCC_KN_PEER_NULL ((ucc_rank_t)-1)
typedef uint16_t ucc_kn_radix_t;

enum {
    KN_NODE_BASE,  /*< Participates in the main loop of the recursive KN algorithm */
    KN_NODE_PROXY, /*< Participates in the main loop and receives/sends the data from/to EXTRA */
    KN_NODE_EXTRA  /*< Just sends/receives the data to/from its PROXY */
};

/**
 *  @param [in]  _size           Size of the team
 *  @param [in]  _radix          Knomial radix
 *  @param [out] _pow_radix_sup  Smallest integer such that _radix**_pow_radix_sup >= _size
 *  #param [out] _full_pow_size  Largest power of _radix <= size. Ie. it is equal to
                                 _radix**_pow_radix_sup if _radix**_pow_radix_sup == size, OR
                                 _radix**(_pow_radix_sup - 1) otherwise
*/
#define CALC_POW_RADIX_SUP(_size, _radix, _pow_radix_sup, _full_pow_size)      \
    do {                                                                       \
        int pk = 1;                                                            \
        int fs = _radix;                                                       \
        while (fs < _size) {                                                   \
            pk++;                                                              \
            fs *= _radix;                                                      \
        }                                                                      \
        _pow_radix_sup = pk;                                                   \
        _full_pow_size = (fs != _size) ? fs / _radix : fs;                     \
    } while (0)

typedef struct ucc_knomial_pattern {
    ucc_kn_radix_t radix;
    uint8_t        iteration;
    uint8_t        pow_radix_sup;
    uint8_t        node_type;
    ucc_rank_t     radix_pow;
    ucc_rank_t     n_extra; /**< number of "extra" ranks to be served by "proxies" */
} ucc_knomial_pattern_t;

/**
 *  Initializes recursive knomial tree attributes.
 *  @param [in]  radix Knomial radix
 *  @param [in]  rank  Rank in a team
 *  @param [in]  size  Team size
 *  @param [out] p     ucc_knomial_pattern
 */

static inline void ucc_knomial_pattern_init(ucc_rank_t size, ucc_rank_t rank,
                                            ucc_kn_radix_t radix,
                                            ucc_knomial_pattern_t *p)
{
    ucc_rank_t full_pow_size, n_full_subtrees;
    CALC_POW_RADIX_SUP(size, radix, p->pow_radix_sup, full_pow_size);
    n_full_subtrees  = size / full_pow_size;
    p->n_extra       = size - n_full_subtrees * full_pow_size;
    p->radix         = radix;
    p->iteration     = 0;
    p->radix_pow     = 1;
    p->node_type     = KN_NODE_BASE;
    if (rank < p->n_extra * 2) {
        p->node_type = (rank % 2) ? KN_NODE_EXTRA : KN_NODE_PROXY;
    }
}

static inline ucc_rank_t ucc_knomial_pattern_get_proxy(ucc_knomial_pattern_t *p,
                                                       ucc_rank_t rank)
{
    return rank - 1;
}

static inline ucc_rank_t ucc_knomial_pattern_get_extra(ucc_knomial_pattern_t *p,
                                                       ucc_rank_t rank)
{
    return rank + 1;
}

static inline int ucc_knomial_pattern_loop_done(ucc_knomial_pattern_t *p)
{
    ucc_assert(p->iteration <= p->pow_radix_sup);
    return p->iteration == p->pow_radix_sup;
}

static inline ucc_rank_t ucc_knomial_pattern_get_loop_peer(ucc_knomial_pattern_t *p,
                                                           ucc_rank_t rank,
                                                           ucc_rank_t size,
                                                           ucc_kn_radix_t loop_step)
{
    ucc_assert(p->node_type == KN_NODE_BASE ||
               p->node_type == KN_NODE_PROXY);
    ucc_assert(loop_step >= 1 && loop_step < p->radix);
    ucc_assert((rank >= p->n_extra * 2) || ((rank % 2) == 0));
    ucc_rank_t loop_rank = (rank < p->n_extra * 2) ? rank/2 : rank - p->n_extra;
    ucc_rank_t step_size = p->radix_pow * p->radix;
    ucc_rank_t peer      = (loop_rank + loop_step * p->radix_pow) % step_size +
        (loop_rank - loop_rank % step_size);

    return (peer >= (size - p->n_extra)) ? UCC_KN_PEER_NULL :
        (peer < p->n_extra) ? peer*2 : peer + p->n_extra;
}

static inline void ucc_knomial_pattern_next_iteration(ucc_knomial_pattern_t *p)
{
    p->iteration++;
    p->radix_pow *= p->radix;
    ucc_assert(p->iteration <= p->pow_radix_sup);
}
#endif
