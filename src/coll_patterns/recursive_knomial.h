/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef RECURSIVE_KNOMIAL_H_
#define RECURSIVE_KNOMIAL_H_

#define UCC_KN_PEER_NULL ((ucc_rank_t)-1)
typedef uint16_t ucc_kn_radix_t;

enum {
    KN_NODE_BASE,  /* Participates in the main loop of the recursive KN algorithm */
    KN_NODE_PROXY, /* Participates in the main loop and receives/sends the data from/to EXTRA */
    KN_NODE_EXTRA  /* Just sends/receives the data to/from its PROXY */
};

enum {
    KN_PATTERN_REDUCE_SCATTER = 1,
    KN_PATTERN_REDUCE_SCATTERX,
    KN_PATTERN_REDUCE_SCATTERV,
    KN_PATTERN_ALLGATHER,
    KN_PATTERN_ALLGATHERV,
    KN_PATTERN_ALLGATHERX,
    KN_PATTERN_GATHER,
    KN_PATTERN_GATHERX,
};

typedef struct ucc_knomial_pattern {

    ucc_kn_radix_t radix;         /* knomial tree radix */
    uint8_t        type;          /* pattern type */
    uint8_t        iteration;     /* current iteration */
    uint8_t        n_iters;       /* number of iterations in knomial algorithm */
    uint8_t        pow_radix_sup; /* smallest integer N such that (radix ** N) >= size */
    uint8_t        node_type;     /* type of current rank: BASE, PROXY or EXTRA */
    uint8_t        backward;      /* boolean, iteration direction */
    ucc_rank_t     radix_pow;     /* power of radix for current algorithm iteration
                                   * forward: initial value is 1
                                   * backward: initial valus is full_pow_size if have >1 full subtrees, OR
                                   *           (full_pow_size / radix) otherwise
                                   */
    ucc_rank_t     full_pow_size; /* largest power of radix <= size. It is equal to
                                   * (radix ** pow_radix_sup) if (radix ** pow_radix_sup) == size, OR
                                   * (radix ** (_pow_radix_sup - 1)) otherwise
                                   */
    ucc_rank_t     size;          /* total number of ranks */
    ucc_rank_t     rank;          /* process rank */
    ucc_rank_t     n_extra;       /* number of "extra" ranks to be served by "proxies" */
    size_t         block_size_counts;
    size_t         count;         /* collective buffer size */
    ucc_count_t   *counts;
    ucc_rank_t     block_size;
    ptrdiff_t      block_offset;
    int            is64;
} ucc_knomial_pattern_t;

/**
 * Calculate number of full subtrees
 * @param [in] p ucc_knomial_pattern
 * @return number of full subtrees
 */
static inline ucc_rank_t ucc_kn_pattern_n_full(ucc_knomial_pattern_t *p)
{
    return p->size / p->full_pow_size;
}

static inline ucc_rank_t ucc_kn_pattern_radix_pow_init(ucc_knomial_pattern_t *p,
                                                       int backward)
{
    ucc_rank_t n_full = ucc_kn_pattern_n_full(p);

    return backward ? ((n_full == 1) ? p->full_pow_size / p->radix
                       : p->full_pow_size) : 1;
}

/**
 *  Initializes recursive knomial tree attributes.
 *  @param [in]  radix Knomial radix
 *  @param [in]  rank  Rank in a team
 *  @param [in]  size  Team size
 *  @param [out] p     ucc_knomial_pattern
 */
static inline void
ucc_knomial_pattern_init_impl(ucc_rank_t size, ucc_rank_t rank,
                              ucc_kn_radix_t radix, ucc_knomial_pattern_t *p,
                              int backward, int has_extra)
{
    ucc_rank_t fs = radix;
    ucc_rank_t n_full_subtrees;

    p->pow_radix_sup = 1;
    while (fs < size) {
        p->pow_radix_sup++;
        fs *= radix;
    }
    p->full_pow_size = (fs != size) ? fs / radix : fs;
    p->radix         = radix;
    p->size          = size;
    p->rank          = rank;
    p->backward      = backward;
    p->iteration     = 0;
    n_full_subtrees  = ucc_kn_pattern_n_full(p);
    p->n_extra       = has_extra ? size - n_full_subtrees * p->full_pow_size : 0;
    p->n_iters       = (p->n_extra && n_full_subtrees == 1) ?
        p->pow_radix_sup - 1 : p->pow_radix_sup;
    p->radix_pow     = ucc_kn_pattern_radix_pow_init(p, backward);
    p->node_type     = KN_NODE_BASE;
    if (rank < p->n_extra * 2) {
        p->node_type = (rank % 2) ? KN_NODE_EXTRA : KN_NODE_PROXY;
    }
}

static inline void
ucc_knomial_pattern_init_backward(ucc_rank_t size, ucc_rank_t rank,
                                  ucc_kn_radix_t radix,
                                  ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_impl(size, rank, radix, p, 1, 1);
}

static inline void
ucc_knomial_pattern_init(ucc_rank_t size, ucc_rank_t rank, ucc_kn_radix_t radix,
                         ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_impl(size, rank, radix, p, 0, 1);
}

static inline void
ucc_knomial_pattern_init_no_extra(ucc_rank_t size, ucc_rank_t rank,
                                  ucc_kn_radix_t radix,
                                  ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_impl(size, rank, radix, p, 0, 0);
}

static inline ucc_rank_t
ucc_knomial_pattern_get_proxy(ucc_knomial_pattern_t *p, ucc_rank_t rank)
{
    return rank - 1;
}

static inline ucc_rank_t
ucc_knomial_pattern_get_extra(ucc_knomial_pattern_t *p, ucc_rank_t rank)
{
    return rank + 1;
}

static inline int
ucc_knomial_pattern_loop_done(ucc_knomial_pattern_t *p)
{
    return p->iteration == p->n_iters;
}

static inline int
ucc_knomial_pattern_loop_first_iteration(ucc_knomial_pattern_t *p)
{
    return p->iteration == 0;
}

static inline int
ucc_knomial_pattern_loop_last_iteration(ucc_knomial_pattern_t *p)
{
    return p->iteration == p->n_iters - 1;
}

/* returns new rank id by excluding all extra ranks */
static inline ucc_rank_t
ucc_knomial_pattern_loop_rank(ucc_knomial_pattern_t *p, ucc_rank_t rank)
{
    return (rank < p->n_extra * 2) ? rank / 2 : rank - p->n_extra;
}

/* returns original rank id */
static inline ucc_rank_t
ucc_knomial_pattern_loop_rank_inv(ucc_knomial_pattern_t *p, ucc_rank_t rank)
{
    return (rank < p->n_extra) ? rank * 2 : rank + p->n_extra;
}

static inline ucc_rank_t
ucc_knomial_pattern_get_loop_peer(ucc_knomial_pattern_t *p, ucc_rank_t rank,
                                  ucc_kn_radix_t loop_step)
{
    ucc_assert(p->node_type == KN_NODE_BASE || p->node_type == KN_NODE_PROXY);
    ucc_assert(loop_step >= 1 && loop_step < p->radix);
    ucc_assert((rank >= p->n_extra * 2) || ((rank % 2) == 0));

    ucc_rank_t loop_rank = ucc_knomial_pattern_loop_rank(p, rank);
    ucc_rank_t step_size = p->radix_pow * p->radix;
    ucc_rank_t peer      = (loop_rank + loop_step * p->radix_pow) % step_size +
                           ucc_align_down(loop_rank, step_size);

    return (peer >= (p->size - p->n_extra)) ? UCC_KN_PEER_NULL:
           ucc_knomial_pattern_loop_rank_inv(p, peer);
}

static inline ucc_rank_t
ucc_knomial_pattern_get_base_rank(ucc_knomial_pattern_t *p, ucc_rank_t rank)
{
    ucc_rank_t step_size = p->radix_pow * p->radix;
    ucc_rank_t lrank;
    ucc_kn_radix_t s;

    lrank = ucc_knomial_pattern_loop_rank(p, rank);
    s = ucc_div_round_up(step_size - (lrank % step_size), p->radix_pow);

    if (s == p->radix) {
        return rank;
    } else {
        return ucc_knomial_pattern_get_loop_peer(p, rank, s);
    }
}

/* return the index of rank in the loop assuming smallest rank has index 0 */
static inline ucc_kn_radix_t
ucc_knomial_pattern_get_loop_index(ucc_knomial_pattern_t *p, ucc_rank_t rank)
{
    ucc_rank_t base_rank = ucc_knomial_pattern_get_base_rank(p, rank);
    ucc_rank_t rank0 = ucc_knomial_pattern_loop_rank(p, base_rank);
    ucc_rank_t cur_rank = ucc_knomial_pattern_loop_rank(p, rank);

    return (cur_rank - rank0) / p->radix_pow;
}

static inline void
ucc_knomial_pattern_next_iteration(ucc_knomial_pattern_t *p)
{
    p->iteration++;
    p->radix_pow *= p->radix;
}

static inline void ucc_knomial_pattern_prev_iteration(ucc_knomial_pattern_t *p)
{
    p->iteration--;
    p->radix_pow /= p->radix;
}

static inline void
ucc_knomial_pattern_next_iteration_backward(ucc_knomial_pattern_t *p)
{
    p->iteration++;
    p->radix_pow /= p->radix;
}

static inline ucc_kn_radix_t
ucc_knomial_pattern_get_min_radix(ucc_kn_radix_t cfg_radix,
                                  ucc_rank_t team_size, size_t count)
{
	ucc_kn_radix_t radix = ucc_min(cfg_radix, team_size);

    if (((count + radix - 1) / radix * (radix - 1) > count) ||
        ((radix - 1) > count)) {
    	radix = 2;
    }
    return radix;
}

/* Calculates for each rank at which distance it should receive */
static inline ucc_rank_t
ucc_knomial_calc_recv_dist(ucc_rank_t team_size, ucc_rank_t rank,
                           ucc_rank_t radix, ucc_rank_t root)
{
    ucc_rank_t root_base = 0;
    ucc_rank_t dist      = 1;

    if (rank == root) {
        return 0;
    }

    while (dist <= team_size) {
        if (rank < root_base + radix * dist) {
            break;
        }
        dist *= radix;
    }
    return dist;
}

/* Calculates (sub) opt radix for Allreduce SRA and Bcast SAG,
   by minimizing n_extra ranks */
static inline ucc_rank_t ucc_kn_get_opt_radix(ucc_rank_t team_size,
                                              ucc_kn_radix_t min_radix,
                                              ucc_kn_radix_t max_radix)
{
    ucc_rank_t     n_extra = 0, min_val = team_size;
    ucc_kn_radix_t min_i   = min_radix;
    ucc_kn_radix_t max_r   = ucc_max(max_radix, min_radix);
    ucc_kn_radix_t r;
    ucc_rank_t     fs;

    for (r = min_radix; r <= max_r; r++) {
        fs = r;
        while (fs < team_size) {
            fs = fs * r;
        }
        fs      = (fs == team_size) ? fs : fs / r;
        n_extra = team_size - (team_size / fs) * fs;
        if (n_extra == 0) {
            return r;
        }
        if (n_extra < min_val) {
            min_val = n_extra;
            min_i   = r;
        }
    }
    return min_i;
}

/* A set of convenience macros used to implement sw based progress
   of the algorithms that use kn pattern */
enum {
    UCC_KN_PHASE_INIT,
    UCC_KN_PHASE_LOOP,         /* main loop of recursive k-ing */
    UCC_KN_PHASE_REDUCE,       /* reduce data received from peer */
    UCC_KN_PHASE_EXTRA,        /* recv from extra rank */
    UCC_KN_PHASE_EXTRA_REDUCE, /* reduce data received from extra rank */
    UCC_KN_PHASE_PROXY,        /* recv from proxy rank */
    UCC_KN_PHASE_COMPLETE,     /* any work after main loop, e.g. memcpy */
};

#define UCC_KN_CHECK_PHASE(_p)                                                 \
    case _p:                                                                   \
        goto _p;

#define UCC_KN_REDUCE_GOTO_PHASE(_phase)                                       \
    do {                                                                       \
        switch (_phase) {                                                      \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_EXTRA);                            \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_EXTRA_REDUCE);                     \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_LOOP);                             \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_REDUCE);                           \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_PROXY);                            \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_COMPLETE);                         \
        case UCC_KN_PHASE_INIT:                                                \
            break;                                                             \
        };                                                                     \
    } while (0)

#define UCC_KN_GOTO_PHASE(_phase)                                              \
    do {                                                                       \
        switch (_phase) {                                                      \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_EXTRA);                            \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_LOOP);                             \
            UCC_KN_CHECK_PHASE(UCC_KN_PHASE_PROXY);                            \
        case UCC_KN_PHASE_INIT:                                                \
            break;                                                             \
        };                                                                     \
    } while (0)

#endif
