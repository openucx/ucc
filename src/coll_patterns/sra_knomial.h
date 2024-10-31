/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef SRA_KNOMIAL_H_
#define SRA_KNOMIAL_H_

#include "recursive_knomial.h"

/**
 * Computes actual radix at current iteration
 * @param [in] p ucc_knomial_pattern
 * @return radix
 */
static inline
ucc_rank_t ucc_kn_compute_step_radix(ucc_knomial_pattern_t *p)
{
    int n_full = ucc_kn_pattern_n_full(p);

    return p->radix_pow * p->radix >= p->size ? (n_full > 1 ? n_full : p->radix)
                                              : p->radix;
}

/* segment index in exchange group of tree */
static inline ucc_rank_t
ucc_kn_compute_seg_index(ucc_rank_t peer, ucc_rank_t kpow_num,
                         ucc_knomial_pattern_t *p)
{
    ucc_rank_t peer_position, peer_base_rank, peer_index;

    peer           = ucc_knomial_pattern_loop_rank(p, peer);
    peer_base_rank = ucc_align_down(peer, kpow_num * p->radix);
    peer_position  = peer_base_rank == 0 ? peer : peer % (peer_base_rank);
    peer_index     = peer_position / kpow_num;

    return peer_index;
}

/**
 * Computes segment size
 * @param [in] block_count size of the block
 * @param [in] radix tree radix
 * @param [si] si segment index
 * @return segment size
 */
static inline size_t
ucc_sra_kn_compute_seg_size(size_t block_count, ucc_kn_radix_t radix,
                            ucc_rank_t si)
{
    return ucc_buffer_block_count(block_count, radix, si);
}

/**
 * Computes segment offset
 * @param [in] block_count size of the block
 * @param [in] radix tree radix
 * @param [si] si segment index
 * @return segment offset
 */
static inline size_t
ucc_sra_kn_compute_seg_offset(size_t block_count, ucc_kn_radix_t radix,
                              ucc_rank_t si)
{
    return ucc_buffer_block_offset(block_count, radix, si);
}

static inline size_t
ucc_sra_kn_compute_block_count(size_t count, ucc_rank_t rank,
                               ucc_knomial_pattern_t *p)
{
    size_t     block_count = count;
    ucc_rank_t k_pow       = 1;
    ucc_rank_t i, my_si, my_seg_len, steps;

    steps = p->backward ? p->n_iters - p->iteration - 1 : p->iteration;
    for (i = 0; i < steps; i++) {
        my_si       = ucc_kn_compute_seg_index(rank, k_pow, p);
        my_seg_len  = ucc_sra_kn_compute_seg_size(block_count, p->radix, my_si);
        block_count = my_seg_len;
        k_pow *= p->radix;
    }
    return block_count;
}

static inline void
ucc_sra_kn_get_offset_and_seglen(size_t count, size_t dt_size, ucc_rank_t rank,
                                 ucc_rank_t size, ucc_kn_radix_t radix,
                                 ptrdiff_t *offset, size_t *seglen)
{
    ptrdiff_t             _offset     = 0;
    size_t                my_seg_len  = 0;
    ucc_rank_t            my_si, step_radix;
    size_t                my_seg_offset;
    ucc_knomial_pattern_t p;

    ucc_knomial_pattern_init(size, rank, radix, &p);
    if (KN_NODE_EXTRA == p.node_type) {
        goto out;
    }

    while (!ucc_knomial_pattern_loop_done(&p)) {
        step_radix    = ucc_kn_compute_step_radix(&p);
        my_si         = ucc_kn_compute_seg_index(rank, p.radix_pow, &p);
        my_seg_offset = ucc_sra_kn_compute_seg_offset(count, step_radix, my_si);
        count         = ucc_sra_kn_compute_seg_size(count, step_radix, my_si);
        _offset += my_seg_offset * dt_size;
        ucc_knomial_pattern_next_iteration(&p);
    }
    my_seg_len = count;

out:
    if (offset)
        *offset = _offset;
    if (seglen)
        *seglen = my_seg_len;
}

static inline ptrdiff_t
ucc_sra_kn_get_offset(size_t count, size_t dt_size, ucc_rank_t rank,
                      ucc_rank_t size, ucc_kn_radix_t radix)
{
    ptrdiff_t offset;
    ucc_sra_kn_get_offset_and_seglen(count, dt_size, rank, size, radix, &offset,
                                     NULL);
    return offset;
}

typedef struct ucc_kn_seg_desc {
    ucc_rank_t seg_size;
    ucc_rank_t seg_offset;
    ucc_rank_t seg_start_rank_loop;
    ucc_rank_t seg_end_rank_loop;
    ucc_rank_t seg_start;
    ucc_rank_t seg_end;
} ucc_kn_seg_desc_t;

static inline void
ucc_kn_seg_desc_compute(ucc_knomial_pattern_t *p, ucc_kn_seg_desc_t *seg,
                        ucc_rank_t peer)
{
    ucc_rank_t step_radix = ucc_kn_compute_step_radix(p);
    ucc_rank_t seg_index  = ucc_kn_compute_seg_index(peer, p->radix_pow, p);

    /* size of the peer's segment in "loop ranks" - ie how many ranks are
       aggregated under this segment */
    seg->seg_size = p->block_size / step_radix;
    /* offset of the peer's segment in "loop ranks" */
    seg->seg_offset = seg->seg_size * seg_index;
    /* start and end position of the segment in "loop ranks" */
    seg->seg_start_rank_loop = seg->seg_offset + p->block_offset;
    seg->seg_end_rank_loop   = seg->seg_start_rank_loop + seg->seg_size;
    /* start and end position of the segment in the original "ranks" */
    seg->seg_start =
        ucc_knomial_pattern_loop_rank_inv(p, seg->seg_start_rank_loop);
    seg->seg_end = ucc_knomial_pattern_loop_rank_inv(p, seg->seg_end_rank_loop);
}

static inline void
ucc_knx_block(ucc_rank_t rank, ucc_rank_t size, ucc_kn_radix_t radix,
              size_t count, int iter, size_t *b_count, ptrdiff_t *b_offset)
{
    ucc_rank_t            offset = 0;
    ucc_rank_t            block_count;
    ucc_kn_radix_t        step_radix;
    ucc_rank_t            my_si;
    ucc_knomial_pattern_t p;

    ucc_knomial_pattern_init(size, rank, radix, &p);
    if (KN_NODE_EXTRA == p.node_type) {
        *b_count = *b_offset = 0;
        return;
    }
    block_count = count;
    while (p.iteration < iter) {
        step_radix = ucc_kn_compute_step_radix(&p);
        my_si      = ucc_kn_compute_seg_index(rank, p.radix_pow, &p);
        offset += ucc_buffer_block_offset(block_count, step_radix, my_si);
        block_count = ucc_buffer_block_count(block_count, step_radix, my_si);
        ucc_knomial_pattern_next_iteration(&p);
    }
    *b_count  = block_count;
    *b_offset = offset;
}

static inline void
ucc_kn_g_pattern_init(ucc_rank_t size, ucc_rank_t rank, ucc_kn_radix_t radix,
                      size_t count, ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_no_extra(size, rank, radix, p);
    p->type         = KN_PATTERN_GATHER;
    p->count        = count;
    p->block_size   = p->radix_pow * radix;
    p->block_offset = ucc_knomial_pattern_loop_rank(p, rank) / p->block_size *
                      p->block_size;
}

static inline void
ucc_kn_gx_pattern_init(ucc_rank_t size, ucc_rank_t rank, ucc_kn_radix_t radix,
                       size_t count, ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_backward(size, rank, radix, p);
    p->type  = KN_PATTERN_GATHERX;
    p->count = count;
    if (p->node_type != KN_NODE_EXTRA) {
        p->block_size = ucc_kn_compute_step_radix(p);
        ucc_knx_block(rank, size, radix, count, p->n_iters - 1,
                      &p->block_size_counts, &p->block_offset);

    }

}

static inline void
ucc_kn_g_pattern_peer_seg(ucc_rank_t peer, ucc_knomial_pattern_t *p,
                          size_t *seg_count, ptrdiff_t *seg_offset)
{
    ucc_rank_t step_radix, seg_index;

    *seg_count = 0;
    *seg_offset = 0;
    switch (p->type) {
    case KN_PATTERN_GATHER:
        *seg_count = ucc_min(p->radix_pow, p->size - peer) * (p->count / p->size);
        *seg_offset = peer * (p->count / p->size);
        return;
    case KN_PATTERN_GATHERX:
        step_radix = ucc_kn_compute_step_radix(p);
        seg_index = ucc_kn_compute_seg_index(peer, p->radix_pow, p);
        *seg_offset = ucc_buffer_block_offset(p->block_size_counts, step_radix,
                                              seg_index) + p->block_offset;
        *seg_count = ucc_buffer_block_count(p->block_size_counts, step_radix,
                                            seg_index);
        return;
    default:
        ucc_assert(0);
    }
}

static inline void ucc_kn_g_pattern_next_iter(ucc_knomial_pattern_t *p)
{
    ucc_rank_t rank;
    if (p->type == KN_PATTERN_GATHERX) {
        ucc_knomial_pattern_next_iteration_backward(p);

        if (!ucc_knomial_pattern_loop_done(p)) {
            ucc_knx_block(p->rank, p->size, p->radix, p->count,
                          p->n_iters - 1 - p->iteration,
                          &p->block_size_counts, &p->block_offset);
        }
    } else {
        rank = ucc_knomial_pattern_loop_rank(p, p->rank);
        ucc_knomial_pattern_next_iteration(p);

        if (!ucc_knomial_pattern_loop_done(p)) {
            p->block_size *= ucc_kn_compute_step_radix(p);
            p->block_offset = rank / p->block_size * p->block_size;
        }
    }
}

static inline void
ucc_kn_ag_pattern_init(ucc_rank_t size, ucc_rank_t rank, ucc_kn_radix_t radix,
                       size_t count, ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init(size, rank, radix, p);
    p->type         = KN_PATTERN_ALLGATHER;
    p->count        = count;
    p->block_size   = p->radix_pow * radix;
    p->block_offset = ucc_knomial_pattern_loop_rank(p, rank) / p->block_size *
                      p->block_size;
}

static inline void
ucc_kn_agx_pattern_init(ucc_rank_t size, ucc_rank_t rank, ucc_kn_radix_t radix,
                        size_t count, ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_backward(size, rank, radix, p);
    p->type  = KN_PATTERN_ALLGATHERX;
    p->count = count;
    if (p->node_type != KN_NODE_EXTRA) {
        p->block_size = ucc_kn_compute_step_radix(p);
        ucc_knx_block(rank, size, radix, count, p->n_iters - 1,
                      &p->block_size_counts, &p->block_offset);

    }
}

static inline void
ucc_kn_agv_pattern_init(ucc_rank_t size, ucc_rank_t rank, ucc_kn_radix_t radix,
                        ucc_count_t *counts, int is64,
                        ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init(size, rank, radix, p);
    p->type         = KN_PATTERN_ALLGATHERV;
    p->counts       = counts;
    p->is64         = is64;
    p->block_size   = p->radix_pow * radix;
    p->block_offset = ucc_knomial_pattern_loop_rank(p, rank) / p->block_size *
                      p->block_size;
}

static inline void
ucc_kn_ag_pattern_peer_seg(ucc_rank_t peer, ucc_knomial_pattern_t *p,
                           size_t *seg_count, ptrdiff_t *seg_offset)
{
    ucc_rank_t step_radix, seg_index;
    ucc_kn_seg_desc_t s;

    *seg_count = 0;
    *seg_offset = 0;
    switch (p->type) {
    case KN_PATTERN_ALLGATHERX:
        step_radix = ucc_kn_compute_step_radix(p);
        seg_index = ucc_kn_compute_seg_index(peer, p->radix_pow, p);
        *seg_offset = ucc_buffer_block_offset(p->block_size_counts, step_radix,
                                              seg_index) + p->block_offset;
        *seg_count = ucc_buffer_block_count(p->block_size_counts, step_radix,
                                            seg_index);
        return;
    case KN_PATTERN_ALLGATHER:
        ucc_kn_seg_desc_compute(p, &s, peer);
        *seg_offset = ucc_buffer_block_offset(p->count, p->size, s.seg_start);
        *seg_count = ucc_buffer_block_offset(p->count, p->size, s.seg_end) -
                     *seg_offset;
        return;
    case KN_PATTERN_ALLGATHERV:
        ucc_kn_seg_desc_compute(p, &s, peer);
        *seg_offset = ucc_buffer_vector_block_offset(p->counts, p->is64,
                                                     s.seg_start);
        *seg_count = ucc_buffer_vector_block_offset(p->counts, p->is64,
                                                    s.seg_end) - *seg_offset;
        return;
    default:
        ucc_assert(0);
    }
}

static inline void ucc_kn_ag_pattern_next_iter(ucc_knomial_pattern_t *p)
{
    ucc_rank_t rank;
    if (p->type == KN_PATTERN_ALLGATHERX) {
        ucc_knomial_pattern_next_iteration_backward(p);

        if (!ucc_knomial_pattern_loop_done(p)) {
            ucc_knx_block(p->rank, p->size, p->radix, p->count,
                          p->n_iters - 1 - p->iteration,
                          &p->block_size_counts, &p->block_offset);
        }
    } else {
        rank = ucc_knomial_pattern_loop_rank(p, p->rank);
        ucc_knomial_pattern_next_iteration(p);

        if (!ucc_knomial_pattern_loop_done(p)) {
            p->block_size *= ucc_kn_compute_step_radix(p);
            p->block_offset = rank / p->block_size * p->block_size;
        }
    }
}

static inline void ucc_kn_rs_pattern_init(ucc_rank_t size, ucc_rank_t rank,
                                          ucc_kn_radix_t radix, size_t count,
                                          ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_backward(size, rank, radix, p);
    p->type              = KN_PATTERN_REDUCE_SCATTER;
    p->count             = count;
    p->block_size_counts = count;
    p->block_size        = size - p->n_extra;
    p->block_offset      = 0;
}

static inline void ucc_kn_rsx_pattern_init(ucc_rank_t size, ucc_rank_t rank,
                                           ucc_kn_radix_t radix, size_t count,
                                           ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init(size, rank, radix, p);
    p->type              = KN_PATTERN_REDUCE_SCATTERX;
    p->count             = count;
    p->block_size_counts = count;
    p->block_size        = size - p->n_extra;
}

static inline void
ucc_kn_rs_pattern_peer_seg(ucc_rank_t peer, ucc_knomial_pattern_t *p,
                           size_t *peer_seg_count, ptrdiff_t *peer_seg_offset)
{
    ucc_rank_t step_radix, seg_index;
    ucc_kn_seg_desc_t s;
    ucc_rank_t block_offset_inv;
    /* offset of the segment in counts of datatypes from the
       start of the buffer */
    size_t peer_seg_offset_base;
    /* offset of the current block in counts of datatypes from the
       start of the buffer */
    size_t block_offset_counts;

    *peer_seg_count  = 0;
    *peer_seg_offset = 0;

    switch (p->type) {
    case KN_PATTERN_REDUCE_SCATTERX:
        step_radix = ucc_kn_compute_step_radix(p);
        seg_index  = ucc_kn_compute_seg_index(peer, p->radix_pow, p);
        *peer_seg_offset = ucc_buffer_block_offset(p->block_size_counts,
                                                   step_radix, seg_index);
        *peer_seg_count  = ucc_buffer_block_count(p->block_size_counts,
                                                  step_radix, seg_index);
        return;
    case KN_PATTERN_REDUCE_SCATTER:
        ucc_kn_seg_desc_compute(p, &s, peer);
        block_offset_inv = ucc_knomial_pattern_loop_rank_inv(p, p->block_offset);
        peer_seg_offset_base = ucc_buffer_block_offset(p->count, p->size,
                                                       s.seg_start);
        *peer_seg_count      = ucc_buffer_block_offset(p->count, p->size,
                                                       s.seg_end) -
                               peer_seg_offset_base;
        block_offset_counts = ucc_buffer_block_offset(p->count, p->size,
                                                      block_offset_inv);
        *peer_seg_offset = peer_seg_offset_base - block_offset_counts;
        return;
    case KN_PATTERN_REDUCE_SCATTERV:
        /* not implemented */
        ucc_assert(0);
    default:
        ucc_assert(0);
    }
}

static inline void ucc_kn_rs_pattern_next_iter(ucc_knomial_pattern_t *p)
{
    size_t bs;
    ptrdiff_t offset;
    ucc_kn_seg_desc_t s;

    ucc_kn_rs_pattern_peer_seg(p->rank, p, &bs, &offset);
    p->block_size_counts = bs;

    switch (p->type) {
    case KN_PATTERN_REDUCE_SCATTERX:
        p->block_offset += offset;
        ucc_knomial_pattern_next_iteration(p);
        return;
    case KN_PATTERN_REDUCE_SCATTER:
        ucc_kn_seg_desc_compute(p, &s, p->rank);
        p->block_size    = s.seg_size;
        p->block_offset += s.seg_offset;
        ucc_knomial_pattern_next_iteration_backward(p);
        return;
    case KN_PATTERN_REDUCE_SCATTERV:
        /* not implemented */
        ucc_assert(0);
    default:
        ucc_assert(0);
    }
}

static inline void ucc_kn_rs_pattern_extra_seg(ucc_knomial_pattern_t *p,
                                               size_t *seg_count,
                                               ptrdiff_t *seg_offset)
{
    switch (p->type) {
    case KN_PATTERN_REDUCE_SCATTER:
        *seg_offset = ucc_buffer_block_count(p->count, p->size, p->rank);
        *seg_count  = ucc_buffer_block_count(
            p->count, p->size, ucc_knomial_pattern_get_extra(p, p->rank));
        return;
    default:
        ucc_assert(0);
    }
}

#endif
