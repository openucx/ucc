/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SRA_KNOMIAL_H_
#define SRA_KNOMIAL_H_

#include "recursive_knomial.h"

static inline ucc_rank_t ucc_kn_compute_step_radix(ucc_rank_t             rank,
                                                   ucc_rank_t             size,
                                                   ucc_knomial_pattern_t *p)
{
    int n_full = ucc_kn_pattern_n_full(p);

    return p->radix_pow * p->radix >= p->size ? (n_full > 1 ? n_full : p->radix)
                                              : p->radix;
}

// segment index in exchange group of tree
static inline ucc_rank_t ucc_kn_compute_seg_index(ucc_rank_t peer,
                                                  ucc_rank_t kpow_num,
                                                  ucc_knomial_pattern_t *p)
{
    ucc_rank_t peer_base, peer_position, peer_base_rank, peer_index;

    peer           = ucc_knomial_pattern_loop_rank(p, peer);
    peer_base      = peer / (kpow_num * p->radix);
    peer_base_rank = peer_base * kpow_num * p->radix;
    peer_position  = peer_base_rank == 0 ? peer : peer % (peer_base_rank);
    peer_index     = peer_position / kpow_num;

    return peer_index;
}

static inline size_t vector_block_offset(ucc_knomial_pattern_t *p,
                                         ucc_rank_t             rank)
{
    size_t offset = 0;
    //TODO switch to prefix sum
    if (p->is64) {
        for (int i = 0; i < rank; i++) {
            offset += ((uint64_t *)p->counts)[i];
        }
    } else {
        for (int i = 0; i < rank; i++) {
            offset += ((uint32_t *)p->counts)[i];
        }
    }
    return offset;
}

static inline ucc_rank_t ucc_knx_rank(ucc_rank_t rank, ucc_rank_t size,
                                      ucc_kn_radix_t radix)
{
    ucc_rank_t            offset = 0;
    ucc_rank_t            block_count;
    ucc_kn_radix_t        step_radix;
    ucc_rank_t            my_si;
    ucc_knomial_pattern_t p;

    ucc_knomial_pattern_init(size, rank, radix, &p);
    if (KN_NODE_EXTRA == p.node_type) {
        return UCC_RANK_INVALID;
    }
    block_count = size - p.n_extra;
    while (!ucc_knomial_pattern_loop_done(&p)) {
        step_radix = ucc_kn_compute_step_radix(rank, size, &p);
        my_si      = ucc_kn_compute_seg_index(rank, p.radix_pow, &p);
        offset += ucc_buffer_block_offset(block_count, step_radix, my_si);
        block_count = ucc_buffer_block_count(block_count, step_radix, my_si);
        ucc_knomial_pattern_next_iteration(&p);
    }
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

static inline void ucc_kn_seg_desc_compute(ucc_knomial_pattern_t *p,
                                           ucc_kn_seg_desc_t *    seg,
                                           ucc_rank_t             peer)
{
    ucc_rank_t step_radix = ucc_kn_compute_step_radix(p->rank, p->size, p);
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

/* ========================= ALLGAHTER ============================  */

static inline void ucc_kn_ag_pattern_init(ucc_rank_t size, ucc_rank_t rank,
                                          ucc_kn_radix_t radix, size_t count,
                                          ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init(size, rank, radix, p);
    p->type       = KN_PATTERN_ALLGATHER;
    p->count      = count;
    p->block_size = p->radix_pow * radix;
    p->block_offset =
        ucc_knomial_pattern_loop_rank(p, rank) / p->block_size * p->block_size;
}

static inline void ucc_kn_agv_pattern_init(ucc_rank_t size, ucc_rank_t rank,
                                           ucc_kn_radix_t radix,
                                           ucc_count_t *counts, int is64,
                                           ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init(size, rank, radix, p);
    p->type       = KN_PATTERN_ALLGATHERV;
    p->counts     = counts;
    p->is64       = is64;
    p->block_size = p->radix_pow * radix;
    p->block_offset =
        ucc_knomial_pattern_loop_rank(p, rank) / p->block_size * p->block_size;
}

static inline void ucc_kn_agx_pattern_init(ucc_rank_t size, ucc_rank_t rank,
                                           ucc_kn_radix_t radix, size_t count,
                                           ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_backward(size, rank, radix, p);
    p->type  = KN_PATTERN_ALLGATHERX;
    p->count = count;
    if (p->node_type != KN_NODE_EXTRA) {
        p->block_size   = ucc_kn_compute_step_radix(p->rank, p->size, p);
        p->knx_rank     = ucc_knx_rank(rank, size, radix);
        p->block_offset = p->knx_rank / p->block_size * p->block_size;
    }
}

static inline void ucc_kn_ag_pattern_peer_seg(ucc_rank_t             peer,
                                              ucc_knomial_pattern_t *p,
                                              size_t *peer_seg_count,
                                              size_t *peer_seg_offset)
{
    ucc_kn_seg_desc_t s;

    ucc_kn_seg_desc_compute(p, &s, peer);

    if (p->type == KN_PATTERN_ALLGATHER) {
        /* sum counts between 2 ranks */
        *peer_seg_offset =
            ucc_buffer_block_offset(p->count, p->size, s.seg_start);
        *peer_seg_count =
            ucc_buffer_block_offset(p->count, p->size, s.seg_end) -
            *peer_seg_offset;
    } else if (p->type == KN_PATTERN_ALLGATHERX) {
        /* sum counts between 2 ranks */
        *peer_seg_offset = ucc_buffer_block_offset(
            p->count, p->size - p->n_extra, s.seg_start_rank_loop);
        *peer_seg_count =
            ucc_buffer_block_offset(p->count, p->size - p->n_extra,
                                    s.seg_end_rank_loop) -
            *peer_seg_offset;
    } else {
        *peer_seg_offset = vector_block_offset(p, s.seg_start);
        /* sum counts between 2 ranks */
        *peer_seg_count = vector_block_offset(p, s.seg_end) - *peer_seg_offset;
    }
}

static inline void ucc_kn_ag_pattern_next_iter(ucc_knomial_pattern_t *p)
{
    ucc_rank_t rank;
    if (p->type == KN_PATTERN_ALLGATHERX) {
        rank = p->knx_rank;
        ucc_knomial_pattern_next_iteration_backward(p);
    } else {
        rank = ucc_knomial_pattern_loop_rank(p, p->rank);
        ucc_knomial_pattern_next_iteration(p);
    }
    if (!ucc_knomial_pattern_loop_done(p)) {
        p->block_size *= ucc_kn_compute_step_radix(p->rank, p->size, p);
        p->block_offset = rank / p->block_size * p->block_size;
    }
}

/* ========================= REDUCE-SCATTER ============================  */

static inline void ucc_kn_rs_pattern_init(ucc_rank_t size, ucc_rank_t rank,
                                          ucc_kn_radix_t radix, size_t count,
                                          ucc_knomial_pattern_t *p)
{
    ucc_knomial_pattern_init_backward(size, rank, radix, p);
    p->type              = KN_PATTERN_REDUCE_SCATTER;
    p->count             = count;
    p->block_size_counts = count;
    p->block_size        = size - p->n_extra;
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

static inline void ucc_kn_rsv_pattern_init(ucc_rank_t size, ucc_rank_t rank,
                                           ucc_kn_radix_t radix,
                                           ucc_count_t *counts, int is64,
                                           ucc_knomial_pattern_t *p)
{
    int i;

    ucc_knomial_pattern_init_backward(size, rank, radix, p);
    p->type              = KN_PATTERN_REDUCE_SCATTERV;
    p->counts            = counts;
    p->block_size_counts = 0;
    p->block_size        = size - p->n_extra;
    if (is64) {
        for (i = 0; i < size; i++) {
            p->block_size_counts += ((uint64_t *)counts)[i];
        }
    } else {
        for (i = 0; i < size; i++) {
            p->block_size_counts += ((uint32_t *)counts)[i];
        }
    }
    p->count = p->block_size_counts;
    p->is64  = is64;
}

static inline void ucc_kn_rs_pattern_peer_seg(ucc_rank_t             peer,
                                              ucc_knomial_pattern_t *p,
                                              size_t *peer_seg_count,
                                              size_t *peer_seg_offset)
{
    ucc_kn_seg_desc_t s;
    ucc_rank_t        block_offset_inv;
    /* offset of the segment in counts of datatypes from the
       start of the buffer */
    size_t peer_seg_offset_base;
    /* offset of the current block in counts of datatypes from the
       start of the buffer */
    size_t block_offset_counts;

    ucc_kn_seg_desc_compute(p, &s, peer);

    block_offset_inv = ucc_knomial_pattern_loop_rank_inv(p, p->block_offset);

    if (p->type == KN_PATTERN_REDUCE_SCATTER) {
        peer_seg_offset_base =
            ucc_buffer_block_offset(p->count, p->size, s.seg_start);
        *peer_seg_count =
            ucc_buffer_block_offset(p->count, p->size, s.seg_end) -
            peer_seg_offset_base;
        block_offset_counts =
            ucc_buffer_block_offset(p->count, p->size, block_offset_inv);
    } else if (p->type == KN_PATTERN_REDUCE_SCATTERX) {
        peer_seg_offset_base = ucc_buffer_block_offset(
            p->count, p->size - p->n_extra, s.seg_start_rank_loop);
        *peer_seg_count =
            ucc_buffer_block_offset(p->count, p->size - p->n_extra,
                                    s.seg_end_rank_loop) -
            peer_seg_offset_base;
        block_offset_counts = ucc_buffer_block_offset(
            p->count, p->size - p->n_extra, p->block_offset);
    } else {
        ucc_assert(p->type == KN_PATTERN_REDUCE_SCATTERV);
        peer_seg_offset_base = vector_block_offset(p, s.seg_start);
        /* sum counts between 2 ranks */
        *peer_seg_count =
            vector_block_offset(p, s.seg_end) - peer_seg_offset_base;

        block_offset_counts = vector_block_offset(p, block_offset_inv);
    }

    ucc_assert(peer_seg_offset_base >= block_offset_counts);
    *peer_seg_offset = peer_seg_offset_base - block_offset_counts;
}

static inline void ucc_kn_rs_pattern_next_iter(ucc_knomial_pattern_t *p)
{
    ucc_kn_seg_desc_t s;
    size_t            offset;

    ucc_kn_seg_desc_compute(p, &s, p->rank);
    ucc_kn_rs_pattern_peer_seg(p->rank, p, &p->block_size_counts, &offset);
    p->block_size = s.seg_size;
    p->block_offset += s.seg_offset;

    if (p->type == KN_PATTERN_REDUCE_SCATTERX) {
        ucc_knomial_pattern_next_iteration(p);
        return;
    }
    ucc_knomial_pattern_next_iteration_backward(p);
}

static inline void ucc_kn_rs_pattern_extra_seg(ucc_knomial_pattern_t *p,
                                               size_t *               seg_count,
                                               size_t *seg_offset)
{
    /* offset is size of my data */
    if (p->type == KN_PATTERN_REDUCE_SCATTER) {
        *seg_offset = ucc_buffer_block_count(p->count, p->size, p->rank);
        *seg_count  = ucc_buffer_block_count(
            p->count, p->size, ucc_knomial_pattern_get_extra(p, p->rank));
    } else {
        *seg_offset = p->is64 ? ((uint64_t *)p->counts)[p->rank]
                              : ((uint32_t *)p->counts)[p->rank];
        *seg_count =
            p->is64
                ? ((uint64_t *)
                       p->counts)[ucc_knomial_pattern_get_extra(p, p->rank)]
                : ((uint32_t *)
                       p->counts)[ucc_knomial_pattern_get_extra(p, p->rank)];
    }
}

static inline void ucc_kn_rsx_pattern_dst(ucc_rank_t size, ucc_rank_t rank,
                                          ucc_kn_radix_t radix, size_t count,
                                          size_t *dst_offset, size_t *dst_count)
{
    ucc_knomial_pattern_t p;

    ucc_kn_agx_pattern_init(size, rank, radix, count, &p);
    if (p.node_type == KN_NODE_EXTRA) {
        *dst_offset = *dst_count = 0;
    }

    *dst_offset =
        ucc_buffer_block_offset(count, p.size - p.n_extra, p.knx_rank);
    *dst_count = ucc_buffer_block_count(count, p.size - p.n_extra, p.knx_rank);
}

static inline size_t ucc_kn_rs_max_seg_count(ucc_knomial_pattern_t *p)
{
    size_t count, seg_count, offset;
    if (KN_NODE_EXTRA == p->node_type) {
        return 0;
    }

    if (p->type == KN_PATTERN_REDUCE_SCATTERV) {
        ucc_kn_radix_t radix = ucc_kn_compute_step_radix(p->rank, p->size, p);
        ucc_rank_t     step_size = (p->size - p->n_extra) / radix;
        ucc_rank_t     start, end, i;

        ucc_assert((p->size - p->n_extra) % radix == 0);
        count = 0;
        for (i = 0; i < radix; i++) {
            start = ucc_knomial_pattern_loop_rank_inv(p, i * step_size);
            end   = ucc_knomial_pattern_loop_rank_inv(p, (i + 1) * step_size);
            seg_count =
                vector_block_offset(p, end) - vector_block_offset(p, start);
            if (seg_count > count) {
                count = seg_count;
            }
        }
    } else {
        ucc_kn_rs_pattern_peer_seg(0, p, &count, &offset);
    }
    return count;
}

#endif
