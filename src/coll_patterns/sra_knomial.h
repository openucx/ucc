/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef SRA_KNOMIAL_H_
#define SRA_KNOMIAL_H_

#include "recursive_knomial.h"

static inline ucc_rank_t ucc_sra_kn_compute_step_radix(ucc_rank_t rank,
                                                       ucc_rank_t size,
                                                       ucc_knomial_pattern_t *p)
{
    ucc_rank_t step_radix = 0;
    ucc_rank_t k, peer;

    for (k = 1; k < p->radix; k++) {
        peer = ucc_knomial_pattern_get_loop_peer(p, rank, size, k);
        if (peer == UCC_KN_PEER_NULL)
            continue;
        step_radix += 1;
    }
    step_radix += 1;
    return step_radix;
}

// segment index in exchange group of tree
static inline ucc_rank_t ucc_sra_kn_compute_seg_index(ucc_rank_t peer,
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

// segment size
static inline size_t ucc_sra_kn_compute_seg_size(size_t         block_count,
                                                 ucc_kn_radix_t radix,
                                                 ucc_rank_t     si)
{
    return block_count / radix + (si < (block_count % radix) ? 1 : 0);
}

// segment offset in exhange group of tree
static inline size_t ucc_sra_kn_compute_seg_offset(size_t         block_count,
                                                   ucc_kn_radix_t radix,
                                                   ucc_rank_t     si)
{
    return (block_count / radix) * si +
           ((si < (block_count % radix)) ? si : (block_count % radix));
}

static inline size_t ucc_sra_kn_compute_block_count(size_t     count,
                                                    ucc_rank_t rank,
                                                    ucc_knomial_pattern_t *p)
{
    size_t     block_count = count;
    ucc_rank_t k_pow       = 1;
    ucc_rank_t i, my_si, my_seg_len;

    for (i = 0; i < p->iteration; i++) {
        my_si       = ucc_sra_kn_compute_seg_index(rank, k_pow, p);
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
    size_t                block_count = count;
    ucc_rank_t            step_radix  = 0;
    size_t                my_seg_len  = 0;
    ucc_rank_t            k, r, peer, my_si;
    size_t                my_seg_offset;
    ucc_knomial_pattern_t p;
    ucc_knomial_pattern_init(size, rank, radix, &p);

    if (KN_NODE_EXTRA == p.node_type) {
        if (offset)
            *offset = 0;
        if (seglen)
            *seglen = count;
        return;
    }
    while (!ucc_knomial_pattern_loop_done(&p)) {
        r = 0;
        for (k = 1; k < p.radix; k++) {
            peer = ucc_knomial_pattern_get_loop_peer(&p, rank, size, k);
            if (peer == UCC_KN_PEER_NULL)
                continue;
            r++;
        }
        step_radix = r + 1;
        my_si      = ucc_sra_kn_compute_seg_index(rank, p.radix_pow, &p);
        my_seg_offset =
            ucc_sra_kn_compute_seg_offset(block_count, step_radix, my_si);
        _offset += my_seg_offset * dt_size;
        if (p.iteration < p.pow_radix_sup - 1) {
            block_count =
                ucc_sra_kn_compute_seg_size(block_count, step_radix, my_si);
        }
        ucc_knomial_pattern_next_iteration(&p);
    }
    if (step_radix) {
        my_seg_len =
            ucc_sra_kn_compute_seg_size(block_count, step_radix, my_si);
    }
    if (offset)
        *offset = _offset;
    if (seglen)
        *seglen = my_seg_len;
}

static inline ptrdiff_t ucc_sra_kn_get_offset(size_t count, size_t dt_size,
                                              ucc_rank_t rank, ucc_rank_t size,
                                              ucc_kn_radix_t radix)
{
    ptrdiff_t offset;
    ucc_sra_kn_get_offset_and_seglen(count, dt_size, rank, size, radix, &offset,
                                     NULL);
    return offset;
}

#endif
