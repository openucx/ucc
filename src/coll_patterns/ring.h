/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_RING_PATTERN_H_
#define UCC_RING_PATTERN_H_

#include "utils/ucc_coll_utils.h"
#include "utils/debug/log_def.h"
#include "utils/ucc_malloc.h"

typedef struct ucc_ring_pattern {
    ucc_rank_t          size;
    unsigned            num_rings;
    /* Optional array of maps from ring index to team rank. */
    const ucc_ep_map_t *maps;
    /* Optional cached ring rank per ring (local index). */
    ucc_rank_t         *ring_ranks;
} ucc_ring_pattern_t;

/**
 * Initializes a ucc_ring_pattern_t structure with the given size and number of rings.
 *
 * @param size       Number of ranks in the ring.
 * @param num_rings  Number of rings in the pattern.
 * @param p          Pointer to the ucc_ring_pattern_t structure to initialize.
 *
 * This function sets the size and num_rings fields, and
 * initializes maps and ring_ranks pointers to NULL. Rings
 * are not initialized.
 */
static inline void
ucc_ring_pattern_init(ucc_rank_t size, unsigned num_rings,
                      ucc_ring_pattern_t *p)
{
    p->size       = size;
    p->num_rings  = num_rings;
    p->maps       = NULL;
    p->ring_ranks = NULL;
}

/**
 * Initializes a ucc_ring_pattern_t structure with the provided
 * array of endpoint maps.
 *
 * @param maps       Pointer to array of ucc_ep_map_t, one per ring.
 * @param num_rings  Number of rings (number of maps).
 * @param p          Pointer to the ucc_ring_pattern_t structure to initialize.
 *
 * The function sets the size field based on the ep_num of the first map if
 * num_rings is greater than zero, otherwise sets size to 0. It sets the
 * maps and num_rings fields, and initializes ring_ranks to NULL.
 */
static inline void
ucc_ring_pattern_init_map(const ucc_ep_map_t *maps, unsigned num_rings,
                          ucc_ring_pattern_t *p)
{
    p->size       = num_rings ? (ucc_rank_t)maps[0].ep_num : 0;
    p->num_rings  = num_rings;
    p->maps       = maps;
    p->ring_ranks = NULL;
}

/**
 * Initializes a ucc_ring_pattern_t structure with the given topo.
 *
 * @param topo       Pointer to the ucc_topo_t structure to initialize.
 * @param memory_type Memory type to use for the rings.
 * @param num_rings  Number of rings in the pattern.
 * @param p          Pointer to the ucc_ring_pattern_t structure to initialize.
 *
 * This function initializes the rings using topology information.
 */
ucc_status_t ucc_ring_pattern_init_topo(
    ucc_topo_t *topo, ucc_memory_type_t memory_type, unsigned num_rings,
    ucc_ring_pattern_t *p);

/**
 * Sets the local ring rank(s) for all rings in the given ucc_ring_pattern_t.
 *
 * @param p    Pointer to the ucc_ring_pattern_t structure.
 * @param rank Global (logical) rank to be mapped to local ring rank(s).
 *
 * This function allocates and fills the ring_ranks array in the pattern
 * structure. For each ring, ring_ranks[ring_id] is set to the local rank
 * corresponding to the provided global rank according to the ring's ep_map,
 * or just to rank itself if no maps are present.
 */
static inline void
ucc_ring_pattern_set_rank(ucc_ring_pattern_t *p, ucc_rank_t rank)
{
    ucc_rank_t ring_id;

    ucc_assert(p != NULL);
    ucc_assert(p->num_rings > 0);

    if (p->ring_ranks == NULL) {
        p->ring_ranks =
            (ucc_rank_t *)ucc_malloc(p->num_rings * sizeof(ucc_rank_t),
                                     "ring_ranks");
        ucc_assert(p->ring_ranks != NULL);
    }

    for (ring_id = 0; ring_id < p->num_rings; ring_id++) {
        p->ring_ranks[ring_id] = p->maps ?
            ucc_ep_map_local_rank(p->maps[ring_id], rank) : rank;
        ucc_assert(p->ring_ranks[ring_id] != UCC_RANK_INVALID);
    }
}

static inline ucc_rank_t
ucc_ring_pattern_rank(ucc_ring_pattern_t *p, ucc_rank_t ring_id)
{
    ucc_assert(p != NULL);
    ucc_assert(ring_id < p->num_rings);
    ucc_assert(p->ring_ranks != NULL);

    return p->ring_ranks[ring_id];
}

static inline ucc_rank_t
ucc_ring_pattern_size(ucc_ring_pattern_t *p, ucc_rank_t ring_id)
{
    ucc_assert(ring_id < p->num_rings);

    return p->size;
}

static inline ucc_rank_t
ucc_ring_pattern_eval(ucc_ring_pattern_t *p, ucc_rank_t ring_id,
                      ucc_rank_t rank)
{
    ucc_assert(ring_id < p->num_rings);
    ucc_assert(!p->maps || p->maps[ring_id].ep_num == p->size);

    return p->maps ? ucc_ep_map_eval(p->maps[ring_id], rank) : rank;
}

static inline ucc_rank_t
ucc_ring_pattern_get_send_peer(ucc_ring_pattern_t *p, ucc_rank_t ring_id,
                               ucc_rank_t rank)
{
    ucc_rank_t size = ucc_ring_pattern_size(p, ring_id);

    return ucc_ring_pattern_eval(p, ring_id, (rank + 1) % size);
}

static inline ucc_rank_t
ucc_ring_pattern_get_recv_peer(ucc_ring_pattern_t *p, ucc_rank_t ring_id,
                               ucc_rank_t rank)
{
    ucc_rank_t size = ucc_ring_pattern_size(p, ring_id);

    return ucc_ring_pattern_eval(p, ring_id, (rank - 1 + size) % size);
}

static inline ucc_rank_t
ucc_ring_pattern_get_send_block(ucc_ring_pattern_t *p, ucc_rank_t ring_id,
                                ucc_rank_t rank, ucc_rank_t step)
{
    ucc_rank_t size = ucc_ring_pattern_size(p, ring_id);

    return ucc_ring_pattern_eval(p, ring_id, (rank - step + size) % size);
}

static inline ucc_rank_t
ucc_ring_pattern_get_recv_block(ucc_ring_pattern_t *p, ucc_rank_t ring_id,
                                ucc_rank_t rank, ucc_rank_t step)
{
    ucc_rank_t size = ucc_ring_pattern_size(p, ring_id);

    return ucc_ring_pattern_eval(p, ring_id, (rank - step - 1 + size) % size);
}

static inline void ucc_ring_pattern_print(ucc_ring_pattern_t *p)
{
    ucc_rank_t ring_id, i, rank;
    char       line[256];
    size_t     left;
    int        n;

    ucc_assert(p != NULL);

    for (ring_id = 0; ring_id < p->num_rings; ring_id++) {
        n = snprintf(line, sizeof(line), "ring %u: ", ring_id);
        if (n < 0) {
            continue;
        }
        if ((size_t)n >= sizeof(line)) {
            line[sizeof(line) - 1] = '\0';
            ucc_debug("%s", line);
            continue;
        }
        left = sizeof(line) - (size_t)n;
        for (i = 0; i < p->size; i++) {
            rank = ucc_ring_pattern_eval(p, ring_id, i);
            n = snprintf(line + (sizeof(line) - left), left, "%u%s",
                         rank, (i + 1 == p->size) ? "" : " -> ");
            if (n < 0) {
                break;
            }
            if ((size_t)n >= left) {
                line[sizeof(line) - left] = '\0';
                ucc_debug("%s", line);
                n = snprintf(line, sizeof(line), "          %u%s", rank,
                             (i + 1 == p->size) ? "" : " -> ");
                if (n < 0) {
                    break;
                }
                if ((size_t)n >= sizeof(line)) {
                    line[sizeof(line) - 1] = '\0';
                    ucc_debug("%s", line);
                    left = sizeof(line);
                    continue;
                }
                left = sizeof(line) - (size_t)n;
                continue;
            }
            left -= (size_t)n;
        }
        if (left != sizeof(line)) {
            ucc_debug("%s", line);
        }
    }
}

static inline void ucc_ring_pattern_destroy(ucc_ring_pattern_t *p)
{
    if (!p) {
        return;
    }

    if (p->maps) {
        ucc_free((void *)p->maps);
        p->maps = NULL;
    }

    if (p->ring_ranks) {
        ucc_free(p->ring_ranks);
        p->ring_ranks = NULL;
    }
}

#endif
