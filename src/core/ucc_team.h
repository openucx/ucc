/**
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TEAM_H_
#define UCC_TEAM_H_

#include "ucc/api/ucc.h"
#include "utils/ucc_datastruct.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_list.h"
#include "utils/ucc_spinlock.h"
#include "ucc_context.h"
#include "ucc_team_cache.h"
#include "utils/ucc_math.h"
#include "components/base/ucc_base_iface.h"
#include "components/cl/ucc_cl.h"
#include "components/tl/ucc_tl.h"
#include "coll_score/ucc_coll_score.h"
#include "ucc_service_coll.h" /* full ucc_service_coll_req_t for the embedded vote req */

typedef struct ucc_service_coll_req ucc_service_coll_req_t;
typedef enum {
    UCC_TEAM_ADDR_EXCHANGE, /* value 0: freshly calloc'd default */
    UCC_TEAM_SERVICE_TEAM,
    UCC_TEAM_ALLOC_ID,
    UCC_TEAM_CL_CREATE,
    UCC_TEAM_ACTIVE,
    /* Agreement states for a cacheable size>1 team; appended so ADDR_EXCHANGE
       stays the calloc-zero default. */
    UCC_TEAM_CACHE_AGREE,         /* vote in flight; commit on completion */
    UCC_TEAM_CACHE_MISS_TEARDOWN, /* vote lost: drain the rejected EXACT
                                     candidate to terminal UCC_OK, then rebuild */
} ucc_team_state_t;

/*
 * Cache-state marker for ucc_team_t. All transitions are taken under cache->lock.
 *
 *   NONE --insert--> DORMANT --reserve--> RESERVED --agree-PASS--> LIVE
 *                       ^                     |
 *                       +----agree-FAIL-------+
 *   LIVE --put-to-0--> DORMANT
 *
 * RESERVED is a re-adopt candidate pinned for an in-flight agreement vote: off
 * the live/dormant lists but still in the bucket, refcount unchanged, so a
 * vote-FAIL rolls back to DORMANT with no race. team->refcount tracks the live
 * communicators backing a cached team.
 */
typedef enum ucc_team_cache_state {
    UCC_TEAM_CACHE_STATE_NONE    = 0, /* never cached (safe default) */
    UCC_TEAM_CACHE_STATE_DORMANT = 1, /* cached, no live backing comm */
    UCC_TEAM_CACHE_STATE_RESERVED =
        2,                         /* pinned for an in-flight agreement vote */
    UCC_TEAM_CACHE_STATE_LIVE = 3, /* cached, backing one live comm */
} ucc_team_cache_state_t;

/*
 * Refcounted, read-only shared team artifacts (ctx_map, ctx_ranks, topo) that a
 * cached team and its derived team(s) share through one heap instance instead of
 * recomputing. The topo is materialized build-once (ucc_topo_prepare_shared)
 * before sharing to avoid lazy write races under THREAD_MULTIPLE. TL nested maps
 * store a raw &holder->ctx_map, so the holder must never be copied by value.
 * Created at refcount 1; get() adds a ref, put() frees the holder at 0. The
 * spinlock guards the refcount (reachable from >1 live team).
 */
typedef struct ucc_team_artifacts {
    ucc_ep_map_t   ctx_map;   /*< map to the ctx ranks, defined if CTX type is
                                  global (oob provided) */
    ucc_rank_t    *ctx_ranks; /*< UCC-owned backing array of ctx_map, or NULL */
    ucc_topo_t    *topo;     /*< subset topology, materialized before sharing */
    int            refcount; /*< number of live teams pointing at this holder */
    ucc_spinlock_t lock;     /*< guards refcount (THREAD_MULTIPLE) */
    uint8_t        heap;     /*< 1: heap-allocated (shareable, freed at refcount
                                 0); 0: embedded in a ucc_team_t (never shared,
                                 contents cleaned but struct not freed) */
} ucc_team_artifacts_t;

/* Allocate a heap holder at refcount 1 with zero-initialized artifacts (the team
   fills them during create_test). Heap holders are shareable by derived teams.
   Returns NULL on OOM. */
ucc_team_artifacts_t *ucc_team_artifacts_alloc(void);

/* Initialize an embedded (non-heap, non-shareable) holder in place, at refcount
   1. Used for teams that will never be cached or shared, avoiding a heap
   allocation. Never freed by ucc_team_artifacts_put (lives inside ucc_team_t). */
void                  ucc_team_artifacts_init_inline(ucc_team_artifacts_t *a);

/* Add a reference; returns the same pointer for call-site convenience. */
ucc_team_artifacts_t *ucc_team_artifacts_get(ucc_team_artifacts_t *artifacts);

/* Drop one reference; frees the holder and all it owns at refcount 0. No-op on
   NULL. */
void                  ucc_team_artifacts_put(ucc_team_artifacts_t *artifacts);

/* Read the refcount under the artifacts lock (the same lock get/put use). Use
   this instead of touching ->refcount directly from outside the lock. */
int ucc_team_artifacts_refcount(ucc_team_artifacts_t *artifacts);

/* Accessors for the shared artifacts (make the sharing indirection explicit). */
#define UCC_TEAM_CTX_MAP(_team)   ((_team)->artifacts->ctx_map)
#define UCC_TEAM_CTX_RANKS(_team) ((_team)->artifacts->ctx_ranks)
#define UCC_TEAM_TOPO(_team)      ((_team)->artifacts->topo)

typedef struct ucc_team {
    ucc_team_state_t        state;
    ucc_context_t **        contexts;
    uint32_t                num_contexts;
    ucc_base_team_params_t  bp;
    ucc_team_oob_coll_t     runtime_oob;
    ucc_cl_team_t **        cl_teams;
    int                     n_cl_teams;
    int                     last_team_create_posted;
    uint16_t                id; /*< context-uniq team identifier */
    ucc_rank_t              rank;
    ucc_rank_t              size;
    ucc_tl_team_t *         service_team;
    ucc_service_coll_req_t *sreq;
    ucc_addr_storage_t      addr_storage; /*< addresses of team endpoints */
    void *                  oob_req;
    /* Shared holder for ctx_map/ctx_ranks/topo; access via UCC_TEAM_CTX_MAP /
       UCC_TEAM_CTX_RANKS / UCC_TEAM_TOPO. Points at a heap holder for
       cacheable/shareable teams, else at @artifacts_inline. */
    ucc_team_artifacts_t   *artifacts;
    /* Embedded holder for non-cacheable teams (no extra heap allocation). */
    ucc_team_artifacts_t    artifacts_inline;
    ucc_score_map_t *score_map; /*< score map of CLs (per-team, NOT shared) */
    uint32_t                seq_num;
    int                       refcount;
    ucc_team_cache_identity_t cache_identity;
    ucc_list_link_t           cache_link; /* LRU/registry list */
    /* Intrusive ring chaining same-membership-hash teams off the khash bucket
       head; orthogonal to cache_link (bucket = hash collision chain). */
    ucc_list_link_t           bucket_link;
    ucc_team_cache_state_t
             cache_state; /* NONE/DORMANT/RESERVED/LIVE - see enum */
    /* Set after identity build on a cache miss, cleared at ACTIVE; distinguishes
       "cacheable but not yet inserted" from cache_state NONE. */
    int      cache_pending_insert;
    /* Derived-team marker: a second team for identical membership that borrows the
       parent's shared artifacts while getting its own id/CL/TL/score_map. Set only
       by ucc_team_init_derived. parent_id records the borrowed-from id. */
    int      is_derived;
    uint16_t parent_id;
    /* Cross-rank cache-action agreement: populated in create_post for a cacheable
       size>1 team, consumed by the CACHE_AGREE create_test state. */
    ucc_team_cache_action_t cache_local_action; /* this rank's classification */
    ucc_service_coll_req_t  cache_vote_req; /* embedded (non-heap) vote req */
    uint64_t                cache_vote_in[UCC_TEAM_CACHE_VOTE_LANES];
    uint64_t                cache_vote_out[UCC_TEAM_CACHE_VOTE_LANES];
    /* DERIVED_FROM_LIVE pin held across the vote; consumed by ucc_team_init_derived
       on a PASS, released on a global-MISS. */
    ucc_team_artifacts_t   *cache_derive_artifacts;
    uint16_t                cache_derive_parent_id;
    /* Experimental RESEAT (UCC_TEAM_CACHE_RESEAT): the drifted ext_id (new cid) to
       re-seat the reused DORMANT derived candidate to, and the parent's per-instance
       cookie used by the reseat vote. Meaningful only for RESEAT_DERIVED / derived
       teams respectively. */
    uint16_t                cache_reseat_new_id;
    uint64_t                cache_parent_instance_cookie;
} ucc_team_t;

/* If the bit is set then team_id is provided by the user */
#define UCC_TEAM_ID_EXTERNAL_BIT ((uint16_t)UCC_BIT(15))
#define UCC_TEAM_ID_IS_EXTERNAL(_team) (team->id & UCC_TEAM_ID_EXTERNAL_BIT)
#define UCC_TEAM_ID_MAX ((uint16_t)UCC_BIT(15) - 1)

void ucc_copy_team_params(ucc_team_params_t *dst, const ucc_team_params_t *src);

/* Team-id pool bit helpers (internal; exposed for unit testing the id<->bit
   boundary math). The pool is 1-indexed: bit (pos-1) of word i encodes id
   i*64+pos (pos in 1..64). ucc_team_id_pool_ffs_clear returns the 1-based index
   of the lowest set bit and clears it (0 if none); ucc_team_id_pool_set_bit
   restores the bit for @id to word (id-1)/64. */
int  ucc_team_id_pool_ffs_clear(uint64_t *value);
void ucc_team_id_pool_set_bit(uint64_t *local, int id);

/*
 * Derived-team create mechanism. A derived team reuses a parent (cached, LIVE)
 * team's shared artifacts (ctx_map + topo) while allocating its own team-id,
 * CL/TL objects, and score_map, so two coexisting live comms (e.g. MPI_Comm_dup)
 * are cheaper than a full create and each keeps an independent tag/seq domain.
 *
 * ucc_team_can_derive_from(parent): non-zero iff @parent is a shareable source
 * (ACTIVE, materialized ctx_map/topo); otherwise the team takes a full create.
 * ucc_team_init_derived(team, held_artifacts, parent_id): CONSUMES the caller's
 * already-bumped reference on the parent's holder. Must be called BEFORE
 * ucc_team_create_post_single; always returns UCC_OK.
 */
int  ucc_team_can_derive_from(const ucc_team_t *parent);
ucc_status_t ucc_team_init_derived(
    ucc_team_t *team, ucc_team_artifacts_t *held_artifacts, uint16_t parent_id);

/* Reap all DORMANT cached teams held by @context. Must be called at the START of
   ucc_context_destroy, BEFORE CL/TL context destroy and the shared service-team
   teardown, because a dormant team still holds CL/TL/service refs and a team-id.
   Drives each team's (possibly async) teardown to completion with
   ucc_context_progress. No-op if the context has no team cache; on return the
   dormant registry is empty. */
void         ucc_team_cache_drain(ucc_context_t *context);

/* Pending-destroy state machine. Defined in ucc_team.c (not ucc_team_cache.c) so
   they can reuse the file-local ucc_team_destroy_single teardown. */
ucc_status_t ucc_team_cache_progress_pending(ucc_team_cache_t *cache);
ucc_status_t ucc_team_cache_evict_one(ucc_team_cache_t *cache);

/* Returns addressing information for "rank" in a team.
   If ucc context was created with OOB then addr storage is located on context.
   In that case we need to map rank to ctx_rank first. Otherwise, addr
   storage is per-team: just use rank then.

   The returned value is "header": it stores proc_info, host info, ctx_id and
   addresses of TL/CL components.*/
static inline ucc_context_addr_header_t *
ucc_get_team_ep_header(ucc_context_t *context, ucc_team_t *team,
                       ucc_rank_t rank)
{
    ucc_addr_storage_t *storage      = context->addr_storage.storage
                                           ? &context->addr_storage
                                           : &team->addr_storage;
    ucc_rank_t          storage_rank = context->addr_storage.storage
                                           ? (team ? ucc_ep_map_eval(
                                                UCC_TEAM_CTX_MAP(team), rank)
                                                   : rank)
                                           : rank;

    return UCC_ADDR_STORAGE_RANK_HEADER(storage, storage_rank);
}

/* Gets the component specific address of rank in a team.
   First we get the header, and then find the component address
   by offset.
   Returns NULL if the requested component is not present in the peer's
   packed address. */
static inline void *ucc_get_team_ep_addr(ucc_context_t *context,
                                         ucc_team_t *team, ucc_rank_t rank,
                                         unsigned long component_id)
{
    ucc_context_addr_header_t *h    = ucc_get_team_ep_header(context, team,
                                                             rank);
    void                      *addr = NULL;
    int                        i;
    for (i = 0; i < h->n_components; i++) {
        if (h->components[i].id == component_id) {
            addr = PTR_OFFSET(h, h->components[i].offset);
            break;
        }
    }
    return addr;
}

static inline ucc_rank_t ucc_get_ctx_rank(ucc_team_t *team, ucc_rank_t team_rank)
{
    return ucc_ep_map_eval(UCC_TEAM_CTX_MAP(team), team_rank);
}

static inline ucc_host_id_t ucc_team_rank_host_id(ucc_rank_t rank, ucc_team_t *team)
{
    return UCC_TEAM_TOPO(team)
        ->topo->procs[ucc_get_ctx_rank(team, rank)]
        .host_id;
}

static inline int ucc_team_ranks_on_same_node(ucc_rank_t rank1, ucc_rank_t rank2,
                                              ucc_team_t *team)
{
    ucc_context_addr_header_t *h1 = ucc_get_team_ep_header(team->contexts[0],
                                                           team, rank1);
    ucc_context_addr_header_t *h2 = ucc_get_team_ep_header(team->contexts[0],
                                                           team, rank2);

    return h1->ctx_id.pi.host_hash == h2->ctx_id.pi.host_hash;
}

static inline int ucc_team_map_is_single_node(ucc_team_t *team,
                                              ucc_ep_map_t map)
{
    uint64_t   i;
    ucc_rank_t r, r0;

    r0 = ucc_ep_map_eval(map, 0);
    for (i = 1; i < map.ep_num; i++) {
        r = ucc_ep_map_eval(map, i);
        if (!ucc_team_ranks_on_same_node(r0, r, team)) {
            return 0;
        }
    }
    return 1;
}

#endif
