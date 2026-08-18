/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEAM_CACHE_H_
#define UCC_TEAM_CACHE_H_

#include "config.h"
#include "ucc/api/ucc.h"
#include "ucc/api/ucc_status.h"
#include "utils/ucc_datastruct.h"
#include "utils/ucc_list.h"
#include "utils/ucc_spinlock.h"
#include <stdint.h>

/* Forward declaration to avoid pulling ucc_team.h and its heavy include chain */
typedef struct ucc_team    ucc_team_t;
typedef struct ucc_context ucc_context_t;

/* Eviction policy (UCC_TEAM_CACHE_EVICTION). Enum order MUST match
   ucc_team_cache_eviction_names[]. LFU/LRU evict the dormant team with the
   smallest team->seq_num (fewest collectives served); UCC has no wall-clock
   recency, so LRU is an accepted alias for LFU. */
typedef enum ucc_team_cache_eviction_policy {
    UCC_TEAM_CACHE_EVICTION_NONE =
        0, /* never evict; skip admission at capacity */
    UCC_TEAM_CACHE_EVICTION_FIFO = 1, /* evict oldest dormant (default) */
    UCC_TEAM_CACHE_EVICTION_LFU =
        2, /* evict least-used dormant (min seq_num) */
    UCC_TEAM_CACHE_EVICTION_LRU = 3, /* accepted alias for LFU */
} ucc_team_cache_eviction_policy_t;

/* Non-zero iff @p is a usage-based (min seq_num) policy: LFU or its LRU alias. */
#define UCC_TEAM_CACHE_EVICTION_IS_USAGE_BASED(_p)                             \
    ((_p) == UCC_TEAM_CACHE_EVICTION_LFU || (_p) == UCC_TEAM_CACHE_EVICTION_LRU)

/* String choices for UCC_TEAM_CACHE_EVICTION, indexed by the enum; NULL-terminated. */
extern const char *ucc_team_cache_eviction_names[];

/* Normalized team-cache identity built from a team's materialized membership.
   @hash is membership-only so all same-membership teams share one bucket;
   @ext_id and @instance_cookie are compared but not hashed. */
typedef struct ucc_team_cache_identity {
    uint64_t   hash;    /* membership-only bucket key; not a reuse guarantee */
    ucc_rank_t size;    /* number of endpoints */
    ucc_rank_t self_ep; /* caller's own endpoint (params->ep) */
    uint16_t
        ext_id; /* external id, or 0 for pool-id teams; compared not hashed */
    /* Collectively-assigned, never-recycled stamp identifying the physical team
       instance (MPI cids recycle; team->id alone cannot distinguish). Proposed by
       team-rank 0 and distributed via the agreement vote; used only by the RESEAT /
       dormant-derived path. 0 = unstamped. Compared by the agreement vote (not by
       identity_equal), never hashed. */
    uint64_t instance_cookie;
    ucc_rank_t
        *members; /* heap-owned, length == size; exact-compare material */
} ucc_team_cache_identity_t;

/* Build a normalized identity from team-create params (materializes membership
   and computes the hash). On success @identity owns @members; free with
   ucc_team_cache_identity_free(). Returns UCC_ERR_INVALID_PARAM if no usable map
   is present, UCC_ERR_NO_MEMORY on allocation failure. */
ucc_status_t ucc_team_cache_identity_build(
    const ucc_team_params_t *params, ucc_team_cache_identity_t *identity);

/* Exact compare (hash, size, self_ep, full members array): non-zero if equal. */
int ucc_team_cache_identity_equal(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b);

/* Membership-only compare (ignores ext_id): non-zero if @a and @b describe the
   same member set. Used by lookup_live for coexistence/derived detection. */
int ucc_team_cache_identity_equal_membership(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b);

/* Release the owned @members array and zero the identity. Safe on a zeroed
   identity and idempotent. */
void ucc_team_cache_identity_free(ucc_team_cache_identity_t *identity);

/* Non-zero iff a team is cacheable: every optional behavioral field (ORDERING,
   OUTSTANDING_COLLS, SYNC_TYPE, P2P_CONN, MEM_PARAMS) is unset in params->mask. */
int  ucc_team_cache_is_cacheable(const ucc_team_params_t *params);

/* Cache action classified locally in create_post and reconciled across a team's
   members by the agreement vote. RESEAT_DERIVED re-adopts a DORMANT derived team
   by membership only (its cid drifted); it is gated by UCC_TEAM_CACHE_RESEAT
   (default off, experimental) and relies on the vote reconciling the per-instance
   cookie so all ranks selected the same physical instance. */
typedef enum ucc_team_cache_action {
    UCC_TEAM_CACHE_ACTION_MISS              = 0, /* fresh full build */
    UCC_TEAM_CACHE_ACTION_EXACT_REUSE       = 1, /* re-adopt a DORMANT team */
    UCC_TEAM_CACHE_ACTION_DERIVED_FROM_LIVE = 2, /* derive from a LIVE parent */
    UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED = 3, /* re-adopt+reseat DORMANT derived
                                                 (membership-only, cid drift) */
} ucc_team_cache_action_t;

/* Agreement vote: one UCC_OP_BAND allreduce over the team's members reconciles
   the locally-classified action so every member reaches the same reuse-vs-fresh
   decision. Lanes [1..8] are equality pairs (value, ~value) for action/key/cookie/
   parent-cookie; lane [0] is "prepared"; lane [9] distributes rank-0's proposed
   new cookie. A non-preparing rank contributes a BAND no-op (all-ones). */
#define UCC_TEAM_CACHE_VOTE_LANES 10

/* Fill a vote buffer for one preparing rank before the BAND allreduce. @cookie is
   the candidate instance cookie (0 for EXACT_REUSE/MISS); @parent_cookie is the
   parent's cookie for DERIVED/RESEAT (0 otherwise). @proposed_cookie is written to
   the distribution lane by team-rank 0 only (others pass 0 with is_rank0=0). */
void ucc_team_cache_vote_fill(
    uint64_t *v, int prepared, ucc_team_cache_action_t action, uint64_t key,
    uint64_t cookie, uint64_t parent_cookie, int is_rank0,
    uint64_t proposed_cookie);

/* Evaluate a BAND-reduced vote buffer; returns the agreed action, or MISS if the
   members did not unanimously agree on action, key, and instance cookies. */
ucc_team_cache_action_t ucc_team_cache_vote_result(const uint64_t *v);

/* Read the distributed new-instance cookie (rank-0's proposed value) from a
   BAND-reduced vote buffer. Valid on every member after the allreduce. */
uint64_t                ucc_team_cache_vote_new_cookie(const uint64_t *v);

/* Per-cache statistics counters. */
typedef struct ucc_team_cache_stats {
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t inserts;
} ucc_team_cache_stats_t;

/* Opaque communicator-cache container. All fields are protected by @lock. A team
   on pending_destroy is on none of table/live/dormant, so lookup cannot re-adopt
   it. Local eviction determinism is not required for correctness: the agreement
   vote reconciles any per-rank hit/miss split into a consistent decision. */
typedef struct ucc_team_cache {
    void           *table; /* khash key64 -> ucc_team_t* */
    ucc_list_link_t live;  /* LIVE teams by identity */
    ucc_list_link_t
        dormant; /* DORMANT teams; head = oldest (victim), tail = MRU */
    ucc_list_link_t reserved; /* RESERVED teams pinned for an in-flight vote */
    ucc_list_link_t
                   pending_destroy; /* evicted teams awaiting UCC_OK teardown */
    ucc_spinlock_t lock;
    uint32_t       max_size;
    uint32_t       size;
    ucc_team_cache_eviction_policy_t eviction;
    uint32_t disable_linear_check; /* UCC_TEAM_CACHE_DISABLE_LINEAR_CHECK */
    uint32_t dump_stats;           /* UCC_TEAM_CACHE_DUMP_STATS */
    uint32_t derived;              /* UCC_TEAM_CACHE_DERIVED (default on) */
    uint32_t reseat;               /* UCC_TEAM_CACHE_RESEAT (default off) */
    uint32_t agreement;            /* UCC_TEAM_CACHE_AGREEMENT (default on) */
    /* Monotonic generation for per-instance cookies; bumped under @lock when
       team-rank 0 stamps a fresh instance, seeded non-zero from the context
       seq_num so cookies are distinct across contexts and never recycled. */
    uint64_t cache_gen;
    ucc_team_cache_stats_t stats;
} ucc_team_cache_t;

/* Reserve and return the next per-instance cookie from @cache->cache_gen. Called
   only by team-rank 0 under @cache->lock; never returns 0 and never repeats. */
uint64_t     ucc_team_cache_next_cookie(ucc_team_cache_t *cache);

/* Allocate and initialise a new team cache. @disable_linear_check trusts the
   64-bit hash alone in lookup (skips the exact rank-array compare). Returns
   UCC_ERR_NO_MEMORY on allocation failure. */
ucc_status_t ucc_team_cache_init(
    ucc_team_cache_t **cache, uint32_t max_size,
    ucc_team_cache_eviction_policy_t eviction, uint32_t disable_linear_check);

/* Look up a DORMANT cached team by exact identity, or NULL on a miss. A LIVE team
   is never returned (inactive-only reuse). Does not touch refcount. The caller
   MUST hold @cache->lock across this and the following ucc_team_cache_get() so the
   lookup->adopt window is atomic. */
ucc_team_t *ucc_team_cache_lookup(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id);

/* Look up a LIVE cached team by identity (derived-create path): the parent whose
   shared artifacts a derived team borrows. Does not touch refcount/state or stats.
   While holding @cache->lock the caller MUST pin the parent's artifacts via
   ucc_team_artifacts_get() and thereafter use only that holder plus parent->id. */
ucc_team_t *ucc_team_cache_lookup_live(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id);

/* Look up a DORMANT DERIVED team by membership only, ignoring ext_id (RESEAT): a
   dormant derived team keyed by a drifted cid is missed by the exact lookup. The
   caller re-seats the returned team's id/tag domain to the new cid. Restricted to
   derived dormant teams. Does not touch refcount/state or stats. Under @cache->lock. */
ucc_team_t *ucc_team_cache_lookup_dormant_derived(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id);

/* Insert a team as DORMANT under its cache_identity (the caller immediately
   re-adopts it to LIVE via ucc_team_cache_get()). A full cache skips the insert
   and returns UCC_OK with cache_state left NONE. A hash-key collision chains the
   team onto the bucket via team->bucket_link (tail-append = insertion order); an
   exact-identity duplicate is refused. Under @cache->lock. Returns UCC_ERR_NO_MEMORY
   if the khash table cannot grow. */
ucc_status_t ucc_team_cache_insert(ucc_team_cache_t *cache, ucc_team_t *team);

/* Adopt a DORMANT cached team: refcount++ and cache_state = LIVE. Forms the atomic
   lookup->adopt step with ucc_team_cache_lookup(). Under @cache->lock. */
void         ucc_team_cache_get(ucc_team_t *team);

/* Release a LIVE cached team: refcount--. At zero, cache_state = DORMANT so a later
   lookup can re-adopt it. Returns the new refcount. Under @cache->lock. */
int          ucc_team_cache_put(ucc_team_t *team);

/* Drain and free a team cache. Warns (does not assert) on residual entries; the
   caller guarantees a full drain first. May be NULL (no-op). */
void         ucc_team_cache_destroy(ucc_team_cache_t *cache);

/* Pick the eviction victim from the dormant list per cache->eviction: FIFO returns
   the list head (oldest); LFU/LRU returns the min-seq_num team (tie-break on the
   older list position). NULL if the list is empty. The victim may differ per rank
   under the usage-based policy, which the agreement vote reconciles. Under cache->lock. */
ucc_team_t  *ucc_team_cache_pick_lru_victim(ucc_team_cache_t *cache);

/* Per-context team registry helpers: the single source of truth for moves between
   the live and dormant lists. Callers hold cache->lock; a cached team is on exactly
   one of live/dormant, and on neither once removed. */

/* Add @team to the LIVE registry (ACTIVE-insert and re-adopt paths). The team must
   not already be on any list. Under cache->lock. */
void         ucc_team_cache_registry_add_live(
            ucc_team_cache_t *cache, ucc_team_t *team);

/* Move @team from LIVE to DORMANT (intercepted destroy path). Under cache->lock. */
void ucc_team_cache_registry_make_dormant(
    ucc_team_cache_t *cache, ucc_team_t *team);

/* Move @team from DORMANT to LIVE (re-adopt path). Under cache->lock. */
void ucc_team_cache_registry_make_live(
    ucc_team_cache_t *cache, ucc_team_t *team);

/* Move @team from DORMANT to RESERVED, pinning it for an in-flight agreement vote.
   It stays in the khash bucket with refcount unchanged so a vote-FAIL can roll it
   back to DORMANT without a refcount race. Under cache->lock. */
void ucc_team_cache_registry_make_reserved(
    ucc_team_cache_t *cache, ucc_team_t *team);

/* Remove @team from whichever registry list it is on (eviction / teardown drain).
   After this the team is on neither list. Under cache->lock. */
void ucc_team_cache_registry_remove(ucc_team_cache_t *cache, ucc_team_t *team);

/* Erase @team from the khash bucket table and decrement cache->size (the registry_*
   helpers only touch the list links). No-op if not present. Under cache->lock. */
void ucc_team_cache_table_erase(ucc_team_cache_t *cache, ucc_team_t *team);

/*
 * The pending-destroy state machine (ucc_team_cache_evict_one and
 * ucc_team_cache_progress_pending) is declared in ucc_team.h and defined in
 * ucc_team.c, next to ucc_team_cache_drain, so it can reuse the file-local
 * ucc_team_destroy_single teardown.
 */

/* Dump team-cache hit/miss/eviction counters to a single ucc_info line. Intended
   for context destroy when UCC_TEAM_CACHE_DUMP_STATS is enabled. */
void ucc_team_cache_dump_stats(ucc_team_cache_t *cache);

#endif /* UCC_TEAM_CACHE_H_ */
