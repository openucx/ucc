/**
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

typedef struct ucc_team    ucc_team_t;
typedef struct ucc_context ucc_context_t;

/* NONE -> DORMANT -> RESERVED -> LIVE, all changed under cache->lock */
typedef enum ucc_team_cache_state {
    UCC_TEAM_CACHE_STATE_NONE     = 0, /* never cached */
    UCC_TEAM_CACHE_STATE_DORMANT  = 1, /* cached, no live backing team */
    UCC_TEAM_CACHE_STATE_RESERVED = 2, /* pinned for an in-flight vote */
    UCC_TEAM_CACHE_STATE_LIVE     = 3, /* cached, backing one live team */
} ucc_team_cache_state_t;

/* Order must match ucc_team_cache_eviction_names[] */
typedef enum ucc_team_cache_eviction_policy {
    UCC_TEAM_CACHE_EVICTION_NONE = 0,
    UCC_TEAM_CACHE_EVICTION_FIFO = 1,
} ucc_team_cache_eviction_policy_t;

/* UCC_TEAM_CACHE_EVICTION choices, indexed by the enum, NULL terminated */
extern const char *ucc_team_cache_eviction_names[];

/* Normalized team identity; @hash covers membership only, not @ext_id */
typedef struct ucc_team_cache_identity {
    uint64_t    hash;
    ucc_rank_t  size;
    ucc_rank_t  self_ep;
    uint16_t    ext_id; /* 0 for pool-id teams */
    /* Per-adoption stamp that identifies WHICH cached team was picked.
       Normally the vote's key lane carries @ext_id, which is enough to prove
       every rank selected the same entry. RESEAT breaks that: its lookup
       deliberately ignores @ext_id so a drifted id can still match on
       membership, and two ranks holding different dormant teams of identical
       membership would then agree on a lane that no longer distinguishes them.
       The cookie is stamped when a team is adopted and voted on instead, so the
       agreement still proves a common choice. 0 until the vote stamps it. */
    uint64_t    instance_cookie;
    ucc_rank_t *members; /* heap owned, length == size */
} ucc_team_cache_identity_t;

/* Materialize membership from @params into @identity, which owns its members */
ucc_status_t ucc_team_cache_identity_build(
    const ucc_team_params_t *params, ucc_team_cache_identity_t *identity);

/* Compare membership and ext_id */
int ucc_team_cache_identity_equal(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b);

/* Compare membership only */
int ucc_team_cache_identity_equal_membership(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b);

/* Free @identity->members and zero @identity; idempotent */
void ucc_team_cache_identity_free(ucc_team_cache_identity_t *identity);

/* Non-zero if no optional behavioral team param is set in params->mask */
int  ucc_team_cache_is_cacheable(const ucc_team_params_t *params);

typedef enum ucc_team_cache_action {
    UCC_TEAM_CACHE_ACTION_MISS        = 0, /* fresh full build */
    UCC_TEAM_CACHE_ACTION_EXACT_REUSE = 1, /* re-adopt a DORMANT team */
} ucc_team_cache_action_t;

typedef struct ucc_team_cache_stats {
    uint64_t lookups;
    uint64_t hits;
    uint64_t misses;
    uint64_t evictions;
    uint64_t inserts;
} ucc_team_cache_stats_t;

/* All fields are protected by @lock */
typedef struct ucc_team_cache {
    void                            *table; /* khash key64 -> ucc_team_t* */
    ucc_list_link_t                  live;
    ucc_list_link_t                  dormant; /* head = oldest victim */
    ucc_list_link_t                  reserved;
    ucc_list_link_t                  pending_destroy;
    ucc_spinlock_t                   lock;
    uint32_t                         max_size;
    uint32_t                         size;
    ucc_team_cache_eviction_policy_t eviction;
    uint32_t                         disable_linear_check;
    uint32_t                         dump_stats;
    uint32_t                         agreement;
    uint64_t                         cache_gen; /* source of instance cookies */
    ucc_team_cache_stats_t           stats;
} ucc_team_cache_t;

/* Reserve the next instance cookie; team rank 0 only, under @cache->lock */
uint64_t     ucc_team_cache_next_cookie(ucc_team_cache_t *cache);

ucc_status_t ucc_team_cache_init(
    ucc_team_cache_t **cache, uint32_t max_size,
    ucc_team_cache_eviction_policy_t eviction, uint32_t disable_linear_check);

/* Free @cache, which the caller has already drained; may be NULL */
void        ucc_team_cache_destroy(ucc_team_cache_t *cache);

/* Find a DORMANT team by exact identity, or NULL; under @cache->lock */
ucc_team_t *ucc_team_cache_lookup(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id);

/* Insert @team as DORMANT; a full cache or hash collision skips the insert */
ucc_status_t ucc_team_cache_insert(ucc_team_cache_t *cache, ucc_team_t *team);

/* Adopt a DORMANT team: refcount++, DORMANT -> LIVE; under @cache->lock */
void         ucc_team_cache_get(ucc_team_t *team);

/* Release a LIVE team: refcount--, LIVE -> DORMANT at zero; returns refcount */
int          ucc_team_cache_put(ucc_team_t *team);

/* Eviction victim per @cache->eviction, or NULL if no team is dormant */
ucc_team_t  *ucc_team_cache_pick_victim(ucc_team_cache_t *cache);

/* Registry helpers below move @team between lists, all under @cache->lock */
void         ucc_team_cache_registry_add_live(
            ucc_team_cache_t *cache, ucc_team_t *team);

void ucc_team_cache_registry_make_dormant(
    ucc_team_cache_t *cache, ucc_team_t *team);

void ucc_team_cache_registry_make_live(
    ucc_team_cache_t *cache, ucc_team_t *team);

void ucc_team_cache_registry_make_reserved(
    ucc_team_cache_t *cache, ucc_team_t *team);

/* Remove @team from whichever list it is on */
void ucc_team_cache_registry_remove(ucc_team_t *team);

/* Erase @team from the bucket table and decrement @cache->size */
void ucc_team_cache_table_erase(ucc_team_cache_t *cache, ucc_team_t *team);

/* Log hit/miss/eviction counters, for UCC_TEAM_CACHE_DUMP_STATS */
void ucc_team_cache_dump_stats(ucc_team_cache_t *cache);

#endif /* UCC_TEAM_CACHE_H_ */
