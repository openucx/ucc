/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_team_cache.h"
#include "ucc_team.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_log.h"
#include "utils/ucc_coll_utils.h"
#include "utils/ucc_compiler_def.h"
#include "utils/khash.h"
#include <inttypes.h>
#include <string.h>

/* uint64_t -> ucc_team_t* map local to this TU (handle stored as void* in the
   cache). Each key holds the chain head; same-hash teams are threaded into a ring
   via team->bucket_link, orthogonal to cache_link (the live/dormant registry). */
KHASH_MAP_INIT_INT64(ucc_team_cache_map, ucc_team_t *)

/* Walk the bucket ring headed by @_head, stopping when it returns to @_head. Do
   NOT unlink @_it inside this loop (table_erase uses its own safe walk). */
#define UCC_TEAM_CACHE_BUCKET_FOR_EACH(_it, _head)                             \
    for ((_it) = (_head); (_it) != NULL;                                       \
         (_it) = ((_it)->bucket_link.next == &(_head)->bucket_link)            \
                     ? NULL                                                    \
                     : ucc_container_of(                                       \
                           (_it)->bucket_link.next, ucc_team_t, bucket_link))

/* Indexed by ucc_team_cache_eviction_policy_t; NULL-terminated. Order MUST match
   the enum in ucc_team_cache.h. */
const char *ucc_team_cache_eviction_names[] = {
    [UCC_TEAM_CACHE_EVICTION_NONE] = "none",
    [UCC_TEAM_CACHE_EVICTION_FIFO] = "fifo",
    [UCC_TEAM_CACHE_EVICTION_LFU]  = "lfu",
    [UCC_TEAM_CACHE_EVICTION_LRU]  = "lru", /* alias for lfu */
    NULL};

ucc_status_t ucc_team_cache_init(
    ucc_team_cache_t **cache, uint32_t max_size,
    ucc_team_cache_eviction_policy_t eviction, uint32_t disable_linear_check)
{
    ucc_team_cache_t *c;

    c = ucc_calloc(1, sizeof(*c), "ucc_team_cache");
    if (ucc_unlikely(!c)) {
        ucc_error("failed to allocate ucc_team_cache_t");
        return UCC_ERR_NO_MEMORY;
    }

    c->table = kh_init(ucc_team_cache_map);
    if (ucc_unlikely(!c->table)) {
        ucc_error("failed to allocate ucc_team_cache hash table");
        goto err;
    }

    ucc_list_head_init(&c->live);
    ucc_list_head_init(&c->dormant);
    ucc_list_head_init(&c->reserved);
    ucc_list_head_init(&c->pending_destroy);
    ucc_spinlock_init(&c->lock, 0);

    c->max_size             = max_size;
    c->eviction             = eviction;
    c->disable_linear_check = disable_linear_check;

    ucc_debug(
        "ucc_team_cache created: %p, max_size=%u, eviction=%s, "
        "disable_linear_check=%u",
        (void *)c,
        max_size,
        ucc_team_cache_eviction_names[eviction],
        disable_linear_check);
    *cache = c;
    return UCC_OK;

err:
    ucc_free(c);
    return UCC_ERR_NO_MEMORY;
}

void ucc_team_cache_destroy(ucc_team_cache_t *cache)
{
    if (!cache) {
        return;
    }

    if (cache->size != 0) {
        ucc_warn(
            "ucc_team_cache_destroy called with %u entries still present",
            cache->size);
    }

    if (!ucc_list_is_empty(&cache->pending_destroy)) {
        ucc_warn(
            "ucc_team_cache_destroy called with pending-destroy entries "
            "still present (eviction teardown not flushed)");
    }

    /* Context destroy runs after all teams are quiesced, so no vote is in flight. */
    ucc_assert(ucc_list_is_empty(&cache->reserved));

    void *cache_ptr = (void *)cache;

    kh_destroy(ucc_team_cache_map, (khash_t(ucc_team_cache_map) *)cache->table);
    ucc_spinlock_destroy(&cache->lock);
    ucc_free(cache);
    ucc_debug("ucc_team_cache destroyed: %p", cache_ptr);
}

void ucc_team_cache_dump_stats(ucc_team_cache_t *cache)
{
    double hit_rate = 0.0;

    if (!cache) {
        return;
    }

    if (cache->stats.lookups > 0) {
        hit_rate = (cache->stats.hits * 100.0) / cache->stats.lookups;
    }

    ucc_info(
        "team_cache stats: lookups=%" PRIu64 " hits=%" PRIu64 " (%.1f%%) "
        "misses=%" PRIu64 " inserts=%" PRIu64 " evictions=%" PRIu64,
        cache->stats.lookups,
        cache->stats.hits,
        hit_rate,
        cache->stats.misses,
        cache->stats.inserts,
        cache->stats.evictions);
}

/* FNV-1a over {size, self_ep, members} as a single stream, so prefix length and
   byte order both influence every bit. Used only as an O(1) bucket key; lookup
   still verifies with ucc_team_cache_identity_equal(). */
#define UCC_TEAM_CACHE_FNV1A_OFFSET 0xcbf29ce484222325ULL
#define UCC_TEAM_CACHE_FNV1A_PRIME  0x00000100000001b3ULL

static void ucc_team_cache_fnv1a_accumulate(
    uint64_t *h, const void *data, size_t len)
{
    const uint8_t *p = (const uint8_t *)data;
    size_t         b;

    for (b = 0; b < len; b++) {
        *h ^= (uint64_t)p[b];
        *h *= UCC_TEAM_CACHE_FNV1A_PRIME;
    }
}

ucc_status_t ucc_team_cache_identity_build(
    const ucc_team_params_t *params, ucc_team_cache_identity_t *identity)
{
    ucc_rank_t  size;
    ucc_rank_t  self_ep;
    ucc_rank_t *members;
    ucc_rank_t  i;
    uint64_t    h;

    if (!(params->mask & UCC_TEAM_PARAM_FIELD_EP_MAP)) {
        /* Without a materializable map there is no concrete membership to key on. */
        ucc_debug("team cache identity: no EP_MAP in params, not cacheable");
        return UCC_ERR_INVALID_PARAM;
    }

    size    = (ucc_rank_t)params->ep_map.ep_num;
    self_ep = (params->mask & UCC_TEAM_PARAM_FIELD_EP) ? (ucc_rank_t)params->ep
                                                       : UCC_RANK_INVALID;

    /* A caller-supplied external id defines a distinct id/tag domain, so it is
       part of the identity; a recycled id must not reuse a stale cached team.
       Non-external (pool-id) teams share ext_id==0. */
    identity->ext_id = ((params->mask & UCC_TEAM_PARAM_FIELD_ID) &&
                        (params->id <= UCC_TEAM_ID_MAX))
                           ? (uint16_t)(((uint16_t)params->id) |
                                        UCC_TEAM_ID_EXTERNAL_BIT)
                           : 0;

    if (size < 1) {
        ucc_debug("team cache identity: empty ep_map");
        return UCC_ERR_INVALID_PARAM;
    }

    members = ucc_malloc(size * sizeof(*members), "team_cache_members");
    if (ucc_unlikely(!members)) {
        ucc_error(
            "failed to allocate %zu bytes for team cache members",
            size * sizeof(*members));
        return UCC_ERR_NO_MEMORY;
    }

    /* Materialize the map into our own array so the identity never aliases the
       caller's CB/cb_ctx or user array (params may be freed after create). */
    for (i = 0; i < size; i++) {
        members[i] = ucc_ep_map_eval(params->ep_map, i);
    }

    identity->size            = size;
    identity->self_ep         = self_ep;
    identity->members         = members;
    /* Stamped collectively at cache insert (from the vote), not here; 0 = unstamped. */
    identity->instance_cookie = 0;

    /* self_ep distinguishes the same member set as seen by different ranks;
       ext_id is deliberately not hashed (membership-only bucket). */
    h                         = UCC_TEAM_CACHE_FNV1A_OFFSET;
    ucc_team_cache_fnv1a_accumulate(&h, &size, sizeof(size));
    ucc_team_cache_fnv1a_accumulate(&h, &self_ep, sizeof(self_ep));
    ucc_team_cache_fnv1a_accumulate(
        &h, members, (size_t)size * sizeof(*members));
    identity->hash = h;

    return UCC_OK;
}

int ucc_team_cache_identity_equal(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b)
{
    /* Full match for DORMANT reuse: membership AND external id/tag domain. */
    return ucc_team_cache_identity_equal_membership(a, b) &&
           a->ext_id == b->ext_id;
}

int ucc_team_cache_identity_equal_membership(
    const ucc_team_cache_identity_t *a, const ucc_team_cache_identity_t *b)
{
    if (a->hash != b->hash || a->size != b->size || a->self_ep != b->self_ep) {
        return 0;
    }
    return memcmp(
               a->members, b->members, (size_t)a->size * sizeof(*a->members)) ==
           0;
}

void ucc_team_cache_identity_free(ucc_team_cache_identity_t *identity)
{
    if (!identity) {
        return;
    }
    ucc_free(identity->members);
    memset(identity, 0, sizeof(*identity));
}

uint64_t ucc_team_cache_next_cookie(ucc_team_cache_t *cache)
{
    /* team-rank 0 only, under cache->lock. Skip 0 (unstamped sentinel). */
    uint64_t c = ++cache->cache_gen;
    if (ucc_unlikely(c == 0)) {
        c = ++cache->cache_gen;
    }
    return c;
}

void ucc_team_cache_vote_fill(
    uint64_t *v, int prepared, ucc_team_cache_action_t action, uint64_t key,
    uint64_t cookie, uint64_t parent_cookie, int is_rank0,
    uint64_t proposed_cookie)
{
    if (!prepared) {
        /* Miss rank: prepared=0 and all-ones equality lanes (a BAND no-op). */
        v[0] = 0;
        v[1] = ~(uint64_t)0;
        v[2] = ~(uint64_t)0;
        v[3] = ~(uint64_t)0;
        v[4] = ~(uint64_t)0;
        v[5] = ~(uint64_t)0;
        v[6] = ~(uint64_t)0;
        v[7] = ~(uint64_t)0;
        v[8] = ~(uint64_t)0;
    } else {
        v[0] = 1;
        v[1] = (uint64_t)action;
        v[2] = ~(uint64_t)action;
        v[3] = key;
        v[4] = ~key;
        v[5] = cookie;
        v[6] = ~cookie;
        v[7] = parent_cookie;
        v[8] = ~parent_cookie;
    }
    /* Distribution lane: rank 0 contributes its proposed new-instance cookie;
       others contribute all-ones (BAND identity), so all read rank-0's value. */
    v[9] = is_rank0 ? proposed_cookie : ~(uint64_t)0;
}

ucc_team_cache_action_t ucc_team_cache_vote_result(const uint64_t *v)
{
    int all_prepared  = (v[0] == 1);
    int action_agree  = (v[1] == ~v[2]);
    int key_agree     = (v[3] == ~v[4]);
    int cookie_agree  = (v[5] == ~v[6]);
    int pcookie_agree = (v[7] == ~v[8]);

    if (all_prepared && action_agree && key_agree && cookie_agree &&
        pcookie_agree) {
        return (ucc_team_cache_action_t)v[1];
    }
    return UCC_TEAM_CACHE_ACTION_MISS;
}

uint64_t ucc_team_cache_vote_new_cookie(const uint64_t *v)
{
    return v[9];
}

int ucc_team_cache_is_cacheable(const ucc_team_params_t *params)
{
    /* Refuse to cache teams requesting optional behavioral semantics: those
       fields are not part of the identity, so a cached team could be reused with
       different semantics. FLAGS is not copied by ucc_copy_team_params. */
    uint64_t optional = UCC_TEAM_PARAM_FIELD_ORDERING |
                        UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS |
                        UCC_TEAM_PARAM_FIELD_SYNC_TYPE |
                        UCC_TEAM_PARAM_FIELD_P2P_CONN |
                        UCC_TEAM_PARAM_FIELD_MEM_PARAMS;

    return (params->mask & optional) == 0;
}

/* Per-context team registry helpers: the single source of truth for live/dormant
   list moves. Callers hold cache->lock; a cached team is on exactly one of
   live/dormant while cached, and on neither once reaped. */

void ucc_team_cache_registry_add_live(ucc_team_cache_t *cache, ucc_team_t *team)
{
    ucc_list_add_tail(&cache->live, &team->cache_link);
}

void ucc_team_cache_registry_make_dormant(
    ucc_team_cache_t *cache, ucc_team_t *team)
{
    ucc_list_del(&team->cache_link);
    ucc_list_add_tail(&cache->dormant, &team->cache_link);
}

void ucc_team_cache_registry_make_live(
    ucc_team_cache_t *cache, ucc_team_t *team)
{
    ucc_list_del(&team->cache_link);
    ucc_list_add_tail(&cache->live, &team->cache_link);
}

void ucc_team_cache_registry_make_reserved(
    ucc_team_cache_t *cache, ucc_team_t *team)
{
    /* Pin a DORMANT re-adopt candidate for an in-flight vote: move it off dormant
       onto reserved. It stays in the bucket (chain order undisturbed) but no
       lookup/eviction/drain/destroy can reach it; refcount is left unchanged so a
       vote-FAIL can roll it back to DORMANT without a refcount race. */
    ucc_list_del(&team->cache_link);
    ucc_list_add_tail(&cache->reserved, &team->cache_link);
}

void ucc_team_cache_registry_remove(ucc_team_cache_t *cache, ucc_team_t *team)
{
    (void)cache;
    ucc_list_del(&team->cache_link);
}

void ucc_team_cache_table_erase(ucc_team_cache_t *cache, ucc_team_t *team)
{
    khash_t(
        ucc_team_cache_map) *h = (khash_t(ucc_team_cache_map) *)cache->table;
    uint64_t    hash           = team->cache_identity.hash;
    khiter_t    k;
    ucc_team_t *head, *it, *found = NULL;

    /* Unlink @team from its bucket ring; every cached team (head or sibling)
       counts toward cache->size, so a successful erase decrements it once. A team
       skipped at insert (full/collision) is on no ring: no-op. */
    k = kh_get(ucc_team_cache_map, h, hash);
    if (k == kh_end(h)) {
        return;
    }
    head = kh_value(h, k);

    UCC_TEAM_CACHE_BUCKET_FOR_EACH(it, head)
    {
        if (it == team) {
            found = team;
            break;
        }
    }
    if (!found) {
        return;
    }

    if (team != head) {
        /* Non-head sibling: unlink from the ring, head untouched. */
        ucc_list_del(&team->bucket_link);
        ucc_list_head_init(&team->bucket_link);
    } else if (ucc_list_is_empty(&head->bucket_link)) {
        /* Single-entry chain: drop the whole bucket. */
        kh_del(ucc_team_cache_map, h, k);
    } else {
        /* Head with siblings: promote the next sibling, preserving ring order. */
        ucc_team_t *next = ucc_container_of(
            head->bucket_link.next, ucc_team_t, bucket_link);
        ucc_list_del(&head->bucket_link);
        ucc_list_head_init(&head->bucket_link);
        kh_value(h, k) = next;
    }
    ucc_assert(cache->size > 0);
    cache->size--;
}

/*
 * Cache API (lookup / insert / get / put) - all operate under cache->lock held by
 * the caller. State machine (ucc_team_cache_state_t in ucc_team.h):
 *   NONE --insert--> DORMANT --get--> LIVE --put-to-0--> DORMANT
 * Same-hash teams are chained on the bucket ring, tail-appended (insertion order).
 */

/* Walk the bucket for @id->hash and return the first team matching the requested
   filter, or NULL. @match_ext_id also requires an ext_id match (exact reuse);
   @want_state restricts by cache_state; @require_derived additionally requires a
   derived team. Chain order is cross-rank identical, so every rank picks the same
   team. Caller holds cache->lock. */
static ucc_team_t *ucc_team_cache_bucket_find(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id,
    int match_ext_id, ucc_team_cache_state_t want_state, int require_derived)
{
    khash_t(
        ucc_team_cache_map) *h = (khash_t(ucc_team_cache_map) *)cache->table;
    khiter_t    k;
    ucc_team_t *head, *team;

    k = kh_get(ucc_team_cache_map, h, id->hash);
    if (k == kh_end(h)) {
        return NULL;
    }
    head = kh_value(h, k);

    UCC_TEAM_CACHE_BUCKET_FOR_EACH(team, head)
    {
        if (match_ext_id && team->cache_identity.ext_id != id->ext_id) {
            continue;
        }
        if (!cache->disable_linear_check &&
            !ucc_team_cache_identity_equal_membership(
                &team->cache_identity, id)) {
            continue;
        }
        if (team->cache_state != want_state) {
            continue;
        }
        if (require_derived && !team->is_derived) {
            continue;
        }
        return team;
    }
    return NULL;
}

ucc_team_t *ucc_team_cache_lookup(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id)
{
    ucc_team_t *team;

    cache->stats.lookups++;
    /* DORMANT-only, exact identity (ext_id compared): never return a LIVE team,
       which already backs a communicator. */
    team = ucc_team_cache_bucket_find(
        cache, id, 1, UCC_TEAM_CACHE_STATE_DORMANT, 0);
    if (team) {
        cache->stats.hits++;
        ucc_debug(
            "team_cache %p: lookup HIT team %p (hash=0x%" PRIx64 ")",
            (void *)cache,
            (void *)team,
            id->hash);
    } else {
        cache->stats.misses++;
        ucc_debug(
            "team_cache %p: lookup miss (hash=0x%" PRIx64 ")",
            (void *)cache,
            id->hash);
    }
    return team;
}

ucc_team_t *ucc_team_cache_lookup_live(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id)
{
    /* Membership-only (a derived child has a different ext_id than its live
       parent); does not count stats. */
    return ucc_team_cache_bucket_find(
        cache, id, 0, UCC_TEAM_CACHE_STATE_LIVE, 0);
}

ucc_team_t *ucc_team_cache_lookup_dormant_derived(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id)
{
    /* Membership-only, dormant, derived: a derived team's cid drifts across MPI
       context-id reuse, so the exact lookup misses it. Does not count stats. */
    return ucc_team_cache_bucket_find(
        cache, id, 0, UCC_TEAM_CACHE_STATE_DORMANT, 1);
}

ucc_status_t ucc_team_cache_insert(ucc_team_cache_t *cache, ucc_team_t *team)
{
    khash_t(
        ucc_team_cache_map) *h = (khash_t(ucc_team_cache_map) *)cache->table;
    uint64_t hash              = team->cache_identity.hash;
    khiter_t k;
    int      ret;

    /* Caller must hold cache->lock. */
    if (cache->size >= cache->max_size) {
        ucc_info(
            "team_cache %p: full (size=%u, max=%u) - team %p not cached",
            (void *)cache,
            cache->size,
            cache->max_size,
            (void *)team);
        /* cache_state stays NONE so the destroy path leaves it be. */
        return UCC_OK;
    }

    /* An existing bucket may hold a rare hash collision or same-membership teams
       with a different ext_id/state (e.g. a derived team coexisting with its live
       parent). Chain those; refuse only an exact-identity duplicate (lookup would
       have re-adopted it). */
    k = kh_get(ucc_team_cache_map, h, hash);
    if (k != kh_end(h)) {
        ucc_team_t *head = kh_value(h, k);
        ucc_team_t *it;

        UCC_TEAM_CACHE_BUCKET_FOR_EACH(it, head)
        {
            if (ucc_team_cache_identity_equal(
                    &it->cache_identity, &team->cache_identity)) {
                ucc_info(
                    "team_cache %p: duplicate identity (hash=0x%" PRIx64
                    ") - team %p not re-inserted",
                    (void *)cache,
                    hash,
                    (void *)team);
                return UCC_OK;
            }
        }

        /* Tail-append: chain order = insertion (collective) order. */
        ucc_list_add_tail(&head->bucket_link, &team->bucket_link);
        ucc_debug(
            "team_cache %p: chained team %p onto bucket "
            "(hash=0x%" PRIx64 ")",
            (void *)cache,
            (void *)team,
            hash);
    } else {
        k = kh_put(ucc_team_cache_map, h, hash, &ret);
        if (ucc_unlikely(ret < 0)) {
            ucc_error(
                "team_cache %p: kh_put failed for hash=0x%" PRIx64,
                (void *)cache,
                hash);
            return UCC_ERR_NO_MEMORY;
        }
        /* First team for this key becomes the chain head (self-linked ring). */
        kh_value(h, k) = team;
    }

    /* Insert as DORMANT at the dormant FIFO tail (MRU). The caller immediately
       transitions it DORMANT -> LIVE via registry_make_live. */
    ucc_list_add_tail(&cache->dormant, &team->cache_link);

    team->cache_state = UCC_TEAM_CACHE_STATE_DORMANT;
    cache->size++;
    cache->stats.inserts++;

    ucc_debug(
        "team_cache %p: inserted team %p (hash=0x%" PRIx64 ", size=%u)",
        (void *)cache,
        (void *)team,
        hash,
        cache->size);
    return UCC_OK;
}

void ucc_team_cache_get(ucc_team_t *team)
{
    /* Under cache->lock. DORMANT -> LIVE. */
    team->refcount++;
    team->cache_state = UCC_TEAM_CACHE_STATE_LIVE;
}

int ucc_team_cache_put(ucc_team_t *team)
{
    int rc;

    /* Under cache->lock. On rc==0: LIVE -> DORMANT (ready for re-adoption). */
    ucc_assert(team->refcount > 0); /* catch a double-put in debug builds */
    rc = --team->refcount;
    if (rc == 0) {
        team->cache_state = UCC_TEAM_CACHE_STATE_DORMANT;
    }
    return rc;
}

ucc_team_t *ucc_team_cache_pick_lru_victim(ucc_team_cache_t *cache)
{
    ucc_team_t *victim, *tmp, *best = NULL;

    /* Caller MUST hold cache->lock. All dormant-list entries are DORMANT. */
    if (ucc_list_is_empty(&cache->dormant)) {
        ucc_debug(
            "team_cache %p: pick_victim - dormant list empty", (void *)cache);
        return NULL;
    }

    if (!UCC_TEAM_CACHE_EVICTION_IS_USAGE_BASED(cache->eviction)) {
        /* FIFO (and NONE): the dormant-list head is the oldest inserted. */
        return ucc_list_head(&cache->dormant, ucc_team_t, cache_link);
    }

    /* Usage-based (LFU / LRU alias): least-used = smallest seq_num, tie-broken by
       list position (strictly-less keeps the earlier/oldest entry). */
    ucc_list_for_each_safe (victim, tmp, &cache->dormant, cache_link) {
        if (best == NULL || victim->seq_num < best->seq_num) {
            best = victim;
        }
    }
    ucc_debug(
        "team_cache %p: pick_victim LFU -> team %p (seq_num=%u)",
        (void *)cache,
        (void *)best,
        best->seq_num);
    return best;
}
