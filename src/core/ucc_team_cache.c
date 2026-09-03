/**
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

/* Chain head per key; the cache holds the table as an opaque void * */
KHASH_MAP_INIT_INT64(ucc_team_cache_map, ucc_team_t *)
typedef khash_t(ucc_team_cache_map) ucc_team_cache_map_t;

/* Walk the bucket ring headed by @_head; do not unlink @_it inside the loop */
#define UCC_TEAM_CACHE_BUCKET_FOR_EACH(_it, _head)                             \
    for ((_it) = (_head); (_it) != NULL;                                       \
         (_it) = ((_it)->bucket_link.next == &(_head)->bucket_link)            \
                     ? NULL                                                    \
                     : ucc_container_of(                                       \
                           (_it)->bucket_link.next, ucc_team_t, bucket_link))

/* Order must match ucc_team_cache_eviction_policy_t */
const char *ucc_team_cache_eviction_names[] = {
    [UCC_TEAM_CACHE_EVICTION_NONE] = "none",
    [UCC_TEAM_CACHE_EVICTION_FIFO] = "fifo",
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

    /* All teams are quiesced at context destroy, so no vote is in flight */
    ucc_assert(ucc_list_is_empty(&cache->reserved));

    kh_destroy(ucc_team_cache_map, (ucc_team_cache_map_t *)cache->table);
    ucc_spinlock_destroy(&cache->lock);
    ucc_debug("ucc_team_cache destroyed: %p", (void *)cache);
    ucc_free(cache);
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

/* FNV-1a over {size, self_ep, members}, used only as a bucket key */
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
        ucc_debug("team cache identity: no EP_MAP in params, not cacheable");
        return UCC_ERR_INVALID_PARAM;
    }

    size    = (ucc_rank_t)params->ep_map.ep_num;
    self_ep = (params->mask & UCC_TEAM_PARAM_FIELD_EP) ? (ucc_rank_t)params->ep
                                                       : UCC_RANK_INVALID;

    /* An external id is its own id/tag domain, so it is part of the identity */
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

    /* Copy the map so the identity never aliases caller-owned storage */
    for (i = 0; i < size; i++) {
        members[i] = ucc_ep_map_eval(params->ep_map, i);
    }

    identity->size            = size;
    identity->self_ep         = self_ep;
    identity->members         = members;
    identity->instance_cookie = 0; /* stamped later by the agreement vote */

    /* ext_id is deliberately not hashed, so the bucket is membership only */
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
    uint64_t c = ++cache->cache_gen; /* 0 is the unstamped sentinel */
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
        /* All-ones equality lanes are a BAND no-op */
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
    /* Only rank 0 contributes here, so every member reads its value */
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
    /* These are not part of the identity, so a reuse could change semantics */
    uint64_t optional = UCC_TEAM_PARAM_FIELD_ORDERING |
                        UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS |
                        UCC_TEAM_PARAM_FIELD_SYNC_TYPE |
                        UCC_TEAM_PARAM_FIELD_P2P_CONN |
                        UCC_TEAM_PARAM_FIELD_MEM_PARAMS;

    return (params->mask & optional) == 0;
}

/* A cached team is on exactly one of live/dormant/reserved, or on none */

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
    /* Stays in the bucket, but no lookup, eviction or drain can reach it */
    ucc_list_del(&team->cache_link);
    ucc_list_add_tail(&cache->reserved, &team->cache_link);
}

void ucc_team_cache_registry_remove(ucc_team_t *team)
{
    ucc_list_del(&team->cache_link);
}

void ucc_team_cache_table_erase(ucc_team_cache_t *cache, ucc_team_t *team)
{
    ucc_team_cache_map_t *h    = (ucc_team_cache_map_t *)cache->table;
    uint64_t              hash = team->cache_identity.hash;
    khiter_t              k;
    ucc_team_t           *head, *it, *found = NULL;

    /* A team skipped at insert is on no ring, so the erase is a no-op */
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
        /* Non-head sibling: unlink from the ring, head untouched */
        ucc_list_del(&team->bucket_link);
        ucc_list_head_init(&team->bucket_link);
    } else if (ucc_list_is_empty(&head->bucket_link)) {
        /* Single-entry chain: drop the whole bucket */
        kh_del(ucc_team_cache_map, h, k);
    } else {
        /* Promote the next sibling, preserving ring order */
        ucc_team_t *next = ucc_container_of(
            head->bucket_link.next, ucc_team_t, bucket_link);

        ucc_list_del(&head->bucket_link);
        ucc_list_head_init(&head->bucket_link);
        kh_value(h, k) = next;
    }
    ucc_assert(cache->size > 0);
    cache->size--;
}

/* NONE --insert--> DORMANT --get--> LIVE --put-to-0--> DORMANT */

/* First team in @id's bucket passing every filter, or NULL; under @lock */
static ucc_team_t *ucc_team_cache_bucket_find(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id,
    int match_ext_id, ucc_team_cache_state_t want_state, int require_derived)
{
    ucc_team_cache_map_t *h = (ucc_team_cache_map_t *)cache->table;
    khiter_t              k;
    ucc_team_t           *head, *team;

    k = kh_get(ucc_team_cache_map, h, id->hash);
    if (k == kh_end(h)) {
        return NULL;
    }
    head = kh_value(h, k);

    UCC_TEAM_CACHE_BUCKET_FOR_EACH(team, head)
    {
        /* A different ext_id is a different tag domain, not a reuse */
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
        /* Only a derived team may be re-seated into another tag domain */
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
    /* A LIVE team already backs a communicator, so match DORMANT only */
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
    /* A derived child's ext_id differs from its parent's, so ignore it */
    return ucc_team_cache_bucket_find(
        cache, id, 0, UCC_TEAM_CACHE_STATE_LIVE, 0);
}

ucc_team_t *ucc_team_cache_lookup_dormant_derived(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id)
{
    /* A drifted cid makes the exact lookup miss, so match membership only */
    return ucc_team_cache_bucket_find(
        cache, id, 0, UCC_TEAM_CACHE_STATE_DORMANT, 1);
}

ucc_status_t ucc_team_cache_insert(ucc_team_cache_t *cache, ucc_team_t *team)
{
    ucc_team_cache_map_t *h    = (ucc_team_cache_map_t *)cache->table;
    uint64_t              hash = team->cache_identity.hash;
    khiter_t              k;
    int                   ret;

    if (cache->size >= cache->max_size) {
        ucc_info(
            "team_cache %p: full (size=%u, max=%u) - team %p not cached",
            (void *)cache,
            cache->size,
            cache->max_size,
            (void *)team);
        return UCC_OK; /* cache_state stays NONE, so destroy tears it down */
    }

    /* Same-hash teams coexist on one ring, so only a re-insert is refused */
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

        /* Tail-append: chain order = insertion (collective) order */
        ucc_list_add_tail(&head->bucket_link, &team->bucket_link);
        ucc_debug(
            "team_cache %p: chained team %p onto bucket (hash=0x%" PRIx64 ")",
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
        /* First team for this key becomes the chain head (self-linked ring) */
        kh_value(h, k) = team;
    }

    /* The caller immediately promotes this to LIVE via registry_make_live */
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
    team->refcount++;
    team->cache_state = UCC_TEAM_CACHE_STATE_LIVE;
}

int ucc_team_cache_put(ucc_team_t *team)
{
    int rc;

    ucc_assert(team->refcount > 0); /* catches a double put in debug builds */
    rc = --team->refcount;
    if (rc == 0) {
        team->cache_state = UCC_TEAM_CACHE_STATE_DORMANT;
    }
    return rc;
}

ucc_team_t *ucc_team_cache_pick_victim(ucc_team_cache_t *cache)
{
    if (ucc_list_is_empty(&cache->dormant)) {
        ucc_debug(
            "team_cache %p: pick_victim - dormant list empty", (void *)cache);
        return NULL;
    }

    /* For FIFO and NONE alike, the list head is the oldest insert */
    return ucc_list_head(&cache->dormant, ucc_team_t, cache_link);
}
