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

/* One team per key; the cache holds the table as an opaque void * */
KHASH_MAP_INIT_INT64(ucc_team_cache_map, ucc_team_t *)
typedef khash_t(ucc_team_cache_map) ucc_team_cache_map_t;

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

    k = kh_get(ucc_team_cache_map, h, hash);
    if (k == kh_end(h) || kh_value(h, k) != team) {
        /* Absent, or a colliding team owns the bucket */
        return;
    }
    kh_del(ucc_team_cache_map, h, k);
    ucc_assert(cache->size > 0);
    cache->size--;
}

/* NONE --insert--> DORMANT --get--> LIVE --put-to-0--> DORMANT */

ucc_team_t *ucc_team_cache_lookup(
    ucc_team_cache_t *cache, const ucc_team_cache_identity_t *id)
{
    ucc_team_cache_map_t *h = (ucc_team_cache_map_t *)cache->table;
    khiter_t              k;
    ucc_team_t           *team;

    cache->stats.lookups++;

    k = kh_get(ucc_team_cache_map, h, id->hash);
    if (k == kh_end(h)) {
        cache->stats.misses++;
        ucc_debug(
            "team_cache %p: lookup miss (hash=0x%" PRIx64 ")",
            (void *)cache,
            id->hash);
        return NULL;
    }

    team = kh_value(h, k);

    /* A different ext_id is a different tag domain, so not a valid reuse */
    if (team->cache_identity.ext_id != id->ext_id) {
        cache->stats.misses++;
        return NULL;
    }

    if (!cache->disable_linear_check &&
        !ucc_team_cache_identity_equal_membership(&team->cache_identity, id)) {
        cache->stats.misses++;
        return NULL;
    }

    /* Only a DORMANT team is free to re-adopt */
    if (team->cache_state != UCC_TEAM_CACHE_STATE_DORMANT) {
        cache->stats.misses++;
        return NULL;
    }

    cache->stats.hits++;
    ucc_debug(
        "team_cache %p: lookup HIT team %p (hash=0x%" PRIx64 ")",
        (void *)cache,
        (void *)team,
        id->hash);
    return team;
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

    /* An occupied bucket leaves the new team uncached but still functional */
    k = kh_get(ucc_team_cache_map, h, hash);
    if (k != kh_end(h)) {
        ucc_info(
            "team_cache %p: bucket occupied (hash=0x%" PRIx64
            ") - team %p not cached (duplicate or hash collision)",
            (void *)cache,
            hash,
            (void *)team);
        return UCC_OK;
    }

    k = kh_put(ucc_team_cache_map, h, hash, &ret);
    if (ucc_unlikely(ret < 0)) {
        ucc_error(
            "team_cache %p: kh_put failed for hash=0x%" PRIx64,
            (void *)cache,
            hash);
        return UCC_ERR_NO_MEMORY;
    }
    kh_value(h, k) = team;

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
