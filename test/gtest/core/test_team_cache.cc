/**
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
extern "C" {
#include "core/ucc_team_cache.h"
#include "core/ucc_team.h"
#include "utils/ucc_spinlock.h"
}
#include <common/test.h>
#include <common/test_ucc.h>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <atomic>
#include <random>

/* Unit tests for the team-cache identity (build/hash/equal/free), the
   cacheability policy, the locked cache API and eviction, with no MPI job. */

/* CB closure returning member[ep] from a heap vector (mimics OMPI coll/ucc's
   rank_map_cb); freeable after build to prove identity does not retain it. */
struct cb_ctx {
    std::vector<ucc_rank_t> members;
};

static uint64_t member_cb(uint64_t ep, void *ctx)
{
    cb_ctx *c = static_cast<cb_ctx *>(ctx);
    return (uint64_t)c->members[ep];
}

static ucc_team_params_t make_cb_params(cb_ctx *ctx, ucc_rank_t self_ep)
{
    ucc_team_params_t p;
    memset(&p, 0, sizeof(p));
    p.mask             = UCC_TEAM_PARAM_FIELD_EP_MAP | UCC_TEAM_PARAM_FIELD_EP;
    p.ep               = self_ep;
    p.ep_map.type      = UCC_EP_MAP_CB;
    p.ep_map.ep_num    = ctx->members.size();
    p.ep_map.cb.cb     = member_cb;
    p.ep_map.cb.cb_ctx = ctx;
    return p;
}

/* ARRAY+OOB style, as OpenSHMEM scoll/ucc passes it: a user-owned array. */
static ucc_team_params_t make_array_params(
    ucc_rank_t *arr, ucc_rank_t size, ucc_rank_t self_ep)
{
    ucc_team_params_t p;
    memset(&p, 0, sizeof(p));
    p.mask                   = UCC_TEAM_PARAM_FIELD_EP_MAP |
                               UCC_TEAM_PARAM_FIELD_EP;
    p.ep                     = self_ep;
    p.ep_map.type            = UCC_EP_MAP_ARRAY;
    p.ep_map.ep_num          = size;
    p.ep_map.array.map       = arr;
    p.ep_map.array.elem_size = sizeof(ucc_rank_t);
    return p;
}

static ucc_team_params_t make_strided_params(
    uint64_t start, int64_t stride, ucc_rank_t size, ucc_rank_t self_ep)
{
    ucc_team_params_t p;
    memset(&p, 0, sizeof(p));
    p.mask                  = UCC_TEAM_PARAM_FIELD_EP_MAP |
                              UCC_TEAM_PARAM_FIELD_EP;
    p.ep                    = self_ep;
    p.ep_map.type           = UCC_EP_MAP_STRIDED;
    p.ep_map.ep_num         = size;
    p.ep_map.strided.start  = start;
    p.ep_map.strided.stride = stride;
    return p;
}

/* ARRAY membership plus a caller-supplied external team id (FIELD_ID), as an
   MPI communicator passes its context id. */
static ucc_team_params_t make_array_id_params(
    ucc_rank_t *arr, ucc_rank_t size, ucc_rank_t self_ep, uint64_t id)
{
    ucc_team_params_t p = make_array_params(arr, size, self_ep);
    p.mask |= UCC_TEAM_PARAM_FIELD_ID;
    p.id = id;
    return p;
}

/* Zero-init an identity and build it from @p, asserting success.  Callers must
   ucc_team_cache_identity_free the result. */
static void build_identity(
    const ucc_team_params_t &p, ucc_team_cache_identity_t &id)
{
    memset(&id, 0, sizeof(id));
    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p, &id));
}

/* Assert two params materialize to an equal identity (same hash AND
   exact-compare equal).  Frees both identities. */
static void expect_identities_equal(
    const ucc_team_params_t &pa, const ucc_team_params_t &pb)
{
    ucc_team_cache_identity_t a, b;
    build_identity(pa, a);
    build_identity(pb, b);
    EXPECT_EQ(a.hash, b.hash);
    EXPECT_NE(0, ucc_team_cache_identity_equal(&a, &b));
    ucc_team_cache_identity_free(&a);
    ucc_team_cache_identity_free(&b);
}

class test_team_cache : public ucc::test {};

/* Identical membership + DIFFERENT external ids must NOT be full-equal (no
   dormant reuse across id/tag domains), but must stay membership-equal so
   coexistence/derived detection finds the live parent.  Same id -> fully equal. */
UCC_TEST_F(test_team_cache, external_id_isolates_dormant_reuse)
{
    ucc_rank_t                arr[4] = {0, 1, 2, 3};
    ucc_team_params_t         p3     = make_array_id_params(arr, 4, 1, 3);
    ucc_team_params_t         p3b    = make_array_id_params(arr, 4, 1, 3);
    ucc_team_params_t         p5     = make_array_id_params(arr, 4, 1, 5);

    ucc_team_cache_identity_t a, b, c;
    build_identity(p3, a);
    build_identity(p3b, b);
    build_identity(p5, c);

    /* Membership-only hash: all three share a hash bucket. */
    EXPECT_EQ(a.hash, b.hash);
    EXPECT_EQ(a.hash, c.hash);

    /* Same members + same id -> full match; different id -> no full match. */
    EXPECT_NE(0, ucc_team_cache_identity_equal(&a, &b));
    EXPECT_EQ(0, ucc_team_cache_identity_equal(&a, &c));
    /* Membership matches regardless of id (derived/coexistence detection). */
    EXPECT_NE(0, ucc_team_cache_identity_equal_membership(&a, &c));

    ucc_team_cache_identity_free(&a);
    ucc_team_cache_identity_free(&b);
    ucc_team_cache_identity_free(&c);
}

/* Identity ignores ep_map style / closure pointers: two CB closures, CB vs
   ARRAY, and FULL/STRIDED(0,1) vs ARRAY [0..size) all produce an equal identity. */
UCC_TEST_F(test_team_cache, cross_style_cb_vs_array_equal)
{
    ucc_rank_t same[4] = {3, 5, 7, 9};
    expect_identities_equal(
        make_array_params(same, 4, 1), make_array_params(same, 4, 1));

    /* Two distinct CB closures materializing the same members. */
    cb_ctx c1, c2;
    c1.members = {2, 4, 6, 8};
    c2.members = {2, 4, 6, 8};
    expect_identities_equal(make_cb_params(&c1, 0), make_cb_params(&c2, 0));

    /* Cross-style: EP_MAP CB (coll/ucc) vs EP_MAP_ARRAY (scoll/ucc). */
    cb_ctx c;
    c.members          = {10, 20, 30};
    ucc_rank_t arr3[3] = {10, 20, 30};
    expect_identities_equal(
        make_cb_params(&c, 2), make_array_params(arr3, 3, 2));

    /* CB / ARRAY / FULL / STRIDED(0,1) over [0..5) must all be equal. */
    ucc_rank_t arr[5] = {0, 1, 2, 3, 4};
    cb_ctx     c5;
    c5.members             = {0, 1, 2, 3, 4};
    ucc_team_params_t pcb  = make_cb_params(&c5, 0);
    ucc_team_params_t parr = make_array_params(arr, 5, 0);
    ucc_team_params_t pstr = make_strided_params(0, 1, 5, 0);

    ucc_team_params_t pfull;
    memset(&pfull, 0, sizeof(pfull));
    pfull.mask          = UCC_TEAM_PARAM_FIELD_EP_MAP | UCC_TEAM_PARAM_FIELD_EP;
    pfull.ep            = 0;
    pfull.ep_map.type   = UCC_EP_MAP_FULL;
    pfull.ep_map.ep_num = 5;

    ucc_team_cache_identity_t cb, a, b, full;
    build_identity(pcb, cb);
    build_identity(parr, a);
    build_identity(pstr, b);
    build_identity(pfull, full);

    EXPECT_NE(0, ucc_team_cache_identity_equal(&a, &cb));
    EXPECT_NE(0, ucc_team_cache_identity_equal(&a, &b));
    EXPECT_NE(0, ucc_team_cache_identity_equal(&a, &full));

    ucc_team_cache_identity_free(&cb);
    ucc_team_cache_identity_free(&a);
    ucc_team_cache_identity_free(&b);
    ucc_team_cache_identity_free(&full);
}

/* Freeing/mutating the caller's closure and user array after build does not
   change the identity (params are materialized, not retained). */
UCC_TEST_F(test_team_cache, identity_owns_materialized_members)
{
    cb_ctx *c       = new cb_ctx();
    c->members      = {11, 13, 17, 19};

    ucc_rank_t *arr = (ucc_rank_t *)malloc(4 * sizeof(ucc_rank_t));
    ASSERT_NE(nullptr, arr);
    arr[0]                         = 11;
    arr[1]                         = 13;
    arr[2]                         = 17;
    arr[3]                         = 19;

    ucc_team_params_t         pcb  = make_cb_params(c, 3);
    ucc_team_params_t         parr = make_array_params(arr, 4, 3);

    ucc_team_cache_identity_t from_cb, from_arr, ref;
    build_identity(pcb, from_cb);
    build_identity(parr, from_arr);

    ucc_rank_t        refarr[4] = {11, 13, 17, 19};
    ucc_team_params_t pref      = make_array_params(refarr, 4, 3);
    build_identity(pref, ref);

    /* Destroy/mutate the caller-owned inputs. */
    delete c;
    arr[0] = 999;
    arr[2] = 42;
    free(arr);

    EXPECT_EQ(ref.hash, from_cb.hash);
    EXPECT_EQ(ref.hash, from_arr.hash);
    EXPECT_NE(0, ucc_team_cache_identity_equal(&ref, &from_cb));
    EXPECT_NE(0, ucc_team_cache_identity_equal(&ref, &from_arr));

    ucc_team_cache_identity_free(&from_cb);
    ucc_team_cache_identity_free(&from_arr);
    ucc_team_cache_identity_free(&ref);
}

/* Differing membership -> not equal (size, self_ep, array contents, stride). */
UCC_TEST_F(test_team_cache, differing_membership_not_equal)
{
    ucc_rank_t                base[4]     = {1, 2, 3, 4};
    ucc_rank_t                diff_val[4] = {1, 2, 3, 5};
    ucc_rank_t                diff_len[3] = {1, 2, 3};

    ucc_team_params_t         pbase       = make_array_params(base, 4, 1);
    ucc_team_params_t         pval        = make_array_params(diff_val, 4, 1);
    ucc_team_params_t         plen        = make_array_params(diff_len, 3, 1);
    ucc_team_params_t         pep         = make_array_params(base, 4, 2);
    ucc_team_params_t         pstr1       = make_strided_params(0, 1, 4, 0);
    ucc_team_params_t         pstr2       = make_strided_params(0, 2, 4, 0);

    ucc_team_cache_identity_t base_id, val_id, len_id, ep_id, s1, s2;
    build_identity(pbase, base_id);
    build_identity(pval, val_id);
    build_identity(plen, len_id);
    build_identity(pep, ep_id);
    build_identity(pstr1, s1);
    build_identity(pstr2, s2);

    EXPECT_EQ(0, ucc_team_cache_identity_equal(&base_id, &val_id));
    EXPECT_EQ(0, ucc_team_cache_identity_equal(&base_id, &len_id));
    EXPECT_EQ(0, ucc_team_cache_identity_equal(&base_id, &ep_id));
    EXPECT_EQ(0, ucc_team_cache_identity_equal(&s1, &s2));

    ucc_team_cache_identity_free(&base_id);
    ucc_team_cache_identity_free(&val_id);
    ucc_team_cache_identity_free(&len_id);
    ucc_team_cache_identity_free(&ep_id);
    ucc_team_cache_identity_free(&s1);
    ucc_team_cache_identity_free(&s2);
}

/* is_cacheable: true when only EP_MAP/EP/OOB/etc are set; false when any
   optional behavioral field is set.  FLAGS is ignored. */
UCC_TEST_F(test_team_cache, is_cacheable_policy)
{
    ucc_rank_t        arr[2] = {0, 1};
    ucc_team_params_t p      = make_array_params(arr, 2, 0);

    EXPECT_NE(0, ucc_team_cache_is_cacheable(&p));

    ucc_team_params_t pflags = p;
    pflags.mask |= UCC_TEAM_PARAM_FIELD_FLAGS;
    pflags.flags = 0x1;
    EXPECT_NE(0, ucc_team_cache_is_cacheable(&pflags));

    const uint64_t optional[] = {
        UCC_TEAM_PARAM_FIELD_ORDERING,
        UCC_TEAM_PARAM_FIELD_OUTSTANDING_COLLS,
        UCC_TEAM_PARAM_FIELD_SYNC_TYPE,
        UCC_TEAM_PARAM_FIELD_P2P_CONN,
        UCC_TEAM_PARAM_FIELD_MEM_PARAMS,
    };
    for (size_t i = 0; i < sizeof(optional) / sizeof(optional[0]); i++) {
        ucc_team_params_t po = p;
        po.mask |= optional[i];
        EXPECT_EQ(0, ucc_team_cache_is_cacheable(&po))
            << "optional field index " << i << " should block caching";
    }
}

/* Build a lookup key from ARRAY membership + external id. */
static void build_id_key(
    ucc_rank_t *arr, ucc_rank_t size, ucc_rank_t self_ep, uint64_t id,
    ucc_team_cache_identity_t &key)
{
    ucc_team_params_t p = make_array_id_params(arr, size, self_ep, id);
    build_identity(p, key);
}

/* RAII wrapper for ucc_team_cache_init/destroy; args match ucc_team_cache_init.
   Implicitly usable as a ucc_team_cache_t*. */
struct ScopedCache {
    ucc_team_cache_t *cache = nullptr;

    ScopedCache(
        uint32_t max_size, ucc_team_cache_eviction_policy_t evict,
        int disable_linear_check)
    {
        EXPECT_EQ(
            UCC_OK,
            ucc_team_cache_init(&cache, max_size, evict, disable_linear_check));
    }
    ~ScopedCache()
    {
        ucc_team_cache_destroy(cache);
    }

    operator ucc_team_cache_t *() const
    {
        return cache;
    }
    ucc_team_cache_t *operator->() const
    {
        return cache;
    }
};

static bool team_on_list(ucc_list_link_t *head, ucc_team_t *team)
{
    ucc_team_t *t;
    ucc_list_for_each (t, head, cache_link) {
        if (t == team) {
            return true;
        }
    }
    return false;
}

/* Registry list-surgery matrix: add-live -> make-dormant -> make-live -> remove.
   Asserts the team is on exactly one list (or none after remove) at each step. */
UCC_TEST_F(test_team_cache, registry_add_dormant_live_remove)
{
    ScopedCache cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_team_t  team;
    memset(&team, 0, sizeof(team));
    ucc_list_head_init(&team.cache_link);

    EXPECT_FALSE(team_on_list(&cache->live, &team));
    EXPECT_FALSE(team_on_list(&cache->dormant, &team));

    ucc_team_cache_registry_add_live(cache, &team);
    EXPECT_TRUE(team_on_list(&cache->live, &team));
    EXPECT_FALSE(team_on_list(&cache->dormant, &team));
    EXPECT_FALSE(ucc_list_is_empty(&cache->live));

    ucc_team_cache_registry_make_dormant(cache, &team);
    EXPECT_FALSE(team_on_list(&cache->live, &team));
    EXPECT_TRUE(team_on_list(&cache->dormant, &team));
    EXPECT_TRUE(ucc_list_is_empty(&cache->live));
    EXPECT_FALSE(ucc_list_is_empty(&cache->dormant));

    ucc_team_cache_registry_make_live(cache, &team);
    EXPECT_TRUE(team_on_list(&cache->live, &team));
    EXPECT_FALSE(team_on_list(&cache->dormant, &team));

    ucc_team_cache_registry_remove(&team);
    EXPECT_FALSE(team_on_list(&cache->live, &team));
    EXPECT_FALSE(team_on_list(&cache->dormant, &team));
    EXPECT_TRUE(ucc_list_is_empty(&cache->live));
    EXPECT_TRUE(ucc_list_is_empty(&cache->dormant));
}

/* Cache API tests (lookup / insert / get / put) use bare stub teams with only
   the cache-related fields initialized; they never call create_post. */

/* Allocate and minimally initialize a stub team. refcount starts at 0 to match
   the production convention for a cached DORMANT team. */
static ucc_team_t *alloc_stub_team(void)
{
    ucc_team_t *t = (ucc_team_t *)calloc(1, sizeof(*t));
    if (!t) {
        return nullptr;
    }
    t->refcount    = 0;
    t->cache_state = UCC_TEAM_CACHE_STATE_NONE;
    ucc_list_head_init(&t->cache_link);
    memset(&t->cache_identity, 0, sizeof(t->cache_identity));
    return t;
}

static void free_stub_team(ucc_team_t *t)
{
    ucc_team_cache_identity_free(&t->cache_identity);
    free(t);
}

/* Detach a stub team from the table + registry.  Caller must hold cache->lock. */
static void erase_stub(ucc_team_cache_t *cache, ucc_team_t *t)
{
    ucc_team_cache_table_erase(cache, t);
    ucc_team_cache_registry_remove(t);
}

/* Lock, erase, unlock, and free each team - the common stub teardown. */
static void erase_and_free(
    ucc_team_cache_t *cache, std::initializer_list<ucc_team_t *> teams)
{
    ucc_spin_lock(&cache->lock);
    for (ucc_team_t *t : teams) {
        erase_stub(cache, t);
    }
    ucc_spin_unlock(&cache->lock);
    for (ucc_team_t *t : teams) {
        free_stub_team(t);
    }
}

/* lookup does NOT return a LIVE team: insert -> get (DORMANT->LIVE) -> miss. */
UCC_TEST_F(test_team_cache, lookup_does_not_return_live)
{
    ScopedCache       cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_rank_t        arr[3] = {2, 4, 6};
    ucc_team_params_t p      = make_array_params(arr, 3, 0);

    ucc_team_t       *team   = alloc_stub_team();
    ASSERT_NE(nullptr, team);
    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p, &team->cache_identity));

    ucc_team_cache_identity_t key;
    build_identity(p, key);

    ucc_spin_lock(&cache->lock);
    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, team));
    ucc_team_cache_get(team);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, team->cache_state);
    ucc_team_t *found = ucc_team_cache_lookup(cache, &key);
    ucc_spin_unlock(&cache->lock);

    EXPECT_EQ(nullptr, found);
    EXPECT_EQ(0u, cache->stats.hits);

    ucc_team_cache_identity_free(&key);
    /* Release team back to dormant so erase_and_free can clean it up. */
    ucc_spin_lock(&cache->lock);
    ucc_team_cache_put(team);
    ucc_team_cache_registry_make_dormant(cache, team);
    ucc_spin_unlock(&cache->lock);
    erase_and_free(cache, {team});
}

/* Full-cache skip: second insert into a full max_size==1 cache stays NONE. */
UCC_TEST_F(test_team_cache, full_cache_skip)
{
    ScopedCache       cache(1, UCC_TEAM_CACHE_EVICTION_NONE, 0);

    ucc_rank_t        arr1[2] = {0, 1};
    ucc_rank_t        arr2[2] = {2, 3};
    ucc_team_params_t p1      = make_array_params(arr1, 2, 0);
    ucc_team_params_t p2      = make_array_params(arr2, 2, 0);

    ucc_team_t       *t1      = alloc_stub_team();
    ucc_team_t       *t2      = alloc_stub_team();
    ASSERT_NE(nullptr, t1);
    ASSERT_NE(nullptr, t2);

    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p1, &t1->cache_identity));
    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p2, &t2->cache_identity));

    ucc_spin_lock(&cache->lock);
    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t1));
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, t1->cache_state);
    EXPECT_EQ(1u, cache->size);

    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t2));
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_NONE, t2->cache_state);
    EXPECT_EQ(1u, cache->size);
    ucc_spin_unlock(&cache->lock);

    erase_and_free(
        cache, {t1});   /* t1 was inserted (DORMANT); remove before free */
    free_stub_team(t2); /* t2 was not inserted (NONE); free directly */
}

/* A duplicate identity finds the bucket occupied and is left uncached. */
UCC_TEST_F(test_team_cache, duplicate_identity_skips_insert)
{
    ScopedCache       cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_rank_t        arr[3] = {5, 6, 7};
    ucc_team_params_t p      = make_array_params(arr, 3, 0);

    ucc_team_t       *t1     = alloc_stub_team();
    ucc_team_t       *t2     = alloc_stub_team();
    ASSERT_NE(nullptr, t1);
    ASSERT_NE(nullptr, t2);

    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p, &t1->cache_identity));
    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p, &t2->cache_identity));

    ucc_spin_lock(&cache->lock);
    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t1));
    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t2));
    ucc_spin_unlock(&cache->lock);

    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, t1->cache_state);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_NONE, t2->cache_state);
    EXPECT_EQ(1u, cache->size);
    EXPECT_EQ(1u, cache->stats.inserts);

    erase_and_free(cache, {t1});
    free_stub_team(t2);
}

/* get/put refcount arithmetic and LIVE<->DORMANT transitions. */
UCC_TEST_F(test_team_cache, get_put_refcount_and_state)
{
    ScopedCache       cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_rank_t        arr[2] = {0, 1};
    ucc_team_params_t p      = make_array_params(arr, 2, 0);

    ucc_team_t       *team   = alloc_stub_team(); /* refcount 0 (DORMANT) */
    ASSERT_NE(nullptr, team);
    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p, &team->cache_identity));

    ucc_team_cache_identity_t key;
    build_identity(p, key);

    ucc_spin_lock(&cache->lock);
    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, team));
    EXPECT_EQ(0, team->refcount);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, team->cache_state);

    /* Adopt the dormant team: 0 -> 1, LIVE. */
    ucc_team_cache_get(team);
    ucc_team_cache_registry_make_live(cache, team);
    EXPECT_EQ(1, team->refcount);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, team->cache_state);

    /* Last user drops: 1 -> 0, DORMANT. */
    EXPECT_EQ(0, ucc_team_cache_put(team));
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, team->cache_state);
    ucc_team_cache_registry_make_dormant(cache, team);

    /* A team that went LIVE then back to DORMANT is look-up-able again. */
    EXPECT_EQ(team, ucc_team_cache_lookup(cache, &key));

    /* Two live users need two puts to return to DORMANT. */
    ucc_team_cache_get(team);
    ucc_team_cache_registry_make_live(cache, team);
    ucc_team_cache_get(team);
    EXPECT_EQ(2, team->refcount);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, team->cache_state);

    EXPECT_EQ(1, ucc_team_cache_put(team));
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, team->cache_state);
    EXPECT_EQ(0, ucc_team_cache_put(team));
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, team->cache_state);
    ucc_team_cache_registry_make_dormant(cache, team);
    ucc_spin_unlock(&cache->lock);

    ucc_team_cache_identity_free(&key);
    erase_and_free(cache, {team});
}

/* RESERVED state (the agreement-vote pin): a DORMANT candidate moved to RESERVED
   is off the dormant/live lists (lookup can't return it) but stays in the bucket
   with refcount unchanged, so a vote-FAIL rolls it back to DORMANT and a vote-PASS
   promotes it to LIVE via get (0 -> 1). */
UCC_TEST_F(test_team_cache, reserved_state_pin_and_rollback)
{
    ScopedCache       cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);
    ucc_rank_t        arr[3] = {10, 20, 30};
    ucc_team_params_t p      = make_array_params(arr, 3, 0);

    ucc_team_t       *t      = alloc_stub_team();
    ASSERT_NE(nullptr, t);
    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p, &t->cache_identity));

    ucc_spin_lock(&cache->lock);
    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, t));
    ASSERT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, t->cache_state);

    ucc_team_cache_identity_t key;
    build_identity(p, key);

    ASSERT_EQ(t, ucc_team_cache_lookup(cache, &key));

    /* Pin for an in-flight vote: DORMANT -> RESERVED, refcount untouched. */
    ucc_team_cache_registry_make_reserved(cache, t);
    t->cache_state = UCC_TEAM_CACHE_STATE_RESERVED;
    EXPECT_EQ(0, t->refcount);
    /* lookup is DORMANT-only: a RESERVED team must not be returned. */
    EXPECT_EQ(nullptr, ucc_team_cache_lookup(cache, &key));

    /* Vote FAIL: roll back RESERVED -> DORMANT, re-adoptable. */
    ucc_team_cache_registry_make_dormant(cache, t);
    t->cache_state = UCC_TEAM_CACHE_STATE_DORMANT;
    EXPECT_EQ(t, ucc_team_cache_lookup(cache, &key));

    /* Vote PASS: RESERVED -> LIVE via get (0 -> 1). */
    ucc_team_cache_registry_make_reserved(cache, t);
    t->cache_state = UCC_TEAM_CACHE_STATE_RESERVED;
    ucc_team_cache_get(t);
    ucc_team_cache_registry_make_live(cache, t);
    EXPECT_EQ(1, t->refcount);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, t->cache_state);
    EXPECT_EQ(
        nullptr, ucc_team_cache_lookup(cache, &key)); /* LIVE not returned */
    ucc_spin_unlock(&cache->lock);

    ucc_team_cache_identity_free(&key);
    erase_and_free(cache, {t});
}

/* Build a stub team with @arr membership and insert it into @cache as DORMANT. */
static ucc_team_t *insert_stub_dormant(
    ucc_team_cache_t *cache, ucc_rank_t *arr, ucc_rank_t n, ucc_rank_t self_ep)
{
    ucc_team_params_t p = make_array_params(arr, n, self_ep);

    ucc_team_t       *t = alloc_stub_team();
    if (!t) {
        return nullptr;
    }
    if (UCC_OK != ucc_team_cache_identity_build(&p, &t->cache_identity)) {
        free_stub_team(t);
        return nullptr;
    }

    ucc_spin_lock(&cache->lock);
    ucc_status_t st = ucc_team_cache_insert(cache, t);
    ucc_spin_unlock(&cache->lock);

    if (st != UCC_OK || t->cache_state != UCC_TEAM_CACHE_STATE_DORMANT) {
        free_stub_team(t);
        return nullptr;
    }
    return t;
}

/* Insert three distinct-membership DORMANT stub teams (list order A,B,C). */
static void insert_three_dormant(
    ucc_team_cache_t *cache, ucc_team_t **tA, ucc_team_t **tB, ucc_team_t **tC)
{
    ucc_rank_t mA[3] = {10, 20, 30};
    ucc_rank_t mB[3] = {11, 21, 31};
    ucc_rank_t mC[3] = {12, 22, 32};
    *tA              = insert_stub_dormant(cache, mA, 3, 0);
    *tB              = insert_stub_dormant(cache, mB, 3, 0);
    *tC              = insert_stub_dormant(cache, mC, 3, 0);
    ASSERT_NE(nullptr, *tA);
    ASSERT_NE(nullptr, *tB);
    ASSERT_NE(nullptr, *tC);
}

/* Victim selection: FIFO returns the insertion head (oldest). */
UCC_TEST_F(test_team_cache, evict_victim_selection)
{
    SCOPED_TRACE("fifo");
    ScopedCache cache(8, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_team_t *tA, *tB, *tC;
    insert_three_dormant(cache, &tA, &tB, &tC);
    ASSERT_EQ(3u, cache->size);

    tA->seq_num = 10;
    tB->seq_num = 3;
    tC->seq_num = 20;

    ucc_spin_lock(&cache->lock);
    ucc_team_t *victim = ucc_team_cache_pick_victim(cache);
    ucc_spin_unlock(&cache->lock);

    /* FIFO: oldest inserted (tA) regardless of seq_num. */
    EXPECT_EQ(tA, victim);

    erase_and_free(cache, {tA, tB, tC});
}

/* pick_victim skips LIVE teams: adopt both -> dormant empty -> NULL;
   release both -> a victim is returned. */
UCC_TEST_F(test_team_cache, evict_skips_live_returns_no_resource)
{
    ScopedCache cache(8, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_rank_t  mA[2] = {100, 200};
    ucc_rank_t  mB[2] = {101, 201};

    ucc_team_t *tA    = insert_stub_dormant(cache, mA, 2, 0);
    ucc_team_t *tB    = insert_stub_dormant(cache, mB, 2, 0);
    ASSERT_NE(nullptr, tA);
    ASSERT_NE(nullptr, tB);

    ucc_spin_lock(&cache->lock);
    ucc_team_cache_get(tA);
    ucc_team_cache_registry_make_live(cache, tA);
    ucc_team_cache_get(tB);
    ucc_team_cache_registry_make_live(cache, tB);

    EXPECT_EQ(nullptr, ucc_team_cache_pick_victim(cache));

    ucc_team_cache_put(tA);
    ucc_team_cache_registry_make_dormant(cache, tA);
    ucc_team_cache_put(tB);
    ucc_team_cache_registry_make_dormant(cache, tB);

    EXPECT_NE(nullptr, ucc_team_cache_pick_victim(cache));
    ucc_spin_unlock(&cache->lock);

    erase_and_free(cache, {tA, tB});
}

/* Linear-check knob: disable_linear_check controls whether lookup runs the exact
   rank-array compare after a hash match.  Collisions are injected by overwriting
   the lookup key's hash. */

/* disable=0 (safe): the exact compare rejects a collision as MISS.
   disable=1 (trust-hash): the compare is skipped and the hash match returns. */
UCC_TEST_F(test_team_cache, linear_check_on_rejects_collision)
{
    auto run_collision = [](int disable_linear_check, bool expect_hit) {
        SCOPED_TRACE(
            disable_linear_check ? "disable_linear_check=1 (trust-hash)"
                                 : "disable_linear_check=0 (safe)");
        ScopedCache cache(
            16, UCC_TEAM_CACHE_EVICTION_FIFO, disable_linear_check);

        ucc_rank_t  arrA[3] = {1, 2, 3};
        ucc_team_t *teamA   = alloc_stub_team();
        ASSERT_NE(nullptr, teamA);
        ucc_team_params_t pA = make_array_params(arrA, 3, 0);
        ASSERT_EQ(
            UCC_OK, ucc_team_cache_identity_build(&pA, &teamA->cache_identity));

        ucc_spin_lock(&cache->lock);
        ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, teamA));
        ucc_spin_unlock(&cache->lock);
        EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, teamA->cache_state);

        /* Lookup key {7,8,9}; hash overridden to teamA's to force a collision. */
        ucc_rank_t                arrB[3] = {7, 8, 9};
        ucc_team_params_t         pB      = make_array_params(arrB, 3, 0);
        ucc_team_cache_identity_t keyB;
        build_identity(pB, keyB);

        ASSERT_NE(teamA->cache_identity.hash, keyB.hash);
        keyB.hash = teamA->cache_identity.hash;

        ucc_spin_lock(&cache->lock);
        ucc_team_t *found = ucc_team_cache_lookup(cache, &keyB);
        ucc_spin_unlock(&cache->lock);

        if (expect_hit) {
            EXPECT_EQ(teamA, found);
            EXPECT_EQ(1u, cache->stats.hits);
            EXPECT_EQ(0u, cache->stats.misses);
        } else {
            EXPECT_EQ(nullptr, found);
            EXPECT_EQ(1u, cache->stats.misses);
            EXPECT_EQ(0u, cache->stats.hits);
        }

        ucc_team_cache_identity_free(&keyB);
        erase_and_free(cache, {teamA});
    };

    run_collision(/*disable_linear_check=*/0, /*expect_hit=*/false);
    run_collision(/*disable_linear_check=*/1, /*expect_hit=*/true);
}

/* Trust-hash mode skips the membership compare but NOT the ext_id compare, so a
   dormant team under one ext_id is not re-adopted for a different ext_id. */
UCC_TEST_F(test_team_cache, linear_check_off_still_honors_ext_id)
{
    ScopedCache cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 1);

    ucc_rank_t  arr[3] = {1, 2, 3};
    ucc_team_t *teamA  = alloc_stub_team();
    ASSERT_NE(nullptr, teamA);
    ucc_team_params_t pA = make_array_id_params(arr, 3, 0, 7);
    ASSERT_EQ(
        UCC_OK, ucc_team_cache_identity_build(&pA, &teamA->cache_identity));

    ucc_spin_lock(&cache->lock);
    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, teamA));
    ucc_spin_unlock(&cache->lock);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, teamA->cache_state);

    /* Same membership, DIFFERENT external id -> same hash, different ext_id. */
    ucc_team_cache_identity_t keyB;
    build_id_key(arr, 3, 0, 8, keyB);

    ASSERT_EQ(teamA->cache_identity.hash, keyB.hash);
    ASSERT_NE(teamA->cache_identity.ext_id, keyB.ext_id);

    ucc_spin_lock(&cache->lock);
    ucc_team_t *found = ucc_team_cache_lookup(cache, &keyB);
    ucc_spin_unlock(&cache->lock);

    EXPECT_EQ(nullptr, found)
        << "trust-hash mode must still reject a differing ext_id";
    EXPECT_EQ(1u, cache->stats.misses);
    EXPECT_EQ(0u, cache->stats.hits);

    ucc_team_cache_identity_free(&keyB);
    erase_and_free(cache, {teamA});
}

/* All counters start zeroed; insert/hit/miss/hit-after-put accumulate them as
   expected.  Also folds the insert->DORMANT + size==1 postcondition. */
UCC_TEST_F(test_team_cache, stats_accumulate_correctly)
{
    ScopedCache cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    EXPECT_EQ(0u, cache->stats.lookups);
    EXPECT_EQ(0u, cache->stats.hits);
    EXPECT_EQ(0u, cache->stats.misses);
    EXPECT_EQ(0u, cache->stats.inserts);
    EXPECT_EQ(0u, cache->stats.evictions);

    ucc_rank_t        arr1[2] = {0, 1};
    ucc_rank_t        arr2[3] = {0, 1, 2};
    ucc_team_params_t p1      = make_array_params(arr1, 2, 0);
    ucc_team_params_t p2      = make_array_params(arr2, 3, 0);

    ucc_team_t       *t1      = alloc_stub_team();
    ucc_team_t       *t2      = alloc_stub_team();
    ASSERT_NE(nullptr, t1);
    ASSERT_NE(nullptr, t2);

    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p1, &t1->cache_identity));
    ASSERT_EQ(UCC_OK, ucc_team_cache_identity_build(&p2, &t2->cache_identity));

    ucc_team_cache_identity_t k1, k2;
    build_identity(p1, k1);
    build_identity(p2, k2);

    ucc_spin_lock(&cache->lock);

    /* Insert t1: DORMANT, size 1, one insert, no lookup counted. */
    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t1));
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, t1->cache_state);
    EXPECT_EQ(1u, cache->size);
    EXPECT_EQ(1u, cache->stats.inserts);
    EXPECT_EQ(0u, cache->stats.lookups);

    /* Lookup t1 (hit). */
    EXPECT_EQ(t1, ucc_team_cache_lookup(cache, &k1));
    EXPECT_EQ(1u, cache->stats.lookups);
    EXPECT_EQ(1u, cache->stats.hits);
    EXPECT_EQ(0u, cache->stats.misses);

    /* Lookup t2 identity (miss, not inserted). */
    EXPECT_EQ(nullptr, ucc_team_cache_lookup(cache, &k2));
    EXPECT_EQ(2u, cache->stats.lookups);
    EXPECT_EQ(1u, cache->stats.hits);
    EXPECT_EQ(1u, cache->stats.misses);

    /* Insert t2. */
    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t2));
    EXPECT_EQ(2u, cache->stats.inserts);

    /* Lookup t1 again (another hit). */
    EXPECT_EQ(t1, ucc_team_cache_lookup(cache, &k1));
    EXPECT_EQ(3u, cache->stats.lookups);
    EXPECT_EQ(2u, cache->stats.hits);
    EXPECT_EQ(1u, cache->stats.misses);

    ucc_spin_unlock(&cache->lock);

    ucc_team_cache_identity_free(&k1);
    ucc_team_cache_identity_free(&k2);
    erase_and_free(cache, {t1, t2});
}

/* dump_stats executes without fault (indirectly validates the format string),
   including the zero-lookups divide-by-zero guard and the NULL no-op. */
UCC_TEST_F(test_team_cache, dump_stats_no_crash)
{
    ScopedCache cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_spin_lock(&cache->lock);
    cache->stats.lookups   = 100;
    cache->stats.hits      = 80;
    cache->stats.misses    = 20;
    cache->stats.inserts   = 50;
    cache->stats.evictions = 10;
    ucc_spin_unlock(&cache->lock);
    ucc_team_cache_dump_stats(cache);

    ucc_spin_lock(&cache->lock);
    cache->stats.lookups = 0;
    ucc_spin_unlock(&cache->lock);
    ucc_team_cache_dump_stats(cache);

    ucc_team_cache_dump_stats(nullptr);
}

/* Cache-concurrency stress: overlapping create_post on the same context is
   invalid; cache->lock guards concurrent DESTROY/LOOKUP/CREATE on different
   contexts. These tests drive the locked cache API directly from std::threads. */
static bool cache_concurrency_runnable(void)
{
    unsigned hw = std::thread::hardware_concurrency();
    return (hw == 0) || (hw >= 2);
}

/* Build @n stub teams with pairwise-distinct membership and refcount 0. */
static void build_distinct_stub_teams(
    std::vector<ucc_team_t *>            &teams,
    std::vector<std::vector<ucc_rank_t>> &members, int n, int size_mod)
{
    for (int i = 0; i < n; i++) {
        int sz = 2 + (i % size_mod);
        members[i].resize(sz);
        for (int j = 0; j < sz; j++) {
            members[i][j] = (ucc_rank_t)j;
        }
        ucc_team_params_t p = make_array_params(
            members[i].data(), (ucc_rank_t)sz, 0);

        teams[i] = alloc_stub_team();
        ASSERT_NE(nullptr, teams[i]);
        teams[i]->refcount = 0;
        ASSERT_EQ(
            UCC_OK,
            ucc_team_cache_identity_build(&p, &teams[i]->cache_identity));
    }
}

/* Erase each resident stub under the lock and free it - concurrency teardown. */
static void drain_stub_teams(
    ucc_team_cache_t *cache, std::vector<ucc_team_t *> &teams)
{
    for (auto *t : teams) {
        ucc_spin_lock(&cache->lock);
        erase_stub(cache, t);
        ucc_spin_unlock(&cache->lock);
        free_stub_team(t);
    }
}

/* Contended DESTROY + LOOKUP path.  A pool of DORMANT stub teams; N threads
   race, each iteration under cache->lock: lookup, adopt on a DORMANT hit, then
   release. */
UCC_TEST_F(test_team_cache, concurrent_lookup_adopt_release_stress)
{
    if (!cache_concurrency_runnable()) {
        GTEST_SKIP() << "host lacks >= 2 concurrent threads for cache stress";
    }

    const char *iters_env   = std::getenv("UCC_GTEST_CACHE_STRESS_ITERS");
    const char *threads_env = std::getenv("UCC_GTEST_CACHE_STRESS_THREADS");
    const int   n_iters     = iters_env ? std::atoi(iters_env) : 500;
    const int   n_threads   = threads_env ? std::atoi(threads_env) : 8;
    const int   n_teams     = 16;

    ScopedCache cache(64, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    std::vector<ucc_team_t *>              teams(n_teams);
    std::vector<ucc_team_cache_identity_t> keys(n_teams);
    std::vector<std::vector<ucc_rank_t>>   members(n_teams);
    std::vector<std::atomic<bool>>         owned(n_teams);

    build_distinct_stub_teams(teams, members, n_teams, /*size_mod=*/n_teams);
    for (int i = 0; i < n_teams; i++) {
        ucc_spin_lock(&cache->lock);
        ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, teams[i]));
        ucc_spin_unlock(&cache->lock);
        ASSERT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, teams[i]->cache_state);

        ucc_team_params_t p = make_array_params(
            members[i].data(), (ucc_rank_t)members[i].size(), 0);
        build_identity(p, keys[i]);
        owned[i].store(false);
    }

    const uint32_t    size_before = cache->size;

    std::atomic<bool> double_adopt{false};
    std::atomic<bool> bad_refcount{false};

    auto              worker = [&](int seed) {
        std::mt19937 rng((unsigned)(seed * 2654435761u + 1));
        for (int it = 0; it < n_iters; it++) {
            int idx = rng() % n_teams;

            /* Adopt phase: lookup -> get -> make_live, all under the lock. */
            ucc_spin_lock(&cache->lock);
            ucc_team_t *t = ucc_team_cache_lookup(cache, &keys[idx]);
            if (t != nullptr) {
                if (t->cache_state != UCC_TEAM_CACHE_STATE_DORMANT ||
                    t->refcount != 0) {
                    bad_refcount.store(true);
                }
                ucc_team_cache_get(t);
                ucc_team_cache_registry_make_live(cache, t);
            }
            ucc_spin_unlock(&cache->lock);

            if (t == nullptr) {
                continue; /* another thread holds it live: legal miss */
            }

            /* Only one thread may hold this team live at a time. */
            if (owned[idx].exchange(true)) {
                double_adopt.store(true);
            }
            std::this_thread::yield();
            owned[idx].store(false);

            /* Release phase: put -> make_dormant on last drop. */
            ucc_spin_lock(&cache->lock);
            int rc = ucc_team_cache_put(t);
            if (rc < 0) {
                bad_refcount.store(true);
            }
            if (rc == 0) {
                ucc_team_cache_registry_make_dormant(cache, t);
            }
            ucc_spin_unlock(&cache->lock);
        }
    };

    std::vector<std::thread> pool;
    for (int i = 0; i < n_threads; i++) {
        pool.emplace_back(worker, i);
    }
    for (auto &th : pool) {
        th.join();
    }

    EXPECT_FALSE(double_adopt.load())
        << "a LIVE team was adopted by two threads concurrently";
    EXPECT_FALSE(bad_refcount.load())
        << "cache refcount/state invariant violated under concurrency";

    EXPECT_EQ(size_before, cache->size);
    for (int i = 0; i < n_teams; i++) {
        EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, teams[i]->cache_state)
            << "team " << i << " must settle back to DORMANT";
        EXPECT_EQ(0, teams[i]->refcount)
            << "team " << i << " refcount must settle to 0";
    }

    for (int i = 0; i < n_teams; i++) {
        ucc_team_cache_identity_free(&keys[i]);
    }
    drain_stub_teams(cache, teams);
}
