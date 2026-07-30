/**
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
extern "C" {
#include "core/ucc_team_cache.h"
#include "core/ucc_team.h"
#include "core/ucc_context.h"
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
   cacheability policy, the locked cache API, eviction, and the agreement vote,
   with no MPI job. */

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

/* ARRAY+OOB style (UCC_EP_MAP_ARRAY): caller-owned array, no closure. */
static ucc_team_params_t make_array_params(
    ucc_rank_t *arr, ucc_rank_t size, ucc_rank_t self_ep)
{
    ucc_team_params_t p;
    memset(&p, 0, sizeof(p));
    p.mask             = UCC_TEAM_PARAM_FIELD_EP_MAP | UCC_TEAM_PARAM_FIELD_EP;
    p.ep               = self_ep;
    p.ep_map.type      = UCC_EP_MAP_ARRAY;
    p.ep_map.ep_num    = size;
    p.ep_map.array.map = arr;
    p.ep_map.array.elem_size = sizeof(ucc_rank_t);
    return p;
}

static ucc_team_params_t make_strided_params(
    uint64_t start, int64_t stride, ucc_rank_t size, ucc_rank_t self_ep)
{
    ucc_team_params_t p;
    memset(&p, 0, sizeof(p));
    p.mask          = UCC_TEAM_PARAM_FIELD_EP_MAP | UCC_TEAM_PARAM_FIELD_EP;
    p.ep            = self_ep;
    p.ep_map.type   = UCC_EP_MAP_STRIDED;
    p.ep_map.ep_num = size;
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

/* Build a lookup key from ARRAY membership + external id. */
static void build_id_key(
    ucc_rank_t *arr, ucc_rank_t size, ucc_rank_t self_ep, uint64_t id,
    ucc_team_cache_identity_t &key)
{
    ucc_team_params_t p = make_array_id_params(arr, size, self_ep, id);
    build_identity(p, key);
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

/* Save/restore an environment variable across a scope. */
struct ScopedEnv {
    std::string name;
    std::string saved;
    bool        had;

    ScopedEnv(const char *n) : name(n)
    {
        const char *v = getenv(n);
        had           = (v != nullptr);
        if (had) {
            saved = v;
        }
    }
    ~ScopedEnv()
    {
        if (had) {
            setenv(name.c_str(), saved.c_str(), 1);
        } else {
            unsetenv(name.c_str());
        }
    }
};

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

    /* Membership-only hash: all three share a khash bucket. */
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

    /* Cross-style: EP_MAP CB vs EP_MAP_ARRAY — same membership, different map type. */
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
    arr[0]          = 11;
    arr[1]          = 13;
    arr[2]          = 17;
    arr[3]          = 19;

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

/* Agreement vote: a UCC_OP_BAND allreduce over the members' fill buffers.  These
   tests reduce N per-rank buffers with a local BAND and check vote_result +
   vote_new_cookie without any network. */
static void vote_band_reduce(
    const std::vector<std::vector<uint64_t>> &in, uint64_t *out)
{
    for (int l = 0; l < UCC_TEAM_CACHE_VOTE_LANES; l++) {
        out[l] = ~(uint64_t)0;
    }
    for (const auto &v : in) {
        for (int l = 0; l < UCC_TEAM_CACHE_VOTE_LANES; l++) {
            out[l] &= v[l];
        }
    }
}

/* All-hit agreement: RESEAT_DERIVED with a shared candidate cookie distributes
   rank-0's new cookie; EXACT_REUSE ignores the cookie lane and still agrees. */
UCC_TEST_F(test_team_cache, vote_agreement_and_cookie)
{
    uint64_t out[UCC_TEAM_CACHE_VOTE_LANES];

    /* RESEAT_DERIVED, same cookie on every rank -> agree, cookie distributed. */
    {
        const int                          n = 4;
        std::vector<std::vector<uint64_t>> in(
            n, std::vector<uint64_t>(UCC_TEAM_CACHE_VOTE_LANES));
        for (int r = 0; r < n; r++) {
            ucc_team_cache_vote_fill(
                in[r].data(),
                /*prepared=*/1,
                UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED,
                /*key=*/0x1234,
                /*cookie=*/0xABCDEF,
                /*parent_cookie=*/0x77,
                /*is_rank0=*/(r == 0),
                /*proposed_cookie=*/0xFEED);
        }
        vote_band_reduce(in, out);
        EXPECT_EQ(
            UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED,
            ucc_team_cache_vote_result(out));
        EXPECT_EQ((uint64_t)0xFEED, ucc_team_cache_vote_new_cookie(out));
    }

    /* EXACT_REUSE with cookie=0 everywhere still agrees (ext_id pins instance). */
    {
        std::vector<std::vector<uint64_t>> in(
            3, std::vector<uint64_t>(UCC_TEAM_CACHE_VOTE_LANES));
        for (int r = 0; r < 3; r++) {
            ucc_team_cache_vote_fill(
                in[r].data(),
                1,
                UCC_TEAM_CACHE_ACTION_EXACT_REUSE,
                0x55,
                /*cookie=*/0,
                /*parent_cookie=*/0,
                /*is_rank0=*/(r == 0),
                0x1);
        }
        vote_band_reduce(in, out);
        EXPECT_EQ(
            UCC_TEAM_CACHE_ACTION_EXACT_REUSE, ucc_team_cache_vote_result(out));
    }
}

/* Two members hold DIFFERENT candidate cookies for the same membership -> the
   cookie equality lane breaks -> global MISS, so no false reuse. */
UCC_TEST_F(test_team_cache, vote_reseat_different_cookie_misses)
{
    std::vector<std::vector<uint64_t>> in(
        2, std::vector<uint64_t>(UCC_TEAM_CACHE_VOTE_LANES));
    uint64_t out[UCC_TEAM_CACHE_VOTE_LANES];

    ucc_team_cache_vote_fill(
        in[0].data(),
        1,
        UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED,
        0x1234,
        /*cookie=*/0xAAAA,
        0x77,
        /*is_rank0=*/1,
        0xFEED);
    ucc_team_cache_vote_fill(
        in[1].data(),
        1,
        UCC_TEAM_CACHE_ACTION_RESEAT_DERIVED,
        0x1234,
        /*cookie=*/0xBBBB,
        0x77,
        /*is_rank0=*/0,
        0);
    vote_band_reduce(in, out);
    EXPECT_EQ(UCC_TEAM_CACHE_ACTION_MISS, ucc_team_cache_vote_result(out));
}

/* A single miss rank (prepared=0) forces a global MISS even if every other rank
   agrees, and rank-0's new-cookie proposal still survives for the fresh build. */
UCC_TEST_F(test_team_cache, vote_one_miss_rank_forces_miss_keeps_cookie)
{
    std::vector<std::vector<uint64_t>> in(
        3, std::vector<uint64_t>(UCC_TEAM_CACHE_VOTE_LANES));
    uint64_t out[UCC_TEAM_CACHE_VOTE_LANES];

    ucc_team_cache_vote_fill(
        in[0].data(),
        1,
        UCC_TEAM_CACHE_ACTION_EXACT_REUSE,
        0x1234,
        0,
        0,
        /*is_rank0=*/1,
        /*proposed_cookie=*/0x900D);
    ucc_team_cache_vote_fill(
        in[1].data(),
        1,
        UCC_TEAM_CACHE_ACTION_EXACT_REUSE,
        0x1234,
        0,
        0,
        /*is_rank0=*/0,
        0);
    ucc_team_cache_vote_fill(
        in[2].data(),
        /*prepared=*/0,
        UCC_TEAM_CACHE_ACTION_MISS,
        0,
        0,
        0,
        /*is_rank0=*/0,
        0);
    vote_band_reduce(in, out);
    EXPECT_EQ(UCC_TEAM_CACHE_ACTION_MISS, ucc_team_cache_vote_result(out));
    EXPECT_EQ((uint64_t)0x900D, ucc_team_cache_vote_new_cookie(out));
}

/* next_cookie is strictly monotonic and never returns 0. */
UCC_TEST_F(test_team_cache, next_cookie_monotonic_nonzero)
{
    ucc_team_cache_t c;
    memset(&c, 0, sizeof(c));
    c.cache_gen = 0;
    uint64_t a  = ucc_team_cache_next_cookie(&c);
    uint64_t b  = ucc_team_cache_next_cookie(&c);
    EXPECT_NE((uint64_t)0, a);
    EXPECT_NE((uint64_t)0, b);
    EXPECT_LT(a, b);
}

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

    ucc_team_cache_registry_remove(cache, &team);
    EXPECT_FALSE(team_on_list(&cache->live, &team));
    EXPECT_FALSE(team_on_list(&cache->dormant, &team));
    EXPECT_TRUE(ucc_list_is_empty(&cache->live));
    EXPECT_TRUE(ucc_list_is_empty(&cache->dormant));
}

/* Cache API tests (lookup / insert / get / put) use bare stub teams with only
   the cache-related fields initialized; they never call create_post. */

/* Allocate and minimally initialize a stub team. refcount starts at 0 to match
   the production convention for a cached DORMANT team; a test adopts it with
   ucc_team_cache_get() (0 -> 1, LIVE) before releasing it with _put(). */
static ucc_team_t *alloc_stub_team(void)
{
    ucc_team_t *t = (ucc_team_t *)calloc(1, sizeof(*t));
    if (!t) {
        return nullptr;
    }
    t->refcount    = 0; /* DORMANT convention (calloc already zeroes this) */
    t->cache_state = UCC_TEAM_CACHE_STATE_NONE;
    ucc_list_head_init(&t->cache_link);
    ucc_list_head_init(&t->bucket_link);
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
    ucc_team_cache_registry_remove(cache, t);
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
    erase_and_free(cache, {team});
}

/* Full-cache skip: second insert into a full max_size==1 cache stays NONE. */
UCC_TEST_F(test_team_cache, full_cache_skip)
{
    ScopedCache       cache(1, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

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

    erase_and_free(cache, {t1}); /* t1 was inserted (DORMANT); remove before free */
    free_stub_team(t2);          /* t2 was not inserted (NONE); free directly */
}

/* get/put refcount arithmetic and LIVE<->DORMANT transitions, using the
   production convention (a cached DORMANT team has refcount 0):
   insert->DORMANT rc=0; get->LIVE rc=1; put->DORMANT rc=0; then a two-user cycle
   get,get->rc=2; put->LIVE rc=1; put->DORMANT rc=0. */
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

/* Team-id pool bit helpers must be exact inverses at word boundaries: set one
   boundary id, scan it back out, and the pool must be clear again.  (The
   historical bug used id/64 for the release word, leaking multiples of 64.) */
UCC_TEST_F(test_team_cache, id_pool_bit_boundaries)
{
    const int ids[] = {1, 63, 64, 65, 127, 128, 129, 191, 192};

    for (int id : ids) {
        uint64_t pool[4] = {0, 0, 0, 0}; /* covers ids 1..256 */

        ucc_team_id_pool_set_bit(pool, id);

        int set_words = 0;
        for (int w = 0; w < 4; w++) {
            if (pool[w]) {
                set_words++;
            }
        }
        EXPECT_EQ(1, set_words) << "id " << id << " set bits in multiple words";

        int found = 0, pos = 0;
        for (int w = 0; w < 4; w++) {
            if ((pos = ucc_team_id_pool_ffs_clear(&pool[w])) > 0) {
                found = w * 64 + pos;
                break;
            }
        }
        EXPECT_EQ(id, found)
            << "released id " << id << " re-scanned as " << found;

        for (int w = 0; w < 4; w++) {
            EXPECT_EQ((uint64_t)0, pool[w])
                << "id " << id << " left residue in word " << w;
        }
    }
}

/* Eviction correctness.  White-box tests drive pick_lru_victim directly under
   cache->lock; integration tests use the real EP_MAP create path. */

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

/* RESERVED state (the agreement-vote pin): a DORMANT candidate moved to RESERVED
   is off the dormant/live lists (lookup can't return it) but stays in the bucket
   with refcount unchanged, so a vote-FAIL rolls it back to DORMANT and a vote-PASS
   promotes it to LIVE via get (0 -> 1). */
UCC_TEST_F(test_team_cache, reserved_state_pin_and_rollback)
{
    ScopedCache       cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);
    ucc_rank_t        arr[3] = {10, 20, 30};
    ucc_team_params_t p      = make_array_params(arr, 3, 0);

    ucc_team_t *t = insert_stub_dormant(cache, arr, 3, 0); /* DORMANT rc=0 */
    ASSERT_NE(nullptr, t);

    ucc_team_cache_identity_t key;
    build_identity(p, key);

    ucc_spin_lock(&cache->lock);
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

/* Victim selection across policies: FIFO returns the insertion head and ignores
   seq_num; LFU and its LRU alias return the least-used team (min seq_num) and
   break ties by dormant-list order (earliest wins). */
UCC_TEST_F(test_team_cache, evict_victim_selection)
{
    const struct {
        ucc_team_cache_eviction_policy_t policy;
        const char                      *name;
    } cases[] = {
        {UCC_TEAM_CACHE_EVICTION_FIFO, "fifo"},
        {UCC_TEAM_CACHE_EVICTION_LFU, "lfu"},
        {UCC_TEAM_CACHE_EVICTION_LRU, "lru"},
    };

    for (auto &c : cases) {
        SCOPED_TRACE(c.name);
        ScopedCache cache(8, c.policy, 0);

        ucc_team_t *tA, *tB, *tC;
        insert_three_dormant(cache, &tA, &tB, &tC);
        ASSERT_EQ(3u, cache->size);

        /* B is least-used. */
        tA->seq_num = 10;
        tB->seq_num = 3;
        tC->seq_num = 20;

        ucc_spin_lock(&cache->lock);
        ucc_team_t *victim = ucc_team_cache_pick_lru_victim(cache);
        ucc_spin_unlock(&cache->lock);

        ucc_team_t *expect = (c.policy == UCC_TEAM_CACHE_EVICTION_FIFO) ? tA
                                                                        : tB;
        EXPECT_EQ(expect, victim);

        /* Tie on the minimum: the usage policies pick the earliest dormant
           entry (A); FIFO still returns the head. */
        tA->seq_num = 3;
        tB->seq_num = 3;
        ucc_spin_lock(&cache->lock);
        victim = ucc_team_cache_pick_lru_victim(cache);
        ucc_spin_unlock(&cache->lock);
        EXPECT_EQ(tA, victim);

        erase_and_free(cache, {tA, tB, tC});
    }
}

/* pick_lru_victim skips LIVE teams: adopt both -> dormant empty -> NULL;
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

    EXPECT_EQ(nullptr, ucc_team_cache_pick_lru_victim(cache));

    ucc_team_cache_put(tA);
    ucc_team_cache_registry_make_dormant(cache, tA);
    ucc_team_cache_put(tB);
    ucc_team_cache_registry_make_dormant(cache, tB);

    EXPECT_NE(nullptr, ucc_team_cache_pick_lru_victim(cache));
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
UCC_TEST_F(test_team_cache, dump_stats_prints_all_fields)
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
   contexts.  These tests drive the locked cache API directly from std::threads.
   Gated on >= 2 OS threads (the cache spinlock is compiled unconditionally). */
static bool cache_concurrency_runnable(void)
{
    unsigned hw = std::thread::hardware_concurrency();
    return (hw == 0) || (hw >= 2);
}

/* Build @n stub teams with pairwise-distinct membership (rank[j] = i*@stride + j,
   size 2 + i % @size_mod) and refcount 0.  Identities are built but the teams are
   not inserted.  members[] backs the identities and must outlive the teams. */
static void build_distinct_stub_teams(
    std::vector<ucc_team_t *>            &teams,
    std::vector<std::vector<ucc_rank_t>> &members, int n, int stride,
    int size_mod)
{
    for (int i = 0; i < n; i++) {
        int sz = 2 + (i % size_mod);
        members[i].resize(sz);
        for (int j = 0; j < sz; j++) {
            members[i][j] = (ucc_rank_t)(i * stride + j);
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
   race, each iteration under cache->lock: lookup, adopt on a DORMANT hit
   (get + make_live), then release (put + make_dormant on last drop).  A LIVE team
   is never handed out twice (tracked via a per-team owner flag).  After join:
   every team DORMANT, refcount 0, cache->size unchanged. */
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

    build_distinct_stub_teams(
        teams,
        members,
        n_teams,
        /*stride=*/0,
        /*size_mod=*/n_teams);
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

/* Chained-bucket invariants and derived-team lookups (stub teams + identity
   injection). */

/* Allocate a stub team with @arr membership and external id @ext_id. */
static ucc_team_t *alloc_stub_team_with_id(
    ucc_rank_t *arr, ucc_rank_t n, ucc_rank_t self_ep, uint64_t ext_id)
{
    ucc_team_params_t p = make_array_id_params(arr, n, self_ep, ext_id);

    ucc_team_t       *t = alloc_stub_team();
    if (!t) {
        return nullptr;
    }
    if (UCC_OK != ucc_team_cache_identity_build(&p, &t->cache_identity)) {
        free(t);
        return nullptr;
    }
    return t;
}

/* Allocate an is_derived stub team with @arr membership and external id @ext_id. */
static ucc_team_t *alloc_stub_derived_team(
    ucc_rank_t *arr, ucc_rank_t n, ucc_rank_t self_ep, uint64_t ext_id)
{
    ucc_team_t *t = alloc_stub_team_with_id(arr, n, self_ep, ext_id);
    if (t) {
        t->is_derived = 1;
    }
    return t;
}

/* Same membership, different ext_ids chain in one membership-only bucket: each
   insert increments size (not de-duped), full-identity lookup finds each by
   ext_id, and head + sibling erase collapse the chain and drop size correctly. */
UCC_TEST_F(test_team_cache, chained_bucket_same_membership_different_ext_id)
{
    ScopedCache cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_rank_t  arr[3] = {1, 2, 3};

    ucc_team_t *t10    = alloc_stub_team_with_id(arr, 3, 0, 10);
    ucc_team_t *t20    = alloc_stub_team_with_id(arr, 3, 0, 20);
    ucc_team_t *t30    = alloc_stub_team_with_id(arr, 3, 0, 30);
    ASSERT_NE(nullptr, t10);
    ASSERT_NE(nullptr, t20);
    ASSERT_NE(nullptr, t30);

    ASSERT_EQ(t10->cache_identity.hash, t20->cache_identity.hash);
    ASSERT_EQ(t10->cache_identity.hash, t30->cache_identity.hash);
    ASSERT_NE(t10->cache_identity.ext_id, t20->cache_identity.ext_id);
    ASSERT_NE(t10->cache_identity.ext_id, t30->cache_identity.ext_id);

    ucc_spin_lock(&cache->lock);

    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t10));
    EXPECT_EQ(1u, cache->size);
    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t20));
    EXPECT_EQ(2u, cache->size);
    EXPECT_EQ(UCC_OK, ucc_team_cache_insert(cache, t30));
    EXPECT_EQ(3u, cache->size);

    ucc_team_cache_identity_t key10, key20, key30;
    build_id_key(arr, 3, 0, 10, key10);
    build_id_key(arr, 3, 0, 20, key20);
    build_id_key(arr, 3, 0, 30, key30);

    /* Each is independently reachable, and repeated lookups are side-effect free. */
    EXPECT_EQ(t10, ucc_team_cache_lookup(cache, &key10));
    EXPECT_EQ(t20, ucc_team_cache_lookup(cache, &key20));
    EXPECT_EQ(t30, ucc_team_cache_lookup(cache, &key30));
    EXPECT_EQ(t10, ucc_team_cache_lookup(cache, &key10));
    EXPECT_EQ(t30, ucc_team_cache_lookup(cache, &key30));

    /* Erase the chain head (t10): t20 is promoted; size drops by one. */
    ucc_team_cache_table_erase(cache, t10);
    EXPECT_EQ(2u, cache->size);
    EXPECT_EQ(nullptr, ucc_team_cache_lookup(cache, &key10));
    EXPECT_EQ(t20, ucc_team_cache_lookup(cache, &key20));
    EXPECT_EQ(t30, ucc_team_cache_lookup(cache, &key30));

    /* Erase a non-head sibling (t30). */
    ucc_team_cache_table_erase(cache, t30);
    EXPECT_EQ(1u, cache->size);
    EXPECT_EQ(nullptr, ucc_team_cache_lookup(cache, &key30));
    EXPECT_EQ(t20, ucc_team_cache_lookup(cache, &key20));

    /* Erase the last entry (t20): bucket removed. */
    ucc_team_cache_table_erase(cache, t20);
    EXPECT_EQ(0u, cache->size);
    EXPECT_EQ(nullptr, ucc_team_cache_lookup(cache, &key20));

    ucc_spin_unlock(&cache->lock);

    ucc_team_cache_identity_free(&key10);
    ucc_team_cache_identity_free(&key20);
    ucc_team_cache_identity_free(&key30);
    free_stub_team(t10);
    free_stub_team(t20);
    free_stub_team(t30);
}

/* lookup_live returns a LIVE same-membership sibling (skipping the dormant one),
   and returns NULL once no live sibling remains (the parent-gone guard). */
UCC_TEST_F(test_team_cache, lookup_live_returns_live_sibling)
{
    ScopedCache cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_rank_t  arr[3]    = {10, 20, 30};

    ucc_team_t *t_dormant = alloc_stub_team_with_id(arr, 3, 0, 10);
    ucc_team_t *t_live    = alloc_stub_team_with_id(arr, 3, 0, 20);
    ASSERT_NE(nullptr, t_dormant);
    ASSERT_NE(nullptr, t_live);
    ASSERT_EQ(t_dormant->cache_identity.hash, t_live->cache_identity.hash);

    ucc_spin_lock(&cache->lock);

    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, t_dormant));
    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, t_live));
    EXPECT_EQ(2u, cache->size);

    ucc_team_cache_get(t_live);
    ucc_team_cache_registry_make_live(cache, t_live);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, t_live->cache_state);

    ucc_team_cache_identity_t key;
    build_id_key(arr, 3, 0, 10, key);

    /* Full-identity lookup returns the DORMANT team; lookup_live returns the
       LIVE sibling, skipping the dormant one. */
    EXPECT_EQ(t_dormant, ucc_team_cache_lookup(cache, &key));
    EXPECT_EQ(t_live, ucc_team_cache_lookup_live(cache, &key));

    /* Release the live sibling to dormant: no live sibling -> lookup_live NULL. */
    ucc_team_cache_put(t_live);
    ucc_team_cache_registry_make_dormant(cache, t_live);
    EXPECT_EQ(nullptr, ucc_team_cache_lookup_live(cache, &key));

    erase_stub(cache, t_dormant);
    erase_stub(cache, t_live);
    ucc_spin_unlock(&cache->lock);

    ucc_team_cache_identity_free(&key);
    free_stub_team(t_dormant);
    free_stub_team(t_live);
}

/* lookup_dormant_derived returns the FIRST dormant derived team in chain
   (insertion) order, skipping LIVE derived teams and non-derived base teams, and
   advances deterministically as heads are erased, then returns NULL when empty. */
UCC_TEST_F(test_team_cache, lookup_dormant_derived_selection)
{
    ScopedCache cache(16, UCC_TEAM_CACHE_EVICTION_FIFO, 0);

    ucc_rank_t  arr[4]         = {10, 20, 30, 40};

    /* Non-derived dormant base (ineligible), LIVE derived (ineligible), and two
       DORMANT derived teams that are the valid targets, in insertion order. */
    ucc_team_t *t_base         = alloc_stub_team_with_id(arr, 4, 0, 10);
    ucc_team_t *t_live_derived = alloc_stub_derived_team(arr, 4, 0, 20);
    ucc_team_t *t_d1           = alloc_stub_derived_team(arr, 4, 0, 30);
    ucc_team_t *t_d2           = alloc_stub_derived_team(arr, 4, 0, 40);
    ASSERT_NE(nullptr, t_base);
    ASSERT_NE(nullptr, t_live_derived);
    ASSERT_NE(nullptr, t_d1);
    ASSERT_NE(nullptr, t_d2);
    t_base->is_derived = 0;

    ASSERT_EQ(t_base->cache_identity.hash, t_live_derived->cache_identity.hash);
    ASSERT_EQ(t_base->cache_identity.hash, t_d1->cache_identity.hash);
    ASSERT_EQ(t_base->cache_identity.hash, t_d2->cache_identity.hash);

    ucc_spin_lock(&cache->lock);

    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, t_base));
    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, t_live_derived));
    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, t_d1));
    ASSERT_EQ(UCC_OK, ucc_team_cache_insert(cache, t_d2));
    EXPECT_EQ(4u, cache->size);

    ucc_team_cache_get(t_live_derived);
    ucc_team_cache_registry_make_live(cache, t_live_derived);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, t_live_derived->cache_state);

    /* Drifted key: same membership, ext_id=99 (no exact match). */
    ucc_team_cache_identity_t key;
    build_id_key(arr, 4, 0, 99, key);

    /* Exact lookup misses (membership matches but full identity does not). */
    EXPECT_EQ(nullptr, ucc_team_cache_lookup(cache, &key));

    /* Only the DORMANT derived teams are eligible; the first in chain order is
       returned, skipping the LIVE derived and the non-derived base. */
    ucc_team_t *found = ucc_team_cache_lookup_dormant_derived(cache, &key);
    EXPECT_EQ(t_d1, found);
    ASSERT_NE(nullptr, found);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, found->cache_state);
    EXPECT_EQ(1, found->is_derived);
    EXPECT_NE(t_live_derived, found);
    EXPECT_NE(t_base, found);

    /* Idempotent without mutation. */
    EXPECT_EQ(t_d1, ucc_team_cache_lookup_dormant_derived(cache, &key));

    /* Consuming the head advances to the next dormant derived, then NULL. */
    erase_stub(cache, t_d1);
    EXPECT_EQ(t_d2, ucc_team_cache_lookup_dormant_derived(cache, &key));
    erase_stub(cache, t_d2);
    EXPECT_EQ(nullptr, ucc_team_cache_lookup_dormant_derived(cache, &key))
        << "no eligible target: only a LIVE derived and a non-derived base "
           "remain";

    ucc_team_cache_put(t_live_derived);
    ucc_team_cache_registry_make_dormant(cache, t_live_derived);
    erase_stub(cache, t_base);
    erase_stub(cache, t_live_derived);
    ucc_spin_unlock(&cache->lock);

    ucc_team_cache_identity_free(&key);
    free_stub_team(t_base);
    free_stub_team(t_live_derived);
    free_stub_team(t_d1);
    free_stub_team(t_d2);
}

/* Integration tests: full create->use->destroy->recreate cycle through
   UccJob/UccTeam (real create_post / create_test / destroy).  White-box access
   to ucc_team_t.cache_state and ucc_context_t.team_cache asserts dormant-reuse
   invariants.  Caching is enabled per-test via the UccJob env-var mechanism. */
class test_team_cache_integration : public ucc::test {};

/* Return the underlying ucc_team_t* from a per-process team handle. */
static ucc_team_t *team_ptr(UccTeam_h &team, int proc_idx = 0)
{
    return (ucc_team_t *)team->procs[proc_idx].team;
}

/* Return the ucc_context_t* for the given process in a team. */
static ucc_context_t *ctx_ptr(UccTeam_h &team, int proc_idx = 0)
{
    return (ucc_context_t *)team->procs[proc_idx].p.get()->ctx_h;
}

/* Run a single barrier collective on a team and assert it completes. */
static void run_barrier(UccTeam_h &team)
{
    ucc_coll_args_t coll;
    coll.mask      = 0;
    coll.coll_type = UCC_COLL_TYPE_BARRIER;
    UccReq req(team, &coll);
    req.start();
    ASSERT_EQ(UCC_OK, req.wait());
}

/* create->barrier->destroy->recreate-identical must re-adopt the SAME
   ucc_team_t (pointer + team-id preserved) and record a hit on the second
   create, then drive a second lifetime cycle. */
UCC_TEST_F(test_team_cache_integration, dormant_reuse)
{
    UccJob job(
        4,
        UccJob::UCC_JOB_CTX_GLOBAL,
        {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y")});

    UccTeam_h   t1         = job.create_team(2, /*use_team_ep_map=*/true);

    ucc_team_t *tp0_before = team_ptr(t1, 0);
    ucc_team_t *tp1_before = team_ptr(t1, 1);
    uint16_t    id0_before = tp0_before->id;

    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, tp0_before->cache_state);

    ucc_context_t *ctx0 = ctx_ptr(t1, 0);
    ASSERT_NE(nullptr, ctx0->team_cache);

    /* First create: a miss, no hit, one insert. */
    EXPECT_GE(ctx0->team_cache->stats.lookups, 1u);
    EXPECT_EQ(0u, ctx0->team_cache->stats.hits);
    EXPECT_EQ(1u, ctx0->team_cache->stats.inserts);

    run_barrier(t1);

    t1.reset();
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, tp0_before->cache_state);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, tp1_before->cache_state);

    /* Second team, IDENTICAL membership -> re-adopt the dormant team. */
    UccTeam_h   t2        = job.create_team(2, /*use_team_ep_map=*/true);

    ucc_team_t *tp0_after = team_ptr(t2, 0);
    ucc_team_t *tp1_after = team_ptr(t2, 1);

    EXPECT_EQ(tp0_before, tp0_after) << "team not reused";
    EXPECT_EQ(tp1_before, tp1_after) << "team not reused";
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, tp0_after->cache_state);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, tp1_after->cache_state);
    EXPECT_EQ(id0_before, tp0_after->id)
        << "re-adopted team must retain its original team ID";

    /* Second create: one more lookup, exactly one hit, no extra insert. */
    EXPECT_GE(ctx0->team_cache->stats.lookups, 2u);
    EXPECT_EQ(1u, ctx0->team_cache->stats.hits);
    EXPECT_EQ(1u, ctx0->team_cache->stats.inserts);

    run_barrier(t2);

    t2.reset();
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, tp0_after->cache_state);
}

/* With UCC_TEAM_CACHE_ENABLE=n, create->destroy->recreate produces DISTINCT
   ucc_team_t pointers; a team without EP_MAP is never cacheable. */
UCC_TEST_F(test_team_cache_integration, knob_off)
{
    UccJob job(
        4,
        UccJob::UCC_JOB_CTX_GLOBAL,
        {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "n")});

    UccTeam_h      t1        = job.create_team(2, /*use_team_ep_map=*/true);
    ucc_team_t    *tp_before = team_ptr(t1, 0);

    ucc_context_t *ctx0      = ctx_ptr(t1, 0);
    EXPECT_EQ(nullptr, ctx0->team_cache)
        << "team_cache must be NULL when caching is disabled";

    run_barrier(t1);

    /* Create t2 while t1 is still live so pointer-distinctness is meaningful. */
    UccTeam_h   t2       = job.create_team(2, /*use_team_ep_map=*/true);
    ucc_team_t *tp_after = team_ptr(t2, 0);

    EXPECT_NE(tp_before, tp_after);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_NONE, tp_after->cache_state);

    run_barrier(t2);
    t1.reset();

    /* A team created WITHOUT EP_MAP is never cacheable (identity_build requires
       FIELD_EP_MAP): cache_state==NONE and distinct pointers. */
    UccTeam_h   n1  = job.create_team(2, /*use_team_ep_map=*/false);
    ucc_team_t *np1 = team_ptr(n1, 0);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_NONE, np1->cache_state);
    run_barrier(n1);

    UccTeam_h   n2  = job.create_team(2, /*use_team_ep_map=*/false);
    ucc_team_t *np2 = team_ptr(n2, 0);
    EXPECT_NE(np1, np2);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_NONE, np2->cache_state);
    run_barrier(n2);
    n1.reset();
}

/* Two simultaneously-LIVE identical-membership teams (parent + derived) with
   distinct team-ids run interleaved-order collectives.  A shared id would
   cross-match in the (team_id, src_rank, seq_num) tag space and produce a wrong
   result or hang; the derived team's own id keeps the streams isolated.
   gtest limitation: the in-process UCP path can't re-adopt >= 4-proc teams after
   dormancy, so these use <= 3-proc teams (the MPI leg covers the rest). */

/* One in-flight per-proc collective: request + owned buffers + expected. */
struct coll_op {
    ucc_coll_req_h req;
    ucc_context_h  ctx;
    int64_t        sbuf;
    int64_t        rbuf;
    int64_t        expect;
    bool           is_allreduce;
};

static void set_int64_host_buf(ucc_coll_buffer_info_t &info, int64_t *buf)
{
    info.buffer   = buf;
    info.count    = 1;
    info.datatype = UCC_DT_INT64;
    info.mem_type = UCC_MEMORY_TYPE_HOST;
}

/* init + post @args as @op's request on proc @p of @team. */
static void post_op(UccTeam_h &team, int p, ucc_coll_args_t &args, coll_op &op)
{
    op.ctx = ctx_ptr(team, p);
    ASSERT_EQ(UCC_OK, ucc_collective_init(&args, &op.req, team->procs[p].team));
    ASSERT_EQ(UCC_OK, ucc_collective_post(op.req));
}

/* Post an allreduce(SUM) of a single int64 on proc @p; result must equal @expect. */
static void post_allreduce(
    UccTeam_h &team, int p, int64_t val, int64_t expect, coll_op &op)
{
    op.sbuf         = val;
    op.rbuf         = 0;
    op.expect       = expect;
    op.is_allreduce = true;

    ucc_coll_args_t args;
    memset(&args, 0, sizeof(args));
    args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
    args.op        = UCC_OP_SUM;
    set_int64_host_buf(args.src.info, &op.sbuf);
    set_int64_host_buf(args.dst.info, &op.rbuf);

    post_op(team, p, args, op);
}

/* Post a bcast of a single int64 from root proc 0 on proc @p. */
static void post_bcast(UccTeam_h &team, int p, int64_t val, coll_op &op)
{
    op.rbuf         = (p == 0) ? val : 0;
    op.expect       = val;
    op.is_allreduce = false;

    ucc_coll_args_t args;
    memset(&args, 0, sizeof(args));
    args.coll_type = UCC_COLL_TYPE_BCAST;
    args.root      = 0;
    set_int64_host_buf(args.src.info, &op.rbuf);

    post_op(team, p, args, op);
}

/* Drive a set of in-flight per-proc collectives to completion, finalize, and
   validate each.  Every participating context is pumped each round. */
static void drive_and_check(std::vector<coll_op *> ops)
{
    bool all_done = false;
    while (!all_done) {
        all_done = true;
        for (auto *op : ops) {
            ucc_status_t st = ucc_collective_test(op->req);
            ASSERT_GE(st, 0);
            if (st != UCC_OK) {
                all_done = false;
            }
        }
        for (auto *op : ops) {
            ucc_context_progress(op->ctx);
        }
    }
    for (auto *op : ops) {
        ASSERT_EQ(UCC_OK, ucc_collective_finalize(op->req));
        ASSERT_EQ(op->expect, op->rbuf)
            << (op->is_allreduce ? "allreduce" : "bcast") << " wrong result";
    }
}

/* Parent (LIVE) + derived team, identical 2-proc membership, distinct team-ids.
   Each rank posts on each team in OPPOSITE relative order; distinct tag domains
   keep this correct where a shared id would cross-match. */
UCC_TEST_F(test_team_cache_integration, derived_coexist_interleaved)
{
    UccJob job(
        4,
        UccJob::UCC_JOB_CTX_GLOBAL,
        {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y")});

    UccTeam_h   t1 = job.create_team(2, /*use_team_ep_map=*/true);

    ucc_team_t *p0 = team_ptr(t1, 0);
    ucc_team_t *p1 = team_ptr(t1, 1);
    ASSERT_EQ(UCC_TEAM_CACHE_STATE_LIVE, p0->cache_state);
    EXPECT_EQ(0, p0->is_derived);
    ASSERT_NE(nullptr, p0->artifacts);
    run_barrier(t1);

    UccTeam_h   t2 = job.create_team(2, /*use_team_ep_map=*/true);

    ucc_team_t *d0 = team_ptr(t2, 0);
    ucc_team_t *d1 = team_ptr(t2, 1);

    /* Derivation fired: distinct object per proc, is_derived, not itself cached
       (NONE), sharing the parent's artifacts holder. */
    ASSERT_EQ(1, d0->is_derived);
    EXPECT_EQ(1, d1->is_derived);
    EXPECT_NE(p0, d0);
    EXPECT_NE(p1, d1);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_NONE, d0->cache_state);
    EXPECT_EQ(p0->artifacts, d0->artifacts);
    EXPECT_EQ(p1->artifacts, d1->artifacts);
    EXPECT_GE(d0->artifacts->refcount, 2);

    /* Distinct team-ids => distinct tag/seq domains.  ASSERT so a shared id
       fails fast instead of hanging the loop below. */
    ASSERT_NE(p0->id, d0->id);
    ASSERT_NE(team_ptr(t1, 1)->id, team_ptr(t2, 1)->id);

    const int kIters = 40;
    for (int it = 0; it < kIters; it++) {
        int64_t ar_val = 100 + it;
        int64_t ar_exp = ar_val * 2;
        int64_t bc_val = 900000 + it;

        coll_op p0_ar, p0_bc, p1_ar, p1_bc;

        /* proc 0: t1.allreduce then t2.bcast; proc 1: opposite order. */
        post_allreduce(t1, 0, ar_val, ar_exp, p0_ar);
        post_bcast(t2, 0, bc_val, p0_bc);
        post_bcast(t2, 1, bc_val, p1_bc);
        post_allreduce(t1, 1, ar_val, ar_exp, p1_ar);

        drive_and_check({&p0_ar, &p1_ar, &p0_bc, &p1_bc});
    }

    /* Teardown: derived first, parent last.  Parent must remain LIVE + usable. */
    t2.reset();
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, p0->cache_state);
    run_barrier(t1);
    t1.reset();
}

/* Real UccJob, 2-slot FIFO cache.  Create/destroy T1{2p}, T2{3p} (both DORMANT,
   cache full), then create T3{4p}: admission evicts T1 and drains its destroy
   synchronously; T3 lands in the freed slot.  Asserts evictions >= 1 and
   size <= max_size. */
UCC_TEST_F(test_team_cache_integration, evict_id_release_on_evict)
{
    UccJob job(
        4,
        UccJob::UCC_JOB_CTX_GLOBAL,
        {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y"),
         ucc_env_var_t("UCC_TEAM_CACHE_MAX_SIZE", "2"),
         ucc_env_var_t("UCC_TEAM_CACHE_EVICTION", "fifo"),
         ucc_env_var_t("UCC_TEAM_IDS_POOL_SIZE", "1")});

    UccTeam_h t1 = job.create_team(2, /*use_team_ep_map=*/true);
    ASSERT_NE(nullptr, t1);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, team_ptr(t1)->cache_state);
    run_barrier(t1);
    t1.reset(); /* T1 -> DORMANT (slot 1) */

    UccTeam_h t2 = job.create_team(3, /*use_team_ep_map=*/true);
    ASSERT_NE(nullptr, t2);
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, team_ptr(t2)->cache_state);
    run_barrier(t2);

    ucc_context_t    *ctx0  = ctx_ptr(t2, 0);
    ucc_team_cache_t *cache = ctx0->team_cache;
    ASSERT_NE(nullptr, cache);
    uint64_t evictions_before = cache->stats.evictions;

    t2.reset(); /* T2 -> DORMANT (slot 2, cache full) */

    UccTeam_h t3 = job.create_team(4, /*use_team_ep_map=*/true);
    ASSERT_NE(nullptr, t3);
    run_barrier(t3);

    EXPECT_GT(cache->stats.evictions, evictions_before)
        << "inserting T3 into a full cache must evict";
    EXPECT_LE(cache->size, cache->max_size);
    EXPECT_TRUE(ucc_list_is_empty(&cache->pending_destroy))
        << "pending destroys must complete synchronously in gtest context";
}

/* Pool-pressure headroom: six DISTINCT-membership teams churned through a 2-slot
   cache + size-1 ID pool.  Needs a 16-proc job so sizes 2..7 are all distinct. */
UCC_TEST_F(test_team_cache_integration, id_pool_headroom_with_dormant_teams)
{
    UccJob job(
        16,
        UccJob::UCC_JOB_CTX_GLOBAL,
        {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y"),
         ucc_env_var_t("UCC_TEAM_CACHE_MAX_SIZE", "2"),
         ucc_env_var_t("UCC_TEAM_CACHE_EVICTION", "fifo"),
         ucc_env_var_t("UCC_TEAM_IDS_POOL_SIZE", "1")});

    ucc_context_t    *ctx0   = nullptr;
    ucc_team_cache_t *cache  = nullptr;

    const int         kTeams = 6;
    for (int i = 0; i < kTeams; i++) {
        int sz = 2 + i; /* sizes 2..7, all distinct, all fit in a 16-proc job */
        UccTeam_h t = job.create_team(sz, /*use_team_ep_map=*/true);
        ASSERT_NE(nullptr, t)
            << "create_team must succeed (no UCC_ERR_NO_RESOURCE) at team "
            << i;
        if (i == 0) {
            ctx0  = ctx_ptr(t, 0);
            cache = ctx0->team_cache;
        }
        run_barrier(t);
        t.reset();
    }

    ASSERT_NE(nullptr, cache);
    EXPECT_LE(cache->size, cache->max_size);
    EXPECT_GT(cache->stats.evictions, 0u)
        << "eviction must fire across " << kTeams << " distinct teams";
}

/* With UCC_TEAM_CACHE_DUMP_STATS=y the knob is wired onto the cache struct and
   the destroy path dumps stats; the knob-off leg reads back zero. */
UCC_TEST_F(test_team_cache_integration, dump_stats_integration_knob_on)
{
    UccJob job(
        4,
        UccJob::UCC_JOB_CTX_GLOBAL,
        {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y"),
         ucc_env_var_t("UCC_TEAM_CACHE_DUMP_STATS", "y")});

    UccTeam_h t = job.create_team(2, /*use_team_ep_map=*/true);
    ASSERT_NE(nullptr, t);

    ucc_context_t *ctx = ctx_ptr(t, 0);
    ASSERT_NE(nullptr, ctx->team_cache);
    EXPECT_NE(0u, ctx->team_cache->dump_stats);

    run_barrier(t);
    t.reset(); /* destroy triggers dump before drain */

    UccJob job_off(
        4,
        UccJob::UCC_JOB_CTX_GLOBAL,
        {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y"),
         ucc_env_var_t("UCC_TEAM_CACHE_DUMP_STATS", "n")});

    UccTeam_h t_off = job_off.create_team(2, /*use_team_ep_map=*/true);
    ASSERT_NE(nullptr, t_off);

    ucc_context_t *ctx_off = ctx_ptr(t_off, 0);
    ASSERT_NE(nullptr, ctx_off->team_cache);
    EXPECT_EQ(0u, ctx_off->team_cache->dump_stats);

    run_barrier(t_off);
    t_off.reset();
}

/* With DISABLE_LINEAR_CHECK=y the hash-trust path is used; for non-colliding
   membership it produces the same dormant reuse as the safe path. */
UCC_TEST_F(test_team_cache_integration, disable_linear_check_accepted)
{
    UccJob job(
        4,
        UccJob::UCC_JOB_CTX_GLOBAL,
        {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y"),
         ucc_env_var_t("UCC_TEAM_CACHE_DISABLE_LINEAR_CHECK", "y")});

    UccTeam_h t1 = job.create_team(2, /*use_team_ep_map=*/true);
    ASSERT_NE(nullptr, t1);

    ucc_team_t    *tp0_before = team_ptr(t1, 0);
    ucc_context_t *ctx0       = ctx_ptr(t1, 0);

    ASSERT_NE(nullptr, ctx0->team_cache);
    EXPECT_NE(0u, ctx0->team_cache->disable_linear_check);

    run_barrier(t1);
    t1.reset();
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_DORMANT, tp0_before->cache_state);

    UccTeam_h t2 = job.create_team(2, /*use_team_ep_map=*/true);
    ASSERT_NE(nullptr, t2);

    ucc_team_t *tp0_after = team_ptr(t2, 0);
    EXPECT_EQ(tp0_before, tp0_after)
        << "hash-trust re-adopt must return the same ucc_team_t pointer";
    EXPECT_EQ(UCC_TEAM_CACHE_STATE_LIVE, tp0_after->cache_state);
    EXPECT_GE(ctx0->team_cache->stats.hits, 1u);

    run_barrier(t2);
}

/* Knob defaults and overrides: unset -> derived ON, reseat OFF; explicit
   derived=n / reseat=y reach the cache struct. */
UCC_TEST_F(test_team_cache_integration, derived_reseat_knob_defaults)
{
    {
        UccJob job(
            4,
            UccJob::UCC_JOB_CTX_GLOBAL,
            {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y")});

        UccTeam_h t1 = job.create_team(2, /*use_team_ep_map=*/true);
        ASSERT_NE(nullptr, t1);
        ucc_context_t *ctx0 = ctx_ptr(t1, 0);
        ASSERT_NE(nullptr, ctx0->team_cache);

        EXPECT_NE(0u, ctx0->team_cache->derived)
            << "UCC_TEAM_CACHE_DERIVED must default ON";
        EXPECT_EQ(0u, ctx0->team_cache->reseat)
            << "UCC_TEAM_CACHE_RESEAT must default OFF (opt-in)";
        run_barrier(t1);
    }

    /* The UccJob env mechanism only restores pre-existing vars, so guard these
       ourselves to avoid leaking DERIVED=n into later tests. */
    ScopedEnv derived_guard("UCC_TEAM_CACHE_DERIVED");
    ScopedEnv reseat_guard("UCC_TEAM_CACHE_RESEAT");

    {
        UccJob job(
            4,
            UccJob::UCC_JOB_CTX_GLOBAL,
            {ucc_env_var_t("UCC_TEAM_CACHE_ENABLE", "y"),
             ucc_env_var_t("UCC_TEAM_CACHE_DERIVED", "n"),
             ucc_env_var_t("UCC_TEAM_CACHE_RESEAT", "y")});

        UccTeam_h t1 = job.create_team(2, /*use_team_ep_map=*/true);
        ASSERT_NE(nullptr, t1);
        ucc_context_t *ctx0 = ctx_ptr(t1, 0);
        ASSERT_NE(nullptr, ctx0->team_cache);

        EXPECT_EQ(0u, ctx0->team_cache->derived)
            << "UCC_TEAM_CACHE_DERIVED=n must clear the cache flag";
        EXPECT_NE(0u, ctx0->team_cache->reseat)
            << "UCC_TEAM_CACHE_RESEAT=y must set the cache flag";
        run_barrier(t1);
    }
}
