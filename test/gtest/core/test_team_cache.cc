/**
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
extern "C" {
#include "core/ucc_team_cache.h"
#include "core/ucc_team.h"
}
#include <common/test.h>
#include <common/test_ucc.h>
#include <vector>
#include <cstdlib>
#include <cstring>

/* Unit tests for the team-cache identity (build/hash/equal/free) and the
   cacheability policy, with no MPI job. */

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
