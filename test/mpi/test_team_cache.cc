/**
 * Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

/*
 * Multi-rank team-cache correctness tests. These cover behaviors that need real
 * multi-rank OOB collectives: dormant-reuse hit counting, freed-callback safety,
 * cross-rank agreement, and singleton teams. Single-process behavior is covered
 * by the gtest suite.
 *
 * Teams are created and destroyed directly so the recreate cycle is controlled
 * per test. The suite therefore runs only after the harness teams have been
 * destroyed, on a context with no unrelated live teams: a live world team would
 * otherwise divert a create onto the derived-from-live path and occupy cache
 * entries that the eviction tests reason about.
 *
 * Each test returns a verdict that run_team_cache_tests reduces across ranks and
 * tallies, so a skipped suite is reported as skipped rather than being
 * indistinguishable from a pass. A test that detects a failure records it and
 * still completes its collective sequence, so peers never block on a rank that
 * left early.
 */

#include "test_mpi.h"
#include "core/ucc_context.h"
#include "core/ucc_team_cache.h"
#include "utils/ucc_coll_utils.h"
#include <cstdlib>
#include <cstring>

/* create/destroy/recreate cycles used by the reuse tests. */
static const int kReuseIters = 5;

/* Verdict of one test; run_team_cache_tests reduces these across ranks. */
typedef enum {
    TC_PASS = 0,
    TC_FAIL = 1,
    TC_SKIP = 2
} tc_verdict_t;

/* The context's team cache, or NULL when caching is disabled. */
static ucc_team_cache_t *cache_of(ucc_context_h ctx)
{
    return ((ucc_context_t *)ctx)->team_cache;
}

/* Report a failed check. The caller records TC_FAIL and keeps going so that its
   remaining collectives stay matched with the peers'. */
static void tc_report_fail(const char *name, int rank, const char *what)
{
    std::cerr << "*** UCC TEST FAIL: " << name << " rank " << rank << ": "
              << what << "\n";
}

static tc_verdict_t tc_skip(const char *name, int rank, const char *why)
{
    if (0 == rank) {
        std::cout << "SKIP " << name << ": " << why << "\n";
    }
    return TC_SKIP;
}

/* Pump the context until a single collective request completes; abort on error
   (a failed collective leaves the ranks out of step, so there is nothing left
   to report cooperatively). */
static void progress_until(ucc_context_h ctx, ucc_coll_req_h req,
                           const char *what)
{
    ucc_status_t st;

    while (UCC_OK != (st = ucc_collective_test(req))) {
        if (st < 0) {
            std::cerr << "*** UCC TEST FAIL: " << what << " ("
                      << ucc_status_string(st) << ")\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        ucc_context_progress(ctx);
    }
}

static ucc_ep_map_t ep_map_full(int size)
{
    ucc_ep_map_t m;

    memset(&m, 0, sizeof(m));
    m.type   = UCC_EP_MAP_FULL;
    m.ep_num = size;
    return m;
}

/* Full-membership world team (EP_MAP FULL over MPI_COMM_WORLD). */
static ucc_team_h create_world_team(ucc_context_h ctx, int size,
                                    uint64_t ext_id = 0)
{
    ucc_ep_map_t m = ep_map_full(size);

    return ucc_test_create_team(ctx, MPI_COMM_WORLD, &m, ext_id, false);
}

/* Subset team over @comm using an explicit team-idx -> ctx-rank array map. */
static ucc_team_h create_array_team(ucc_context_h ctx, MPI_Comm comm,
                                    ucc_rank_t *map, int nmembers)
{
    ucc_ep_map_t m;

    memset(&m, 0, sizeof(m));
    m.type            = UCC_EP_MAP_ARRAY;
    m.ep_num          = nmembers;
    m.array.map       = map;
    m.array.elem_size = sizeof(ucc_rank_t);
    return ucc_test_create_team(ctx, comm, &m, 0, false);
}

/* destroy_ucc_team - blocking ucc_team_destroy with a context progress pump.
   The harness's UccTestMpi::destroy_team spins without progressing, which is
   only safe for teardown at exit. */
static void destroy_ucc_team(ucc_team_h team, ucc_context_h ctx)
{
    ucc_status_t status;

    while (UCC_INPROGRESS == (status = ucc_team_destroy(team))) {
        ucc_context_progress(ctx);
    }
    if (UCC_OK != status) {
        std::cerr << "*** UCC TEST FAIL: ucc_team_destroy failed ("
                  << ucc_status_string(status) << ")\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

/* Non-zero if the team that used to live at @handle is on the dormant list.
   @handle is compared by value only: when destroy admits the team to the cache
   the object is retained, but on the teardown path it is freed, and reading
   team->cache_state there would be a use-after-free. */
static int is_dormant(ucc_team_cache_t *cache, uintptr_t handle)
{
    ucc_team_t *dt;

    ucc_list_for_each (dt, &cache->dormant, cache_link) {
        if ((uintptr_t)dt == handle) {
            return 1;
        }
    }
    return 0;
}

/* Drain dormant entries left by earlier tests, with barriers so that every rank
   drains before any rank starts creating again. */
static void drain_cache(ucc_context_h ctx)
{
    MPI_Barrier(MPI_COMM_WORLD);
    ucc_team_cache_drain((ucc_context_t *)ctx);
    MPI_Barrier(MPI_COMM_WORLD);
}

/* run_barrier_on_team - blocking barrier on @team; aborts on failure. */
static void run_barrier_on_team(ucc_team_h team, ucc_context_h ctx)
{
    ucc_coll_args_t args;
    ucc_coll_req_h  req;

    memset(&args, 0, sizeof(args));
    args.coll_type = UCC_COLL_TYPE_BARRIER;

    UCC_CHECK(ucc_collective_init(&args, &req, team));
    UCC_CHECK(ucc_collective_post(req));
    progress_until(ctx, req, "barrier");
    UCC_CHECK(ucc_collective_finalize(req));
}

/* Fill allreduce(SUM) args for a single int64. @flags carries
   UCC_COLL_ARGS_FLAG_PERSISTENT for the persistent-handle test. */
static void fill_allreduce_int64(ucc_coll_args_t *args, int64_t *sbuf,
                                 int64_t *rbuf, uint64_t flags)
{
    memset(args, 0, sizeof(*args));
    args->coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    args->op                = UCC_OP_SUM;
    args->src.info.buffer   = sbuf;
    args->src.info.count    = 1;
    args->src.info.datatype = UCC_DT_INT64;
    args->src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    args->dst.info.buffer   = rbuf;
    args->dst.info.count    = 1;
    args->dst.info.datatype = UCC_DT_INT64;
    args->dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
    if (flags) {
        args->mask  = UCC_COLL_ARGS_FIELD_FLAGS;
        args->flags = flags;
    }
}

/* Blocking allreduce(SUM) of a single int64 over @team; pumps @ctx and aborts
   on error. Returns the reduced value. */
static int64_t run_allreduce_int64(ucc_team_h team, ucc_context_h ctx,
                                   int64_t sbuf, const char *where)
{
    int64_t         rbuf = 0;
    ucc_coll_args_t args;
    ucc_coll_req_h  req;

    fill_allreduce_int64(&args, &sbuf, &rbuf, 0);
    UCC_CHECK(ucc_collective_init(&args, &req, team));
    UCC_CHECK(ucc_collective_post(req));
    progress_until(ctx, req, where);
    UCC_CHECK(ucc_collective_finalize(req));
    return rbuf;
}

/* Expected allreduce(SUM) of every rank's world_rank. */
static int64_t rank_sum(int world_size)
{
    return (int64_t)world_size * (world_size - 1) / 2;
}

/* ==========================================================================
 * dup_coexist_derived: a parent world team and its live derived (MPI_Comm_dup
 * analogue) run interleaved collectives concurrently. Regression for the alias
 * hazard where two coexisting identical-membership teams share a team id.
 *   - ext_ids==false: implicit-id dup (the base regression).
 *   - ext_ids==true : parent/derived carry DISTINCT external ids and must share
 *     one artifacts holder.
 * ========================================================================== */

/* Drive one allreduce(SUM int64) + one bcast(int64 from root 0), in the given
   order, over MPI_COMM_WORLD membership; validate both results. @order==0:
   allreduce then bcast; @order==1: reverse. Both are posted before either
   completes so they are in flight together (and interleave across teams).
   Returns 0 on success, 1 if a result was wrong. */
static int drive_ar_bc(ucc_team_h ar_team, ucc_team_h bc_team,
                       ucc_context_h ctx, int rank, int size, int iter,
                       int order)
{
    int64_t ar_send = 100 + iter;
    int64_t ar_recv = 0;
    int64_t ar_exp  = (int64_t)(100 + iter) * size;
    int64_t bc_buf  = (rank == 0) ? (900000 + iter) : 0;
    int64_t bc_exp  = 900000 + iter;

    ucc_coll_args_t ar_args, bc_args;
    ucc_coll_req_h  ar_req, bc_req;
    ucc_status_t    sa, sb;

    fill_allreduce_int64(&ar_args, &ar_send, &ar_recv, 0);

    memset(&bc_args, 0, sizeof(bc_args));
    bc_args.coll_type         = UCC_COLL_TYPE_BCAST;
    bc_args.root              = 0;
    bc_args.src.info.buffer   = &bc_buf;
    bc_args.src.info.count    = 1;
    bc_args.src.info.datatype = UCC_DT_INT64;
    bc_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;

    /* Post in the requested relative order (per-rank interleave). */
    if (order == 0) {
        UCC_CHECK(ucc_collective_init(&ar_args, &ar_req, ar_team));
        UCC_CHECK(ucc_collective_post(ar_req));
        UCC_CHECK(ucc_collective_init(&bc_args, &bc_req, bc_team));
        UCC_CHECK(ucc_collective_post(bc_req));
    } else {
        UCC_CHECK(ucc_collective_init(&bc_args, &bc_req, bc_team));
        UCC_CHECK(ucc_collective_post(bc_req));
        UCC_CHECK(ucc_collective_init(&ar_args, &ar_req, ar_team));
        UCC_CHECK(ucc_collective_post(ar_req));
    }

    /* Progress both to completion. */
    do {
        sa = ucc_collective_test(ar_req);
        sb = ucc_collective_test(bc_req);
        if (sa < 0 || sb < 0) {
            std::cerr << "*** UCC TEST FAIL: coll test ("
                      << ucc_status_string(sa < 0 ? sa : sb) << ")\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
        ucc_context_progress(ctx);
    } while (sa != UCC_OK || sb != UCC_OK);

    UCC_CHECK(ucc_collective_finalize(ar_req));
    UCC_CHECK(ucc_collective_finalize(bc_req));

    if (ar_recv != ar_exp || bc_buf != bc_exp) {
        std::cerr << "*** UCC TEST FAIL: dup_coexist rank " << rank << " iter "
                  << iter << ": allreduce got " << ar_recv << " (exp " << ar_exp
                  << "), bcast got " << bc_buf << " (exp " << bc_exp << ")\n";
        return 1;
    }
    return 0;
}

static tc_verdict_t test_dup_coexist_derived(ucc_context_h ctx, int world_rank,
                                             int world_size, bool ext_ids)
{
    const int      kIters     = ext_ids ? 8 : 6;
    const uint64_t parent_id  = ext_ids ? 100 : 0;
    const uint64_t derived_id = ext_ids ? 200 : 0;
    const char    *name       = ext_ids ? "dup_coexist_derived[ext_ids]"
                                        : "dup_coexist_derived";
    tc_verdict_t   v          = TC_PASS;
    ucc_team_h     parent, derived;

    if (!cache_of(ctx)->derived) {
        return tc_skip(name, world_rank, "UCC_TEAM_CACHE_DERIVED not enabled");
    }

    /* A dup needs a LIVE insertable parent to derive from; drain dormant
       squatters so the world-membership bucket is clean. */
    drain_cache(ctx);

    /* Parent world team (derivable), kept live. */
    parent = create_world_team(ctx, world_size, parent_id);
    run_barrier_on_team(parent, ctx);

    /* Second identical-membership team while the parent is live -> derived
       create path (MPI_Comm_dup analogue), with a distinct external id under
       ext_ids. */
    derived = create_world_team(ctx, world_size, derived_id);

    /* Distinct team ids: coexisting teams must not alias tag/seq domains. */
    if (((ucc_team_t *)parent)->id == ((ucc_team_t *)derived)->id) {
        tc_report_fail(name, world_rank, "coexisting teams share a team id");
        v = TC_FAIL;
    }
    /* Must actually take the derived path (a full-create regression would still
       pass the distinct-id check above). */
    if (!((ucc_team_t *)derived)->is_derived) {
        tc_report_fail(name, world_rank,
                       "coexisting identical-membership team was not derived "
                       "(full-create regression)");
        v = TC_FAIL;
    }
    /* External-id variant: derived must share the parent's artifacts holder. */
    if (ext_ids &&
        ((ucc_team_t *)parent)->artifacts != ((ucc_team_t *)derived)->artifacts) {
        tc_report_fail(name, world_rank,
                       "derived team did not share the parent artifacts holder");
        v = TC_FAIL;
    }

    /* Interleaved-order collectives with opposite per-rank ordering across both
       live teams: even ranks allreduce(parent)+bcast(derived), odd the
       reverse. */
    for (int it = 0; it < kIters; it++) {
        int order = (world_rank % 2 == 0) ? 0 : 1;

        if (drive_ar_bc(parent, derived, ctx, world_rank, world_size, it,
                        order)) {
            v = TC_FAIL;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    destroy_ucc_team(derived, ctx);
    run_barrier_on_team(parent, ctx); /* parent still usable */
    destroy_ucc_team(parent, ctx);

    if (v == TC_PASS && 0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return v;
}

/* ==========================================================================
 * dormant_reuse_stats: create/destroy/recreate kReuseIters times; from iter 1
 * the dormant team is re-adopted (HIT) every time.
 * ========================================================================== */
static tc_verdict_t test_dormant_reuse_stats(ucc_context_h ctx, int world_rank,
                                             int world_size)
{
    const char       *name  = "dormant_reuse_stats";
    ucc_team_cache_t *cache = cache_of(ctx);
    tc_verdict_t      v     = TC_PASS;
    uint64_t          hits0, hitsN;

    /* Drain dormant squatters from prior tests (a leftover would skew hits). */
    drain_cache(ctx);
    hits0 = cache->stats.hits;

    for (int i = 0; i < kReuseIters; i++) {
        ucc_team_h team = create_world_team(ctx, world_size);

        run_barrier_on_team(team, ctx);
        MPI_Barrier(MPI_COMM_WORLD);
        destroy_ucc_team(team, ctx);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    hitsN = cache->stats.hits;
    /* The first create is a miss+insert; the recreates must all be hits. */
    if (hitsN - hits0 < (uint64_t)(kReuseIters - 1)) {
        std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                  << ": expected >=" << (kReuseIters - 1)
                  << " dormant hits, got " << (hitsN - hits0) << "\n";
        v = TC_FAIL;
    }
    if (v == TC_PASS && 0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return v;
}

/* ==========================================================================
 * derived_reuse: with a live parent kept throughout, create/destroy a derived
 * team of identical membership kReuseIters times. From iter 1 the dormant
 * derived is re-adopted (cache HIT).
 *   - drift==false: the derived's external id is stable, so an exact-identity
 *     lookup re-adopts it.
 *   - drift==true : the derived's external id drifts every iteration, so only a
 *     membership-match re-adopt (reseat) can hit; requires
 *     UCC_TEAM_CACHE_RESEAT and UCC_TEAM_CACHE_DERIVED, else it skips.
 * ========================================================================== */
static tc_verdict_t test_derived_reuse(ucc_context_h ctx, int world_rank,
                                       int world_size, bool drift)
{
    const char       *name  = drift ? "derived_reuse[drift]" : "derived_reuse";
    ucc_team_cache_t *cache = cache_of(ctx);
    tc_verdict_t      v     = TC_PASS;
    ucc_team_h        parent;

    if (!cache->derived) {
        return tc_skip(name, world_rank, "UCC_TEAM_CACHE_DERIVED not enabled");
    }
    if (drift && !cache->reseat) {
        return tc_skip(name, world_rank, "UCC_TEAM_CACHE_RESEAT not enabled");
    }

    drain_cache(ctx);

    /* Parent stays live throughout; its stable ext_id=1 is distinct from every
       derived id used below. */
    parent = create_world_team(ctx, world_size, /*ext_id=*/1);
    run_barrier_on_team(parent, ctx);

    for (int i = 0; i < kReuseIters; i++) {
        uint64_t hits_before = cache->stats.hits;
        /* Derived: same membership as the parent. Under drift the ext_id
           changes every iteration (100, 101, ...) so the exact-identity lookup
           misses and only a reseat re-adopt can hit; otherwise ext_id=2 is
           stable. */
        uint64_t   derived_id = drift ? (uint64_t)(100 + i) : 2;
        ucc_team_h derived    = create_world_team(ctx, world_size, derived_id);
        int64_t    sbuf       = (int64_t)(100 + i);
        int64_t    exp        = sbuf * (int64_t)world_size;
        int64_t    rbuf       = run_allreduce_int64(derived, ctx, sbuf,
                                                    "allreduce on derived team");

        if (rbuf != exp) {
            std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                      << " iter " << i << ": allreduce got " << rbuf << " (exp "
                      << exp << ")\n";
            v = TC_FAIL;
        }
        /* From iter 1 onwards: expect a cache hit re-adopting the dormant
           derived (via reseat under drift). */
        if (i > 0 && cache->stats.hits <= hits_before) {
            std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                      << " iter " << i
                      << ": no cache hit for the derived re-adopt\n";
            v = TC_FAIL;
        }

        MPI_Barrier(MPI_COMM_WORLD);
        destroy_ucc_team(derived, ctx); /* -> dormant */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    run_barrier_on_team(parent, ctx); /* parent must still be functional */
    destroy_ucc_team(parent, ctx);

    if (v == TC_PASS && 0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return v;
}

/* ==========================================================================
 * ep_map_cb_freed_after_cache: a cached team must not retain the caller's ep_map
 * callback context past its lifetime. OMPI coll/ucc passes a UCC_EP_MAP_CB whose
 * cb_ctx is the MPI communicator; after the comm is freed, any deref of that
 * cb_ctx is a use-after-free. Here cb_ctx is a heap box that is POISONED+FREED
 * after the team goes dormant; re-adopting the team and evaluating its
 * operational map must never call back into the freed box.
 * ========================================================================== */

struct cb_ctx_box {
    uint64_t   magic;    /* CB_CTX_MAGIC while live, poisoned after free */
    ucc_rank_t ranks[1]; /* flexible: team ep -> ctx rank (identity here) */
};
static const uint64_t CB_CTX_MAGIC = 0xC0FFEE5AULL;

static uint64_t poisonable_rank_cb(uint64_t ep, void *cb_ctx)
{
    struct cb_ctx_box *box = (struct cb_ctx_box *)cb_ctx;

    /* If the cache retained this (freed) context, magic no longer matches -
       fail loudly instead of silently reading poisoned memory. */
    if (box->magic != CB_CTX_MAGIC) {
        std::cerr << "*** UCC TEST FAIL: use-after-free - ep_map callback "
                     "invoked on a freed communicator context\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return box->ranks[ep];
}

static ucc_team_h create_cb_team(ucc_context_h ctx, int size,
                                 struct cb_ctx_box *box)
{
    ucc_ep_map_t m;

    memset(&m, 0, sizeof(m));
    m.type      = UCC_EP_MAP_CB;
    m.ep_num    = size;
    m.cb.cb     = poisonable_rank_cb;
    m.cb.cb_ctx = (void *)box;
    return ucc_test_create_team(ctx, MPI_COMM_WORLD, &m, 0, false);
}

static struct cb_ctx_box *alloc_cb_box(int world_size)
{
    size_t box_sz = sizeof(struct cb_ctx_box) +
                    (world_size - 1) * sizeof(ucc_rank_t);
    struct cb_ctx_box *box = (struct cb_ctx_box *)malloc(box_sz);

    if (box == NULL) {
        std::cerr << "*** UCC TEST FAIL: cb_ctx_box allocation failed\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    box->magic = CB_CTX_MAGIC;
    for (int i = 0; i < world_size; i++) {
        box->ranks[i] = (ucc_rank_t)i; /* world identity mapping */
    }
    return box;
}

static tc_verdict_t test_ep_map_cb_freed_after_cache(ucc_context_h ctx,
                                                     int world_rank,
                                                     int world_size)
{
    const char        *name  = "ep_map_cb_freed_after_cache";
    ucc_team_cache_t  *cache = cache_of(ctx);
    tc_verdict_t       v     = TC_PASS;
    uint64_t           hits_before;
    struct cb_ctx_box *box;
    ucc_team_h         team, team2;
    ucc_team_t        *t;

    drain_cache(ctx);
    hits_before = cache->stats.hits;
    box         = alloc_cb_box(world_size);

    /* First create + use + destroy -> the team goes dormant. */
    team = create_cb_team(ctx, world_size, box);
    run_barrier_on_team(team, ctx);
    destroy_ucc_team(team, ctx);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Free the callback context (as MPI_Comm_free would). A cached team that
       still points here would now be dangling. */
    box->magic = 0xDEADDEADULL; /* poison so a stale deref is caught */
    free(box);

    /* Re-create the identical team. On a cache hit this re-adopts the dormant
       team whose operational map is UCC-owned, so it never touches the freed
       box. The fresh box only satisfies the create API. */
    box   = alloc_cb_box(world_size);
    team2 = create_cb_team(ctx, world_size, box);
    run_barrier_on_team(team2, ctx);

    /* The re-adopt is the whole point: without a hit, nothing below exercises a
       retained cb_ctx and a pass would be meaningless. */
    if (cache->stats.hits <= hits_before) {
        tc_report_fail(name, world_rank,
                       "re-create did not hit the dormant cache, so the "
                       "retained-cb_ctx path was never exercised");
        v = TC_FAIL;
    }

    /* Evaluate the re-adopted team's operational ctx_map for every endpoint -
       the exact access TL/UCP performs when resolving a peer. With the fix the
       map is UCC-owned, so this resolves correctly without calling back into the
       freed box; without it, poisonable_rank_cb aborts on the poison. */
    t = (ucc_team_t *)team2;
    for (ucc_rank_t e = 0; e < (ucc_rank_t)world_size; e++) {
        ucc_rank_t got = ucc_ep_map_eval(UCC_TEAM_CTX_MAP(t), e);

        if (got != e) {
            std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                      << ": operational ctx_map endpoint " << (int)e
                      << " resolved to " << (int)got << " (expected " << (int)e
                      << ")\n";
            v = TC_FAIL;
        }
    }

    destroy_ucc_team(team2, ctx);
    free(box);

    if (v == TC_PASS && 0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return v;
}

/* ==========================================================================
 * overlap_agreement: overlapping subcommunicator scopes plus a small cache force
 * DIVERGENT per-rank eviction, which previously deadlocked (one rank re-adopts a
 * dormant team while a peer that evicted it enters a fresh collective build and
 * waits forever). The cross-rank agreement must reconcile the split hit/miss to
 * a consistent fresh build.
 *
 * Requires UCC_TEAM_CACHE_MAX_SIZE<=2 and >=3 ranks. Without the small cache no
 * eviction happens, so the test would construct no divergence at all and still
 * print PASS; the max_size guard below makes that configuration an explicit
 * skip instead.
 * ========================================================================== */
static tc_verdict_t test_overlap_agreement(ucc_context_h ctx, int world_rank,
                                           int world_size)
{
    const char       *name      = "overlap_agreement";
    ucc_team_cache_t *cache     = cache_of(ctx);
    ucc_rank_t        ranksA[2] = {0, 1};
    ucc_rank_t        ranksB[2] = {1, 2};
    ucc_rank_t        ranksD[3] = {0, 1, 2};
    MPI_Comm          commA, commB, commD;
    ucc_team_h        t;

    if (world_size < 3 || cache->max_size > 2) {
        return tc_skip(name, world_rank, "needs >=3 ranks and MAX_SIZE<=2");
    }

    drain_cache(ctx);

    /* Overlapping member sets: A{0,1}, B{1,2}, D{0,1,2}. */
    MPI_Comm_split(MPI_COMM_WORLD, (world_rank <= 1) ? 0 : MPI_UNDEFINED,
                   world_rank, &commA);
    MPI_Comm_split(MPI_COMM_WORLD,
                   (world_rank >= 1 && world_rank <= 2) ? 0 : MPI_UNDEFINED,
                   world_rank, &commB);
    MPI_Comm_split(MPI_COMM_WORLD, (world_rank <= 2) ? 0 : MPI_UNDEFINED,
                   world_rank, &commD);

    /* 1) A dormant on {0,1}; 2) B dormant on {1,2} (rank 1's cache is now full
       at MAX_SIZE=2); 3) D on {0,1,2} evicts the oldest dormant (A) on rank 1
       but not on rank 0 -> divergence; 4) re-create A: rank 0 re-adopts, rank 1
       missed. Pre-agreement this deadlocks; the vote must reconcile to a fresh
       build on both. */
    if (commA != MPI_COMM_NULL) {
        t = create_array_team(ctx, commA, ranksA, 2);
        destroy_ucc_team(t, ctx);
    }
    if (commB != MPI_COMM_NULL) {
        t = create_array_team(ctx, commB, ranksB, 2);
        destroy_ucc_team(t, ctx);
    }
    if (commD != MPI_COMM_NULL) {
        t = create_array_team(ctx, commD, ranksD, 3);
        destroy_ucc_team(t, ctx);
    }
    if (commA != MPI_COMM_NULL) {
        t = create_array_team(ctx, commA, ranksA, 2);
        run_barrier_on_team(t, ctx); /* must complete, not deadlock */
        destroy_ucc_team(t, ctx);
        MPI_Comm_free(&commA);
    }
    if (commB != MPI_COMM_NULL) {
        MPI_Comm_free(&commB);
    }
    if (commD != MPI_COMM_NULL) {
        MPI_Comm_free(&commD);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return TC_PASS;
}

/* ==========================================================================
 * derived_exact_rebuild: a dormant DERIVED team that loses the cross-rank vote
 * must be rebuilt as a full team, not left half-derived. A live parent world
 * team (ext_id 1) plus a dormant derived world team (ext_id 2) fill the cache on
 * every rank. A subcomm team created only on ranks {0,1} evicts the dormant
 * derived there (FIFO), while ranks >=2 keep it. Re-creating the ext_id-2 world
 * team then splits the vote (ranks >=2 EXACT_REUSE the derived candidate, ranks
 * {0,1} miss -> DERIVED_FROM_LIVE), forcing a global MISS and the in-place
 * rebuild on ranks >=2. Without the de-derive fix those ranks skip ADDR_EXCHANGE
 * and desync/crash; with it, the allreduce below completes and is_derived is 0.
 * ========================================================================== */
static tc_verdict_t test_derived_exact_rebuild(ucc_context_h ctx,
                                               int world_rank, int world_size)
{
    const char       *name  = "derived_exact_rebuild";
    ucc_team_cache_t *cache = cache_of(ctx);
    tc_verdict_t      v     = TC_PASS;
    ucc_team_h        parent, derived, rebuilt;
    MPI_Comm          commAB;
    int64_t           got, exp;

    if (!cache->derived || world_size < 3 || cache->max_size > 2) {
        return tc_skip(name, world_rank,
                       "needs derived caching, >=3 ranks, MAX_SIZE<=2");
    }

    drain_cache(ctx);

    /* Live parent (ext_id 1) + dormant derived (ext_id 2): 2 entries, which is
       MAX_SIZE. */
    parent = create_world_team(ctx, world_size, /*ext_id=*/1);
    run_barrier_on_team(parent, ctx);
    derived = create_world_team(ctx, world_size, /*ext_id=*/2);
    run_barrier_on_team(derived, ctx);
    MPI_Barrier(MPI_COMM_WORLD);
    destroy_ucc_team(derived, ctx); /* -> dormant derived */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Ranks {0,1} only: a subcomm team evicts the dormant derived (cache full);
       ranks >=2 never create it and keep the dormant derived. */
    MPI_Comm_split(MPI_COMM_WORLD, (world_rank <= 1) ? 0 : MPI_UNDEFINED,
                   world_rank, &commAB);
    if (commAB != MPI_COMM_NULL) {
        ucc_rank_t ab[2] = {0, 1};
        ucc_team_h t     = create_array_team(ctx, commAB, ab, 2);

        run_barrier_on_team(t, ctx);
        destroy_ucc_team(t, ctx);
        MPI_Comm_free(&commAB);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Re-create the ext_id-2 world team: split vote -> global MISS -> rebuild.
       The allreduce exercises the ctx_map/topo the rebuild must have
       populated. */
    rebuilt = create_world_team(ctx, world_size, /*ext_id=*/2);
    got     = run_allreduce_int64(rebuilt, ctx, (int64_t)world_rank,
                                  "allreduce on rebuilt team");
    exp     = rank_sum(world_size);
    if (got != exp) {
        std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                  << ": allreduce got " << got << " (exp " << exp << ")\n";
        v = TC_FAIL;
    }
    if (((ucc_team_t *)rebuilt)->is_derived != 0 ||
        ((ucc_team_t *)rebuilt)->parent_id != 0) {
        std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                  << ": rebuilt team still marked derived (is_derived="
                  << ((ucc_team_t *)rebuilt)->is_derived << ")\n";
        v = TC_FAIL;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    destroy_ucc_team(rebuilt, ctx);
    run_barrier_on_team(parent, ctx); /* parent still functional */
    destroy_ucc_team(parent, ctx);
    ucc_team_cache_drain((ucc_context_t *)ctx);

    if (v == TC_PASS && 0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return v;
}

/* ==========================================================================
 * nonblocking_create_post: ucc_team_create_post must return promptly (post the
 * vote, not block) even if one rank is late entering it. Rank 0 sleeps briefly;
 * the other ranks call create_post and must return before rank 0 arrives.
 *
 * The threshold is the full rank-0 delay rather than a fraction of it: the
 * failure being detected is create_post blocking until rank 0 shows up, and the
 * CI legs run oversubscribed, where a tighter bound measures scheduling noise.
 * ========================================================================== */
static tc_verdict_t test_nonblocking_create_post(ucc_context_h ctx,
                                                 int world_rank, int world_size)
{
    const char       *name     = "nonblocking_create_post";
    const int         kSleepMs = 2000;
    tc_verdict_t      v        = TC_PASS;
    ucc_team_params_t p;
    ucc_team_h        team;
    ucc_status_t      status;
    double            t_start, elapsed_ms;

    if (world_size < 2) {
        return tc_skip(name, world_rank, "needs >=2 ranks");
    }

    /* Drain so this is a clean fresh create (the vote is posted regardless). */
    drain_cache(ctx);

    t_start = MPI_Wtime();
    if (world_rank == 0) {
        /* Delay entering create_post: peers must not block waiting on us. */
        usleep(kSleepMs * 1000);
    }

    /* Posted inline rather than through ucc_test_create_team, which blocks to
       completion; only the post itself is being timed. */
    memset(&p, 0, sizeof(p));
    p.mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
             UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_EP_MAP;
    p.oob.allgather = oob_allgather;
    p.oob.req_test  = oob_allgather_test;
    p.oob.req_free  = oob_allgather_free;
    p.oob.coll_info = (void *)(uintptr_t)MPI_COMM_WORLD;
    p.oob.n_oob_eps = world_size;
    p.oob.oob_ep    = world_rank;
    p.ep            = world_rank;
    p.ep_range      = UCC_COLLECTIVE_EP_RANGE_CONTIG;
    p.ep_map        = ep_map_full(world_size);

    UCC_CHECK(ucc_team_create_post(&ctx, 1, &p, &team));
    elapsed_ms = (MPI_Wtime() - t_start) * 1000.0;

    /* On a non-zero rank, create_post must have returned before rank 0's sleep
       elapsed - proving it posted (did not block on) the vote. */
    if (world_rank != 0 && elapsed_ms >= (double)kSleepMs) {
        std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                  << ": create_post blocked " << elapsed_ms
                  << "ms (rank 0 delay " << kSleepMs << "ms)\n";
        v = TC_FAIL;
    }

    while (UCC_INPROGRESS == (status = ucc_team_create_test(team))) {
        ucc_context_progress(ctx);
    }
    if (status < 0) {
        std::cerr << "*** UCC TEST FAIL: " << name << " create ("
                  << ucc_status_string(status) << ")\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    run_barrier_on_team(team, ctx);
    MPI_Barrier(MPI_COMM_WORLD);
    destroy_ucc_team(team, ctx);
    MPI_Barrier(MPI_COMM_WORLD);

    if (v == TC_PASS && 0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return v;
}

/* ==========================================================================
 * singleton_team: a size-1 cacheable team creates + reuses correctly with no
 * network vote (self-membership; the size>1 gate is not taken). Each rank builds
 * its own {self} team independently over MPI_COMM_SELF, so the recreates must
 * still be served from the local cache.
 * ========================================================================== */
static tc_verdict_t test_singleton_team(ucc_context_h ctx, int world_rank,
                                        int world_size)
{
    const char       *name  = "singleton_team";
    ucc_team_cache_t *cache = cache_of(ctx);
    tc_verdict_t      v     = TC_PASS;
    ucc_rank_t        self_map[1];
    uint64_t          hits0;

    (void)world_size;
    drain_cache(ctx);
    hits0 = cache->stats.hits;

    for (int i = 0; i < 3; i++) {
        ucc_team_h t;

        self_map[0] = (ucc_rank_t)world_rank; /* team idx 0 -> my ctx rank */
        t           = create_array_team(ctx, MPI_COMM_SELF, self_map, 1);
        run_barrier_on_team(t, ctx);
        destroy_ucc_team(t, ctx);
    }

    /* The first create is a miss; the other two must re-adopt the dormant team.
       Without this the test verified nothing beyond "did not crash". */
    if (cache->stats.hits - hits0 < 2) {
        std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                  << ": expected >=2 singleton cache hits, got "
                  << (cache->stats.hits - hits0) << "\n";
        v = TC_FAIL;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (v == TC_PASS && 0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return v;
}

/* ==========================================================================
 * persistent_handle_safety
 *
 * Verify that outstanding persistent collective handles block cache DORMANT
 * admission at team-destroy time, preventing handle-to-team aliasing under
 * EXACT_REUSE.
 *
 * Part A (counter lifecycle): init persistent allreduce -> count=1 -> finalize
 *   -> count=0 -> destroy -> team goes DORMANT -> recreate -> DORMANT hit.
 * Part B (bypass guard): init persistent allreduce -> count=1 -> destroy ->
 *   team NOT DORMANT (real teardown) -> no dormant entry for it -> recreate ->
 *   cache MISS (fresh build), not a stale re-adoption.
 *   The handle from Part B is intentionally not finalized: calling finalize
 *   after its team is destroyed is undefined behaviour; the handle's memory is
 *   an accepted bounded leak at test exit.
 *
 * Dormancy is checked against the cache's dormant list rather than through the
 * team handle: on the teardown path the team object is freed, so reading
 * team->cache_state would be a use-after-free in exactly the case the check
 * exists to catch.
 * ========================================================================== */
static tc_verdict_t test_persistent_handle_safety(ucc_context_h ctx,
                                                  int world_rank,
                                                  int world_size)
{
    const char       *name  = "persistent_handle_safety";
    ucc_team_cache_t *cache = cache_of(ctx);
    tc_verdict_t      v     = TC_PASS;
    int64_t           sbuf, rbuf, expected;
    ucc_coll_args_t   args;
    ucc_coll_req_h    reqA, reqB;
    ucc_team_h        tA, tA2, tB, tB2;
    uintptr_t         tA_addr, tB_addr, tB2_addr;
    uint64_t          hits_before, misses_before;

    drain_cache(ctx);
    expected = rank_sum(world_size);

    /* --- Part A: handle finalised before destroy --- */
    sbuf    = (int64_t)world_rank;
    rbuf    = 0;
    tA      = create_world_team(ctx, world_size);
    tA_addr = (uintptr_t)tA;

    fill_allreduce_int64(&args, &sbuf, &rbuf, UCC_COLL_ARGS_FLAG_PERSISTENT);
    UCC_CHECK(ucc_collective_init(&args, &reqA, tA));

    if (((ucc_team_t *)tA)->persistent_coll_count != 1) {
        tc_report_fail(name, world_rank,
                       "Part A: persistent_coll_count expected 1 after init");
        v = TC_FAIL;
    }

    UCC_CHECK(ucc_collective_post(reqA));
    progress_until(ctx, reqA, "persistent allreduce (Part A)");
    if (rbuf != expected) {
        std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                  << ": Part A allreduce got " << rbuf << " (exp " << expected
                  << ")\n";
        v = TC_FAIL;
    }

    UCC_CHECK(ucc_collective_finalize(reqA));
    if (((ucc_team_t *)tA)->persistent_coll_count != 0) {
        tc_report_fail(name, world_rank,
                       "Part A: persistent_coll_count expected 0 after "
                       "finalize");
        v = TC_FAIL;
    }

    /* With count == 0, destroy must admit the team as DORMANT. */
    destroy_ucc_team(tA, ctx);
    if (!is_dormant(cache, tA_addr)) {
        tc_report_fail(name, world_rank,
                       "Part A: team must be DORMANT after destroy with "
                       "count==0");
        v = TC_FAIL;
    }

    /* Re-create the identical team: must be a DORMANT hit. */
    hits_before = cache->stats.hits;
    tA2         = create_world_team(ctx, world_size);
    if (cache->stats.hits <= hits_before) {
        tc_report_fail(name, world_rank,
                       "Part A: recreate after DORMANT must be a cache hit");
        v = TC_FAIL;
    }
    run_barrier_on_team(tA2, ctx);
    destroy_ucc_team(tA2, ctx);
    MPI_Barrier(MPI_COMM_WORLD);

    /* --- Part B: outstanding persistent handle blocks DORMANT admission --- */
    sbuf    = (int64_t)world_rank;
    rbuf    = 0;
    tB      = create_world_team(ctx, world_size);
    tB_addr = (uintptr_t)tB;

    fill_allreduce_int64(&args, &sbuf, &rbuf, UCC_COLL_ARGS_FLAG_PERSISTENT);
    UCC_CHECK(ucc_collective_init(&args, &reqB, tB));

    if (((ucc_team_t *)tB)->persistent_coll_count != 1) {
        tc_report_fail(name, world_rank,
                       "Part B: persistent_coll_count expected 1 after init");
        v = TC_FAIL;
    }

    /* Destroy the team while the persistent handle is still outstanding. The
       bypass guard must fire: real teardown, not DORMANT admission. reqB is
       intentionally not finalized - doing so after the team is freed is
       undefined behaviour. The task struct is a bounded leak. */
    destroy_ucc_team(tB, ctx);
    if (is_dormant(cache, tB_addr)) {
        tc_report_fail(name, world_rank,
                       "Part B: team must not be on the dormant list after the "
                       "bypass");
        v = TC_FAIL;
    }

    /* Recreate the identical team: must be a MISS (no dormant entry). */
    misses_before = cache->stats.misses;
    tB2           = create_world_team(ctx, world_size);
    tB2_addr      = (uintptr_t)tB2;
    if (cache->stats.misses <= misses_before) {
        tc_report_fail(name, world_rank,
                       "Part B: recreate after bypass teardown must be a cache "
                       "miss");
        v = TC_FAIL;
    }
    run_barrier_on_team(tB2, ctx);

    /* Now destroy normally (no outstanding handles) -> DORMANT. */
    destroy_ucc_team(tB2, ctx);
    if (!is_dormant(cache, tB2_addr)) {
        tc_report_fail(name, world_rank,
                       "Part B: tB2 must be DORMANT after a normal destroy");
        v = TC_FAIL;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (v == TC_PASS && 0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
    return v;
}

/* Reduce a per-rank verdict to a suite-wide one: any FAIL makes the test a
   failure, otherwise any SKIP makes it a skip. */
static tc_verdict_t tc_reduce(tc_verdict_t local)
{
    int flags[2];

    flags[0] = (local == TC_FAIL) ? 1 : 0;
    flags[1] = (local == TC_SKIP) ? 1 : 0;
    MPI_Allreduce(MPI_IN_PLACE, flags, 2, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (flags[0]) {
        return TC_FAIL;
    }
    return flags[1] ? TC_SKIP : TC_PASS;
}

static void tc_tally(ucc_test_suite_result_t *r, tc_verdict_t local)
{
    switch (tc_reduce(local)) {
    case TC_PASS:
        r->passed++;
        break;
    case TC_FAIL:
        r->failed++;
        break;
    default:
        r->skipped++;
        break;
    }
    MPI_Barrier(MPI_COMM_WORLD);
}

ucc_test_suite_result_t run_team_cache_tests(ucc_context_h ctx, int world_rank,
                                             int world_size)
{
    /* Must match the number of tc_tally calls below: used to report every test
       as skipped when caching is off, so a disabled run is never mistaken for a
       clean one. */
    const int               kNumTests = 11;
    ucc_test_suite_result_t r         = {0, 0, 0};

    if (0 == world_rank) {
        std::cout << "\n===== UCC Team Cache Correctness Tests =====\n";
    }

    /* These tests require caching to be enabled. If UCC_TEAM_CACHE_ENABLE was
       not set, report the whole suite as skipped rather than let a test
       MPI_Abort the job. */
    if (cache_of(ctx) == NULL) {
        if (0 == world_rank) {
            std::cout << "SKIP all team-cache tests: caching disabled "
                         "(set UCC_TEAM_CACHE_ENABLE=y)\n"
                      << "===== Team Cache Tests DONE =====\n";
        }
        r.skipped = kNumTests;
        return r;
    }

    tc_tally(&r, test_dup_coexist_derived(ctx, world_rank, world_size, false));
    tc_tally(&r, test_dup_coexist_derived(ctx, world_rank, world_size, true));
    tc_tally(&r, test_dormant_reuse_stats(ctx, world_rank, world_size));
    tc_tally(&r, test_derived_reuse(ctx, world_rank, world_size, false));
    tc_tally(&r, test_derived_reuse(ctx, world_rank, world_size, true));
    tc_tally(&r, test_ep_map_cb_freed_after_cache(ctx, world_rank, world_size));
    tc_tally(&r, test_overlap_agreement(ctx, world_rank, world_size));
    tc_tally(&r, test_derived_exact_rebuild(ctx, world_rank, world_size));
    tc_tally(&r, test_nonblocking_create_post(ctx, world_rank, world_size));
    tc_tally(&r, test_singleton_team(ctx, world_rank, world_size));
    tc_tally(&r, test_persistent_handle_safety(ctx, world_rank, world_size));

    ucc_assert(r.passed + r.failed + r.skipped == kNumTests);

    if (0 == world_rank) {
        std::cout << "===== Team Cache Tests DONE (" << r.passed << " passed, "
                  << r.failed << " failed, " << r.skipped << " skipped) =====\n"
                  << std::endl;
    }
    return r;
}
