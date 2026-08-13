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
        ucc_rank_t got = ucc_ep_map_eval(t->ctx_map, e);

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
    const int               kNumTests = 3;
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

    tc_tally(&r, test_dormant_reuse_stats(ctx, world_rank, world_size));
    tc_tally(&r, test_ep_map_cb_freed_after_cache(ctx, world_rank, world_size));
    tc_tally(&r, test_singleton_team(ctx, world_rank, world_size));

    ucc_assert(r.passed + r.failed + r.skipped == kNumTests);

    if (0 == world_rank) {
        std::cout << "===== Team Cache Tests DONE (" << r.passed << " passed, "
                  << r.failed << " failed, " << r.skipped << " skipped) =====\n"
                  << std::endl;
    }
    return r;
}
