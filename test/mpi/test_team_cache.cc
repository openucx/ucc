/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

/*
 * Multi-rank team-cache correctness tests. These cover behaviors that need real
 * multi-rank OOB collectives: derived-team dup coexistence, dormant-reuse hit
 * counting, cid-drift re-adoption, freed-callback safety, divergent-eviction
 * agreement, non-blocking create, and singleton teams. Single-process behavior
 * and cache on-vs-off equivalence are covered by the gtest suite and by
 * run_cache_equivalence.sh. Run via ucc_test_mpi with UCC_TEAM_CACHE_ENABLE=y;
 * teams are created and destroyed directly so the recreate cycle is controlled
 * per test.
 */

#include "test_mpi.h"
#include "mpi_util.h"
#include "core/ucc_context.h"
#include "core/ucc_team.h"
#include "core/ucc_team_cache.h"
#include "utils/ucc_coll_utils.h"
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <unistd.h> /* usleep for the nonblocking-create-post timing test */

/* create/destroy/recreate cycles used by the reuse tests. */
static const int kReuseIters = 5;

/* Read per-context team-cache counters directly (no public stats attr). Returns
   false if caching is disabled on this context. */
static bool      get_cache_stats(
         ucc_context_h ctx, uint64_t *hits, uint64_t *inserts, uint64_t *evictions)
{
    ucc_context_t *c = (ucc_context_t *)ctx;
    if (c->team_cache == NULL) {
        return false;
    }
    if (hits) {
        *hits = c->team_cache->stats.hits;
    }
    if (inserts) {
        *inserts = c->team_cache->stats.inserts;
    }
    if (evictions) {
        *evictions = c->team_cache->stats.evictions;
    }
    return true;
}

/* OOB allgather callbacks over an MPI communicator. */
static ucc_status_t oob_allgather_cb(
    void *sbuf, void *rbuf, size_t msglen, void *coll_info, void **req)
{
    MPI_Comm    comm = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(
        sbuf, msglen, MPI_BYTE, rbuf, msglen, MPI_BYTE, comm, &request);
    *req = (void *)(uintptr_t)request;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test_cb(void *req)
{
    MPI_Request request = (MPI_Request)(uintptr_t)req;
    int         completed;
    MPI_Test(&request, &completed, MPI_STATUS_IGNORE);
    return completed ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free_cb(void *req)
{
    return UCC_OK;
}

/* Pump the context until a single collective request completes; abort on error. */
static void progress_until(
    ucc_context_h ctx, ucc_coll_req_h req, const char *what)
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

/* Fill common team-create params (OOB over @comm, the given ep_map, optional
   external id). Shared by create_team and the nonblocking-create-post test. */
static void fill_team_params(
    ucc_team_params_t *p, MPI_Comm comm, ucc_ep_map_t ep_map, uint64_t ext_id)
{
    int rank, size;

    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    memset(p, 0, sizeof(*p));
    p->mask = UCC_TEAM_PARAM_FIELD_EP | UCC_TEAM_PARAM_FIELD_EP_RANGE |
              UCC_TEAM_PARAM_FIELD_OOB | UCC_TEAM_PARAM_FIELD_EP_MAP;
    p->oob.allgather = oob_allgather_cb;
    p->oob.req_test  = oob_allgather_test_cb;
    p->oob.req_free  = oob_allgather_free_cb;
    p->oob.coll_info = (void *)(uintptr_t)comm;
    p->oob.n_oob_eps = size;
    p->oob.oob_ep    = rank;
    p->ep            = rank;
    p->ep_range      = UCC_COLLECTIVE_EP_RANGE_CONTIG;
    p->ep_map        = ep_map;
    if (ext_id != 0) {
        p->mask |= UCC_TEAM_PARAM_FIELD_ID;
        p->id = ext_id;
    }
}

/* Blocking team create over @comm with the given ep_map (and optional external
   id); pumps the context to completion and aborts on error. */
static ucc_team_h create_team(
    ucc_context_h ctx, MPI_Comm comm, ucc_ep_map_t ep_map, uint64_t ext_id)
{
    ucc_team_params_t p;
    ucc_team_h        team;
    ucc_status_t      status;

    fill_team_params(&p, comm, ep_map, ext_id);
    UCC_CHECK(ucc_team_create_post(&ctx, 1, &p, &team));
    while (UCC_INPROGRESS == (status = ucc_team_create_test(team))) {
        ucc_context_progress(ctx);
    }
    if (status < 0) {
        std::cerr << "*** UCC TEST FAIL: team create ("
                  << ucc_status_string(status) << ")\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    return team;
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
static ucc_team_h create_world_team(
    ucc_context_h ctx, int size, uint64_t ext_id = 0)
{
    return create_team(ctx, MPI_COMM_WORLD, ep_map_full(size), ext_id);
}

/* Subset team over @comm using an explicit team-idx -> ctx-rank array map. */
static ucc_team_h create_array_team(
    ucc_context_h ctx, MPI_Comm comm, ucc_rank_t *map, int nmembers)
{
    ucc_ep_map_t m;
    memset(&m, 0, sizeof(m));
    m.type            = UCC_EP_MAP_ARRAY;
    m.ep_num          = nmembers;
    m.array.map       = map;
    m.array.elem_size = sizeof(ucc_rank_t);
    return create_team(ctx, comm, m, 0);
}

/* destroy_ucc_team - blocking ucc_team_destroy with progress pump. */
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

/* run_allreduce_int64 - SUM allreduce of a single int64 on @team, blocking;
   aborts on error. Returns the reduced value. */
static int64_t run_allreduce_int64(
    ucc_team_h team, ucc_context_h ctx, int64_t sbuf, const char *where)
{
    int64_t         rbuf = 0;
    ucc_coll_args_t args;
    ucc_coll_req_h  req;

    memset(&args, 0, sizeof(args));
    args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    args.op                = UCC_OP_SUM;
    args.src.info.buffer   = &sbuf;
    args.src.info.count    = 1;
    args.src.info.datatype = UCC_DT_INT64;
    args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    args.dst.info.buffer   = &rbuf;
    args.dst.info.count    = 1;
    args.dst.info.datatype = UCC_DT_INT64;
    args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    UCC_CHECK(ucc_collective_init(&args, &req, team));
    UCC_CHECK(ucc_collective_post(req));
    progress_until(ctx, req, where);
    UCC_CHECK(ucc_collective_finalize(req));
    return rbuf;
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
   completes so they are in flight together (and interleave across teams). */
static void drive_ar_bc(
    ucc_team_h ar_team, ucc_team_h bc_team, ucc_context_h ctx, int rank,
    int size, int iter, int order)
{
    int64_t         ar_send = 100 + iter;
    int64_t         ar_recv = 0;
    int64_t         ar_exp  = (int64_t)(100 + iter) * size;
    int64_t         bc_buf  = (rank == 0) ? (900000 + iter) : 0;
    int64_t         bc_exp  = 900000 + iter;

    ucc_coll_args_t ar_args, bc_args;
    ucc_coll_req_h  ar_req, bc_req;

    memset(&ar_args, 0, sizeof(ar_args));
    ar_args.coll_type         = UCC_COLL_TYPE_ALLREDUCE;
    ar_args.op                = UCC_OP_SUM;
    ar_args.src.info.buffer   = &ar_send;
    ar_args.src.info.count    = 1;
    ar_args.src.info.datatype = UCC_DT_INT64;
    ar_args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    ar_args.dst.info.buffer   = &ar_recv;
    ar_args.dst.info.count    = 1;
    ar_args.dst.info.datatype = UCC_DT_INT64;
    ar_args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

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
    ucc_status_t sa, sb;
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
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
}

static void test_dup_coexist_derived(
    ucc_context_h ctx, int world_rank, int world_size, bool ext_ids)
{
    const int      kIters     = ext_ids ? 8 : 6;
    const uint64_t parent_id  = ext_ids ? 100 : 0;
    const uint64_t derived_id = ext_ids ? 200 : 0;
    const char    *name       = ext_ids ? "dup_coexist_derived[ext_ids]"
                                        : "dup_coexist_derived";

    /* External-id dup needs a LIVE insertable parent to derive from; drain
       dormant squatters so the world-membership bucket is clean. */
    if (ext_ids) {
        MPI_Barrier(MPI_COMM_WORLD);
        ucc_team_cache_drain((ucc_context_t *)ctx);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    /* Parent world team (derivable), kept live. */
    ucc_team_h parent = create_world_team(ctx, world_size, parent_id);
    run_barrier_on_team(parent, ctx);

    /* Second identical-membership team while parent is live -> derived-create
       path (MPI_Comm_dup analogue), with a distinct external id under ext_ids. */
    ucc_team_h derived = create_world_team(ctx, world_size, derived_id);

    /* Distinct team ids: coexisting teams must not alias tag/seq domains. */
    if (parent->id == derived->id) {
        std::cerr << "*** UCC TEST FAIL: coexisting teams share id "
                  << parent->id << "\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    /* Must actually take the derived path (a full-create regression would still
       pass the distinct-id check above). */
    if (!derived->is_derived) {
        std::cerr << "*** UCC TEST FAIL: coexisting identical-membership team "
                     "was not derived (full-create regression)\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    /* External-id variant: derived must share the parent's artifacts holder. */
    if (ext_ids && parent->artifacts != derived->artifacts) {
        std::cerr << "*** UCC TEST FAIL: derived team did not share parent "
                     "artifacts holder\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    /* Interleaved-order collectives with opposite per-rank ordering across both
       live teams: even ranks allreduce(parent)+bcast(derived), odd the reverse. */
    for (int it = 0; it < kIters; it++) {
        int order = (world_rank % 2 == 0) ? 0 : 1;
        drive_ar_bc(parent, derived, ctx, world_rank, world_size, it, order);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    destroy_ucc_team(derived, ctx);
    run_barrier_on_team(parent, ctx); /* parent still usable */
    destroy_ucc_team(parent, ctx);

    if (0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
}

/* ==========================================================================
 * dormant_reuse_stats: build a cacheable world team, destroy it (-> dormant),
 * and recreate an identical one kReuseIters times. Each recreate after the first
 * must be a dormant HIT. Covers the non-derived dormant-reuse path;
 * test_derived_reuse covers the derived variant.
 * ========================================================================== */
static void test_dormant_reuse_stats(
    ucc_context_h ctx, int world_rank, int world_size)
{
    uint64_t hits0 = 0, hitsN = 0;

    if (!get_cache_stats(ctx, &hits0, NULL, NULL)) {
        if (0 == world_rank) {
            std::cout << "SKIP dormant_reuse_stats: cache disabled\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    /* Drain dormant squatters from prior tests (a leftover would skew hits). */
    MPI_Barrier(MPI_COMM_WORLD);
    ucc_team_cache_drain((ucc_context_t *)ctx);
    MPI_Barrier(MPI_COMM_WORLD);
    get_cache_stats(ctx, &hits0, NULL, NULL);

    for (int i = 0; i < kReuseIters; i++) {
        ucc_team_h team = create_world_team(ctx, world_size);
        run_barrier_on_team(team, ctx);
        MPI_Barrier(MPI_COMM_WORLD);
        destroy_ucc_team(team, ctx);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    get_cache_stats(ctx, &hitsN, NULL, NULL);
    /* First create is a miss+insert; the remaining recreates must be hits. */
    if (hitsN - hits0 < (uint64_t)(kReuseIters - 1)) {
        std::cerr << "*** UCC TEST FAIL: rank " << world_rank
                  << " expected >=" << (kReuseIters - 1)
                  << " dormant hits, got " << (hitsN - hits0) << "\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (0 == world_rank) {
        std::cout << "PASS dormant_reuse_stats\n";
    }
}

/* ==========================================================================
 * derived_reuse: a parent world team stays live while each iteration creates a
 * derived team, runs an allreduce, then frees it (-> dormant). From iter 1 the
 * dormant derived is re-adopted (cache HIT).
 *   - drift==false: the derived's external id is stable, so an exact-identity
 *     lookup re-adopts it.
 *   - drift==true : the derived's external id drifts every iteration, so only a
 *     membership-match re-adopt (reseat) can hit; requires
 *     UCC_TEAM_CACHE_RESEAT and UCC_TEAM_CACHE_DERIVED, else it skips.
 * ========================================================================== */
static void test_derived_reuse(
    ucc_context_h ctx, int world_rank, int world_size, bool drift)
{
    const char *name  = drift ? "derived_reuse[drift]" : "derived_reuse";
    uint64_t    hits0 = 0;

    if (!get_cache_stats(ctx, &hits0, NULL, NULL)) {
        if (0 == world_rank) {
            std::cout << "SKIP " << name << ": cache disabled\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }
    if (drift) {
        ucc_team_cache_t *cache = ((ucc_context_t *)ctx)->team_cache;
        if (!cache->reseat || !cache->derived) {
            if (0 == world_rank) {
                std::cout << "SKIP " << name
                          << ": UCC_TEAM_CACHE_RESEAT/DERIVED not enabled\n";
            }
            MPI_Barrier(MPI_COMM_WORLD);
            return;
        }
    }

    /* Drain any dormant squatters from prior tests. */
    MPI_Barrier(MPI_COMM_WORLD);
    ucc_team_cache_drain((ucc_context_t *)ctx);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Parent stays live throughout; its stable ext_id=1 is distinct from every
       derived id used below. */
    ucc_team_h parent = create_world_team(ctx, world_size, /*ext_id=*/1);
    run_barrier_on_team(parent, ctx);

    for (int i = 0; i < kReuseIters; i++) {
        uint64_t hits_before = 0;
        get_cache_stats(ctx, &hits_before, NULL, NULL);

        /* Derived: same membership as parent. Under drift the ext_id changes
           every iteration (100, 101, ...) so the exact-identity lookup misses
           and only a reseat re-adopt can hit; otherwise ext_id=2 is stable. */
        uint64_t   derived_id = drift ? (uint64_t)(100 + i) : 2;
        ucc_team_h derived    = create_world_team(ctx, world_size, derived_id);

        int64_t    sbuf       = (int64_t)(100 + i);
        int64_t    exp        = sbuf * (int64_t)world_size;
        int64_t    rbuf       = run_allreduce_int64(
            derived, ctx, sbuf, "allreduce on derived team");
        if (rbuf != exp) {
            std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                      << " iter " << i << ": allreduce got " << rbuf << " (exp "
                      << exp << ")\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }

        /* From iter 1 onwards: expect a cache hit re-adopting the dormant
           derived (via reseat under drift). */
        if (i > 0) {
            uint64_t hits_after = 0;
            get_cache_stats(ctx, &hits_after, NULL, NULL);
            if (hits_after <= hits_before) {
                std::cerr << "*** UCC TEST FAIL: " << name << " rank "
                          << world_rank << " iter " << i
                          << ": no cache hit for derived re-adopt\n";
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        destroy_ucc_team(derived, ctx); /* -> dormant */
        MPI_Barrier(MPI_COMM_WORLD);
    }

    run_barrier_on_team(parent, ctx); /* parent must still be functional */
    destroy_ucc_team(parent, ctx);

    if (0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
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
    uint64_t   magic; /* CB_CTX_MAGIC while live, poisoned after free */
    ucc_rank_t size;
    ucc_rank_t ranks[1]; /* flexible: team ep -> ctx rank (identity here) */
};
static const uint64_t CB_CTX_MAGIC = 0xC0FFEE5AULL;

static uint64_t       poisonable_rank_cb(uint64_t ep, void *cb_ctx)
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

static ucc_team_h create_cb_team(
    ucc_context_h ctx, int size, struct cb_ctx_box *box)
{
    ucc_ep_map_t m;
    memset(&m, 0, sizeof(m));
    m.type      = UCC_EP_MAP_CB;
    m.ep_num    = size;
    m.cb.cb     = poisonable_rank_cb;
    m.cb.cb_ctx = (void *)box;
    return create_team(ctx, MPI_COMM_WORLD, m, 0);
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
    box->magic             = CB_CTX_MAGIC;
    box->size              = (ucc_rank_t)world_size;
    for (int i = 0; i < world_size; i++) {
        box->ranks[i] = (ucc_rank_t)i; /* world identity mapping */
    }
    return box;
}

static void test_ep_map_cb_freed_after_cache(
    ucc_context_h ctx, int world_rank, int world_size)
{
    uint64_t hits_before = 0, hits_after = 0;
    bool     cache_on       = get_cache_stats(ctx, &hits_before, NULL, NULL);
    struct cb_ctx_box *box  = alloc_cb_box(world_size);

    /* First create + use + destroy -> team goes dormant (cache on). */
    ucc_team_h         team = create_cb_team(ctx, world_size, box);
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
    box              = alloc_cb_box(world_size);
    ucc_team_h team2 = create_cb_team(ctx, world_size, box);
    run_barrier_on_team(team2, ctx);

    /* Evaluate the re-adopted team's operational map for every endpoint - the
       exact access TL/UCP performs when resolving a peer. With the fix the map
       is UCC-owned, so this resolves correctly without calling back into the
       freed box; without it, poisonable_rank_cb aborts on the poison. */
    {
        ucc_team_t *t = (ucc_team_t *)team2;
        for (ucc_rank_t e = 0; e < (ucc_rank_t)world_size; e++) {
            ucc_rank_t got = ucc_ep_map_eval(UCC_TEAM_CTX_MAP(t), e);
            if (got != e) {
                std::cerr << "*** UCC TEST FAIL: operational ctx_map endpoint "
                          << (int)e << " resolved to " << (int)got
                          << " (expected " << (int)e << ")\n";
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    }

    destroy_ucc_team(team2, ctx);
    free(box);

    if (cache_on) {
        get_cache_stats(ctx, &hits_after, NULL, NULL);
        if (hits_after <= hits_before && world_rank == 0) {
            std::cerr << "*** UCC TEST WARN: cb_freed reuse recorded no cache "
                         "hit (identity may not have matched)\n";
        }
    }
    if (world_rank == 0) {
        std::cout << "PASS ep_map_cb_freed_after_cache\n";
    }
}

/* ==========================================================================
 * overlap_agreement: overlapping subcommunicators plus a small cache force
 * DIVERGENT per-rank eviction, which previously deadlocked (one rank re-adopts a
 * dormant team while a peer that evicted it enters a fresh collective build and
 * waits forever). The cross-rank agreement must reconcile the split hit/miss to
 * a consistent fresh build. Run with UCC_TEAM_CACHE_MAX_SIZE=2 and >=3 ranks.
 * ========================================================================== */
static void test_overlap_agreement(
    ucc_context_h ctx, int world_rank, int world_size)
{
    ucc_rank_t ranksA[2] = {0, 1};
    ucc_rank_t ranksB[2] = {1, 2};
    ucc_rank_t ranksD[3] = {0, 1, 2};
    MPI_Comm   commA, commB, commD;
    ucc_team_h t;

    if (world_size < 3) {
        if (world_rank == 0) {
            std::cout << "SKIP overlap_agreement (needs >=3 ranks)\n";
        }
        return;
    }

    /* Overlapping member sets: A{0,1}, B{1,2}, D{0,1,2}. */
    MPI_Comm_split(
        MPI_COMM_WORLD,
        (world_rank <= 1) ? 0 : MPI_UNDEFINED,
        world_rank,
        &commA);
    MPI_Comm_split(
        MPI_COMM_WORLD,
        (world_rank >= 1 && world_rank <= 2) ? 0 : MPI_UNDEFINED,
        world_rank,
        &commB);
    MPI_Comm_split(
        MPI_COMM_WORLD,
        (world_rank <= 2) ? 0 : MPI_UNDEFINED,
        world_rank,
        &commD);

    /* 1) A dormant on {0,1}; 2) B dormant on {1,2} (rank 1 cache now full at
       MAX_SIZE=2); 3) D on {0,1,2} evicts the oldest dormant (A) on rank 1 but
       not on rank 0 -> divergence; 4) re-create A: rank 0 re-adopts, rank 1
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
    }

    if (commA != MPI_COMM_NULL) {
        MPI_Comm_free(&commA);
    }
    if (commB != MPI_COMM_NULL) {
        MPI_Comm_free(&commB);
    }
    if (commD != MPI_COMM_NULL) {
        MPI_Comm_free(&commD);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (world_rank == 0) {
        std::cout << "PASS overlap_agreement\n";
    }
}

/* ==========================================================================
 * derived_exact_rebuild: a DORMANT derived team re-adopted via EXACT_REUSE that
 * then loses the cross-rank vote must rebuild as a proper FULL team - it must be
 * de-derived so it runs ADDR_EXCHANGE (builds ctx_map/topo) rather than skipping
 * it as a derived team would. Regression for a global-MISS rebuild that left an
 * EXACT_REUSE candidate marked is_derived, producing a NULL topo on rebuild.
 *
 * Setup (needs >=3 ranks and MAX_SIZE=2 to force divergent eviction): a live
 * parent world team (ext_id 1) plus a dormant derived world team (ext_id 2) fill
 * the cache on every rank. A subcomm team created only on ranks {0,1} evicts the
 * dormant derived there (FIFO), while ranks >=2 keep it. Re-creating the ext_id-2
 * world team then splits the vote (ranks >=2 EXACT_REUSE the derived candidate,
 * ranks {0,1} miss -> DERIVED_FROM_LIVE), forcing a global MISS and the in-place
 * rebuild on ranks >=2. Without the de-derive fix those ranks skip ADDR_EXCHANGE
 * and desync/crash; with it, the allreduce below completes and is_derived is 0.
 * ========================================================================== */
static void test_derived_exact_rebuild(
    ucc_context_h ctx, int world_rank, int world_size)
{
    const char       *name = "derived_exact_rebuild";
    ucc_team_cache_t *cache;

    if (!get_cache_stats(ctx, NULL, NULL, NULL)) {
        if (0 == world_rank) {
            std::cout << "SKIP " << name << ": cache disabled\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }
    cache = ((ucc_context_t *)ctx)->team_cache;
    if (!cache->derived || world_size < 3 || cache->max_size > 2) {
        if (0 == world_rank) {
            std::cout << "SKIP " << name
                      << ": needs derived caching, >=3 ranks, MAX_SIZE<=2\n";
        }
        MPI_Barrier(MPI_COMM_WORLD);
        return;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    ucc_team_cache_drain((ucc_context_t *)ctx);
    MPI_Barrier(MPI_COMM_WORLD);

    /* Live parent (ext_id 1) + dormant derived (ext_id 2): 2 entries = MAX_SIZE. */
    ucc_team_h parent = create_world_team(ctx, world_size, /*ext_id=*/1);
    run_barrier_on_team(parent, ctx);
    ucc_team_h derived = create_world_team(ctx, world_size, /*ext_id=*/2);
    run_barrier_on_team(derived, ctx);
    MPI_Barrier(MPI_COMM_WORLD);
    destroy_ucc_team(derived, ctx); /* -> dormant derived */
    MPI_Barrier(MPI_COMM_WORLD);

    /* Ranks {0,1} only: a subcomm team evicts the dormant derived (cache full);
       ranks >=2 never create it and keep the dormant derived. */
    MPI_Comm commAB;
    MPI_Comm_split(
        MPI_COMM_WORLD,
        (world_rank <= 1) ? 0 : MPI_UNDEFINED,
        world_rank,
        &commAB);
    if (commAB != MPI_COMM_NULL) {
        ucc_rank_t ab[2] = {0, 1};
        ucc_team_h t     = create_array_team(ctx, commAB, ab, 2);
        run_barrier_on_team(t, ctx);
        destroy_ucc_team(t, ctx);
        MPI_Comm_free(&commAB);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* Re-create the ext_id-2 world team: split vote -> global MISS -> rebuild.
       The allreduce exercises ctx_map/topo the rebuild must have populated. */
    ucc_team_h rebuilt = create_world_team(ctx, world_size, /*ext_id=*/2);
    int64_t    got     = run_allreduce_int64(
        rebuilt, ctx, (int64_t)world_rank, "allreduce on rebuilt team");
    int64_t exp = (int64_t)world_size * (world_size - 1) / 2;
    if (got != exp) {
        std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                  << ": allreduce got " << got << " (exp " << exp << ")\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    if (((ucc_team_t *)rebuilt)->is_derived != 0 ||
        ((ucc_team_t *)rebuilt)->parent_id != 0) {
        std::cerr << "*** UCC TEST FAIL: " << name << " rank " << world_rank
                  << ": rebuilt team still marked derived (is_derived="
                  << ((ucc_team_t *)rebuilt)->is_derived << ")\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    destroy_ucc_team(rebuilt, ctx);
    run_barrier_on_team(parent, ctx); /* parent still functional */
    destroy_ucc_team(parent, ctx);
    ucc_team_cache_drain((ucc_context_t *)ctx);

    if (0 == world_rank) {
        std::cout << "PASS " << name << "\n";
    }
}

/* ==========================================================================
 * nonblocking_create_post: ucc_team_create_post must return promptly (post the
 * vote, not block) even if one rank is late entering it. Rank 0 sleeps briefly;
 * the other ranks call create_post and must return before rank 0 arrives.
 * ========================================================================== */
static void test_nonblocking_create_post(
    ucc_context_h ctx, int world_rank, int world_size)
{
    ucc_team_params_t p;
    ucc_team_h        team;
    ucc_status_t      status;
    /* Rank 0 sleeps kSleepMs; a non-blocking create_post on a peer returns in
       milliseconds, a blocking one waits ~kSleepMs. Use a large sleep and a
       threshold at half of it so oversubscribed scheduling jitter on the peer
       (well under kThreshMs) cannot false-fail, while a real block (~kSleepMs) is
       still caught. */
    const int         kSleepMs  = 2000;
    const int         kThreshMs = kSleepMs / 2;

    if (world_size < 2) {
        if (0 == world_rank) {
            std::cout << "SKIP nonblocking_create_post (needs >=2 ranks)\n";
        }
        return;
    }

    /* Drain so this is a clean fresh create (the vote is posted regardless). */
    MPI_Barrier(MPI_COMM_WORLD);
    ucc_team_cache_drain((ucc_context_t *)ctx);
    MPI_Barrier(MPI_COMM_WORLD);

    double t_start = MPI_Wtime();
    if (world_rank == 0) {
        /* Delay entering create_post: peers must not be blocked waiting on us. */
        usleep(kSleepMs * 1000);
    }

    fill_team_params(&p, MPI_COMM_WORLD, ep_map_full(world_size), 0);
    UCC_CHECK(ucc_team_create_post(&ctx, 1, &p, &team));
    double t_post = MPI_Wtime();

    /* On a non-zero rank, create_post must have returned well before rank 0's
       sleep elapsed - proving it posted (did not block on) the vote. */
    if (world_rank != 0) {
        double elapsed_ms = (t_post - t_start) * 1000.0;
        if (elapsed_ms > (double)kThreshMs) {
            std::cerr << "*** UCC TEST FAIL: nonblocking_create_post rank "
                      << world_rank << " create_post blocked " << elapsed_ms
                      << "ms (threshold " << kThreshMs << "ms, rank 0 delay "
                      << kSleepMs << "ms)\n";
            MPI_Abort(MPI_COMM_WORLD, -1);
        }
    }

    /* Now drive the create to completion collectively and use it. */
    while (UCC_INPROGRESS == (status = ucc_team_create_test(team))) {
        ucc_context_progress(ctx);
    }
    if (status < 0) {
        std::cerr << "*** UCC TEST FAIL: nonblocking_create_post create ("
                  << ucc_status_string(status) << ")\n";
        MPI_Abort(MPI_COMM_WORLD, -1);
    }
    run_barrier_on_team(team, ctx);
    MPI_Barrier(MPI_COMM_WORLD);
    destroy_ucc_team(team, ctx);
    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == world_rank) {
        std::cout << "PASS nonblocking_create_post\n";
    }
}

/* ==========================================================================
 * singleton_team: a size-1 cacheable team creates + reuses correctly with no
 * network vote (self-membership; the size>1 gate is not taken). Each rank builds
 * its own {self} team independently over MPI_COMM_SELF.
 * ========================================================================== */
static void test_singleton_team(
    ucc_context_h ctx, int world_rank, int world_size)
{
    ucc_rank_t self_map[1];

    for (int i = 0; i < 3; i++) {
        self_map[0]  = (ucc_rank_t)world_rank; /* team idx 0 -> my ctx rank */
        ucc_team_h t = create_array_team(ctx, MPI_COMM_SELF, self_map, 1);
        run_barrier_on_team(t, ctx);
        destroy_ucc_team(t, ctx);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (0 == world_rank) {
        std::cout << "PASS singleton_team\n";
    }
}

void run_team_cache_tests(ucc_context_h ctx, int world_rank, int world_size)
{
    if (0 == world_rank) {
        std::cout << "\n===== UCC Team Cache Correctness Tests =====\n";
    }

    /* These tests require caching to be enabled (they assert reuse/derivation that
       only happens when the cache is live). If UCC_TEAM_CACHE_ENABLE was not set,
       skip the whole suite rather than let a test MPI_Abort the job. */
    if (((ucc_context_t *)ctx)->team_cache == NULL) {
        if (0 == world_rank) {
            std::cout << "SKIP all team-cache tests: caching disabled "
                         "(set UCC_TEAM_CACHE_ENABLE=y)\n"
                      << "===== Team Cache Tests DONE =====\n";
        }
        return;
    }

    test_dup_coexist_derived(ctx, world_rank, world_size, /*ext_ids=*/false);
    MPI_Barrier(MPI_COMM_WORLD);
    test_dup_coexist_derived(ctx, world_rank, world_size, /*ext_ids=*/true);
    MPI_Barrier(MPI_COMM_WORLD);

    test_dormant_reuse_stats(ctx, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    test_derived_reuse(ctx, world_rank, world_size, /*drift=*/false);
    MPI_Barrier(MPI_COMM_WORLD);
    test_derived_reuse(ctx, world_rank, world_size, /*drift=*/true);
    MPI_Barrier(MPI_COMM_WORLD);

    test_ep_map_cb_freed_after_cache(ctx, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    test_overlap_agreement(ctx, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    test_derived_exact_rebuild(ctx, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    test_nonblocking_create_post(ctx, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    test_singleton_team(ctx, world_rank, world_size);
    MPI_Barrier(MPI_COMM_WORLD);

    if (0 == world_rank) {
        std::cout << "===== Team Cache Tests DONE =====\n\n";
    }
}
