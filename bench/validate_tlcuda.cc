/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * Standalone correctness validator for TL/CUDA alltoall and alltoallv.
 *
 * Forces UCC_TLS=cuda so only the CUDA transport layer is exercised.
 * Registers each rank's dst buffer with ucc_mem_map(EXPORT), allgathers
 * the handles to build global_memh, and passes it to UCC — this is exactly
 * the path that triggers the push algorithm (alltoall_push / alltoallv_push).
 *
 * Usage:
 *   mpirun -n <N> ./validate_tlcuda [-v]
 *
 *   -v   verbose: print pass/skip per message size
 *
 * The program exits 0 on success, 1 on any correctness failure.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <random>
#include <mpi.h>
#include <cuda_runtime.h>
#include <ucc/api/ucc.h>

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "[rank %d] CUDA error %s:%d: %s\n",            \
                    g_rank, __FILE__, __LINE__, cudaGetErrorString(_e));    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

#define UCC_CHECK(call)                                                     \
    do {                                                                    \
        ucc_status_t _s = (call);                                           \
        if (_s != UCC_OK) {                                                 \
            fprintf(stderr, "[rank %d] UCC error %s:%d: %s (%d)\n",        \
                    g_rank, __FILE__, __LINE__, ucc_status_string(_s),      \
                    (int)_s);                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

#define MPI_CHECK(call)                                                     \
    do {                                                                    \
        int _e = (call);                                                    \
        if (_e != MPI_SUCCESS) {                                            \
            fprintf(stderr, "[rank %d] MPI error %s:%d code=%d\n",         \
                    g_rank, __FILE__, __LINE__, _e);                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

static int g_rank  = 0;
static int g_nranks = 0;
static bool g_verbose = false;

static int g_fail_count = 0;
static int g_pass_count = 0;
static int g_skip_count = 0;

/* ------------------------------------------------------------------ */
/* UCC OOB allgather (required for team creation)                      */
/* ------------------------------------------------------------------ */

static ucc_status_t oob_allgather(void *sbuf, void *rbuf, size_t msglen,
                                  void *coll_info, void **req)
{
    MPI_Comm    comm    = (MPI_Comm)(uintptr_t)coll_info;
    MPI_Request request;
    MPI_Iallgather(sbuf, (int)msglen, MPI_BYTE,
                   rbuf, (int)msglen, MPI_BYTE, comm, &request);
    *req = (void *)(uintptr_t)request;
    return UCC_OK;
}

static ucc_status_t oob_allgather_test(void *req)
{
    MPI_Request request = (MPI_Request)(uintptr_t)req;
    int         done    = 0;
    MPI_Test(&request, &done, MPI_STATUS_IGNORE);
    return done ? UCC_OK : UCC_INPROGRESS;
}

static ucc_status_t oob_allgather_free(void *req)
{
    (void)req;
    return UCC_OK;
}

/* ------------------------------------------------------------------ */
/* UCC context / team                                                  */
/* ------------------------------------------------------------------ */

struct UccState {
    ucc_lib_h     lib;
    ucc_context_h ctx;
    ucc_team_h    team;
};

static UccState setup_ucc(MPI_Comm comm)
{
    UccState      s;
    ucc_status_t  status;
    int           rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    ucc_lib_config_h lib_cfg;
    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_cfg));
    /* TLS and tuning controlled via UCC_* env vars (e.g. UCC_TLS=cuda,ucp) */

    ucc_lib_params_t lib_params = {};
    lib_params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    UCC_CHECK(ucc_init(&lib_params, lib_cfg, &s.lib));
    ucc_lib_config_release(lib_cfg);

    ucc_context_config_h ctx_cfg;
    UCC_CHECK(ucc_context_config_read(s.lib, NULL, &ctx_cfg));

    ucc_context_params_t ctx_params = {};
    ctx_params.mask             = UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.oob.allgather    = oob_allgather;
    ctx_params.oob.req_test     = oob_allgather_test;
    ctx_params.oob.req_free     = oob_allgather_free;
    ctx_params.oob.coll_info    = (void *)(uintptr_t)comm;
    ctx_params.oob.n_oob_eps    = (uint32_t)size;
    ctx_params.oob.oob_ep       = (uint32_t)rank;
    UCC_CHECK(ucc_context_create(s.lib, &ctx_params, ctx_cfg, &s.ctx));
    ucc_context_config_release(ctx_cfg);

    ucc_team_params_t team_params = {};
    team_params.mask          = UCC_TEAM_PARAM_FIELD_EP      |
                                UCC_TEAM_PARAM_FIELD_EP_RANGE |
                                UCC_TEAM_PARAM_FIELD_OOB;
    team_params.oob.allgather = oob_allgather;
    team_params.oob.req_test  = oob_allgather_test;
    team_params.oob.req_free  = oob_allgather_free;
    team_params.oob.coll_info = (void *)(uintptr_t)comm;
    team_params.oob.n_oob_eps = (uint32_t)size;
    team_params.oob.oob_ep    = (uint32_t)rank;
    team_params.ep            = (uint64_t)rank;
    team_params.ep_range      = UCC_COLLECTIVE_EP_RANGE_CONTIG;
    UCC_CHECK(ucc_team_create_post(&s.ctx, 1, &team_params, &s.team));

    do {
        ucc_context_progress(s.ctx);
        status = ucc_team_create_test(s.team);
    } while (status == UCC_INPROGRESS);
    UCC_CHECK(status);

    return s;
}

static void teardown_ucc(UccState &s)
{
    ucc_status_t status;
    do {
        status = ucc_team_destroy(s.team);
    } while (status == UCC_INPROGRESS);
    ucc_context_destroy(s.ctx);
    ucc_finalize(s.lib);
}

/* ------------------------------------------------------------------ */
/* Build global_memh: export dst buf, allgather handles                */
/* ------------------------------------------------------------------ */

struct GlobalMemh {
    ucc_mem_map_mem_h  local_memh;  /* our own exported handle */
    ucc_mem_map_mem_h *handles;     /* handles[i]: exported (self) or imported (peers) */
    size_t             memh_size;
    int                nranks;
    int                rank;
};

/*
 * Build a global_memh array by:
 *  1. Exporting the local dst buffer.
 *  2. Allgathering all exported handles as raw bytes.
 *  3. Importing each peer's raw bytes — this reconstructs the tl_h/tl_data
 *     pointers from the pack_buffer (the flat serialized TL data), which is
 *     what makes the handle valid in the calling process's address space.
 *     Simply using the raw bytes directly crashes because the exported struct
 *     contains a tl_h pointer from the exporting process's heap.
 */
static GlobalMemh build_global_memh(ucc_context_h ctx, void *buf, size_t len,
                                    MPI_Comm comm)
{
    GlobalMemh g;
    MPI_Comm_size(comm, &g.nranks);
    MPI_Comm_rank(comm, &g.rank);

    ucc_mem_map_t        seg        = { buf, len };
    ucc_mem_map_params_t map_params = { &seg, 1 };
    UCC_CHECK(ucc_mem_map(ctx, UCC_MEM_MAP_MODE_EXPORT,
                          &map_params, &g.memh_size, &g.local_memh));

    /* Gather all exported handles as raw bytes */
    void *all_bytes = malloc((size_t)g.nranks * g.memh_size);
    assert(all_bytes);
    MPI_CHECK(MPI_Allgather(g.local_memh,  (int)g.memh_size, MPI_BYTE,
                            all_bytes,     (int)g.memh_size, MPI_BYTE, comm));

    g.handles = (ucc_mem_map_mem_h *)malloc((size_t)g.nranks *
                                            sizeof(ucc_mem_map_mem_h));
    assert(g.handles);

    for (int i = 0; i < g.nranks; i++) {
        if (i == g.rank) {
            g.handles[i] = g.local_memh; /* already valid */
        } else {
            /* Copy the raw bytes into a fresh buffer, then import in-place.
             * ucc_mem_map(IMPORT) reads the pack_buffer that was serialized
             * by the exporter, allocates a new local tl_h array, and fills
             * tl_h[j].tl_data from the packed data — making the handle valid. */
            void *peer_buf = malloc(g.memh_size);
            assert(peer_buf);
            memcpy(peer_buf, (char *)all_bytes + (size_t)i * g.memh_size,
                   g.memh_size);
            g.handles[i] = peer_buf;

            ucc_mem_map_params_t import_params = { NULL, 0 };
            size_t               dummy         = 0;
            UCC_CHECK(ucc_mem_map(ctx, UCC_MEM_MAP_MODE_IMPORT,
                                  &import_params, &dummy, &g.handles[i]));
        }
    }

    free(all_bytes);
    return g;
}

static void free_global_memh(GlobalMemh &g)
{
    for (int i = 0; i < g.nranks; i++) {
        if (i != g.rank)
            ucc_mem_unmap(&g.handles[i]);
    }
    ucc_mem_unmap(&g.local_memh);
    free(g.handles);
}

/* ------------------------------------------------------------------ */
/* Progress UCC until request completes                                */
/* ------------------------------------------------------------------ */

static void run_collective(ucc_context_h ctx, ucc_coll_req_h req)
{
    ucc_status_t status;
    UCC_CHECK(ucc_collective_post(req));
    do {
        ucc_context_progress(ctx);
        status = ucc_collective_test(req);
    } while (status == UCC_INPROGRESS);
    UCC_CHECK(status);
    UCC_CHECK(ucc_collective_finalize(req));
}

/* ------------------------------------------------------------------ */
/* Alltoall test                                                        */
/* ------------------------------------------------------------------ */

/*
 * Fill pattern: sbuf[peer * chunk + i] = (uint32_t)(rank << 16 | i)
 * Expected in rbuf[peer * chunk + i]  = (uint32_t)(peer << 16 | i)
 * (rank `rank` sent data stamped with its own rank to peer)
 */
static bool test_alltoall(UccState &s, MPI_Comm comm, size_t msg_bytes)
{
    int          nranks    = g_nranks;
    int          rank      = g_rank;
    size_t       chunk     = msg_bytes;            /* per-peer send size in bytes */
    size_t       total_s   = chunk * (size_t)nranks;
    size_t       nelems    = chunk / sizeof(uint32_t);

    if (chunk % sizeof(uint32_t) != 0 || nelems == 0)
        return true; /* skip */

    /* Host buffers for init/check */
    std::vector<uint32_t> h_sbuf(nelems * nranks);
    std::vector<uint32_t> h_rbuf(nelems * nranks, 0);

    for (int peer = 0; peer < nranks; peer++) {
        for (size_t i = 0; i < nelems; i++) {
            h_sbuf[(size_t)peer * nelems + i] =
                (uint32_t)(((uint32_t)rank << 16) | (uint32_t)(i & 0xFFFF));
        }
    }

    /* CUDA device buffers */
    void *d_sbuf, *d_rbuf;
    CUDA_CHECK(cudaMalloc(&d_sbuf, total_s));
    CUDA_CHECK(cudaMalloc(&d_rbuf, total_s));
    CUDA_CHECK(cudaMemcpy(d_sbuf, h_sbuf.data(), total_s, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rbuf, 0, total_s));

    /* Build global_memh for dst */
    GlobalMemh gmemh = build_global_memh(s.ctx, d_rbuf, total_s, comm);

    /* Set up alltoall args */
    ucc_coll_args_t args = {};
    args.mask                    = UCC_COLL_ARGS_FIELD_FLAGS    |
                                   UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;
    args.flags                   = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                                   UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER |
                                   UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL;
    args.coll_type               = UCC_COLL_TYPE_ALLTOALL;
    args.src.info.buffer         = d_sbuf;
    args.src.info.count          = (uint64_t)(nelems * (size_t)nranks);
    args.src.info.datatype       = UCC_DT_UINT32;
    args.src.info.mem_type       = UCC_MEMORY_TYPE_CUDA;
    args.dst.info.buffer         = d_rbuf;
    args.dst.info.count          = (uint64_t)(nelems * (size_t)nranks);
    args.dst.info.datatype       = UCC_DT_UINT32;
    args.dst.info.mem_type       = UCC_MEMORY_TYPE_CUDA;
    args.dst_memh.global_memh    = gmemh.handles;

    ucc_coll_req_h req;
    ucc_status_t   init_status = ucc_collective_init(&args, &req, s.team);
    if (init_status == UCC_ERR_NOT_SUPPORTED ||
        init_status == UCC_ERR_NOT_IMPLEMENTED) {
        if (g_verbose && rank == 0)
            printf("  alltoall %zu B  SKIP (not supported)\n", msg_bytes);
        free_global_memh(gmemh);
        CUDA_CHECK(cudaFree(d_sbuf));
        CUDA_CHECK(cudaFree(d_rbuf));
        g_skip_count++;
        return true;
    }
    UCC_CHECK(init_status);

    run_collective(s.ctx, req);

    /* Validate */
    CUDA_CHECK(cudaMemcpy(h_rbuf.data(), d_rbuf, total_s, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int peer = 0; peer < nranks && ok; peer++) {
        for (size_t i = 0; i < nelems && ok; i++) {
            uint32_t expected = (uint32_t)(((uint32_t)peer << 16) | (uint32_t)(i & 0xFFFF));
            uint32_t got      = h_rbuf[(size_t)peer * nelems + i];
            if (got != expected) {
                fprintf(stderr, "[rank %d] alltoall FAIL size=%zu: "
                        "rbuf[peer=%d][%zu] = 0x%08x, expected 0x%08x\n",
                        rank, msg_bytes, peer, i, got, expected);
                ok = false;
            }
        }
    }

    if (g_verbose && rank == 0)
        printf("  alltoall  %7zu B  %s\n", msg_bytes, ok ? "PASS" : "FAIL");

    free_global_memh(gmemh);
    CUDA_CHECK(cudaFree(d_sbuf));
    CUDA_CHECK(cudaFree(d_rbuf));
    return ok;
}

/* ------------------------------------------------------------------ */
/* Alltoallv test                                                       */
/* ------------------------------------------------------------------ */

static bool test_alltoallv(UccState &s, MPI_Comm comm, size_t msg_bytes)
{
    int    nranks  = g_nranks;
    int    rank    = g_rank;
    size_t dt_size = sizeof(uint32_t);

    if (msg_bytes < dt_size)
        return true; /* skip */

    size_t max_per_peer = msg_bytes / dt_size;

    /* Random counts seeded identically across ranks */
    std::default_random_engine rng(42);
    std::uniform_int_distribution<int> dist(1, (int)max_per_peer);

    std::vector<int> scounts(nranks), rcounts(nranks);
    std::vector<int> sdispls(nranks), rdispls(nranks);

    for (int i = 0; i < nranks; i++)
        scounts[i] = dist(rng);

    MPI_CHECK(MPI_Alltoall(scounts.data(), 1, MPI_INT,
                           rcounts.data(), 1, MPI_INT, comm));

    size_t sncounts = 0, rncounts = 0;
    for (int i = 0; i < nranks; i++) {
        sdispls[i]  = (int)sncounts;
        rdispls[i]  = (int)rncounts;
        sncounts   += (size_t)scounts[i];
        rncounts   += (size_t)rcounts[i];
    }

    size_t sbytes = sncounts * dt_size;
    size_t rbytes = rncounts * dt_size;

    /* Host init: sbuf[sdispls[peer]+i] = (rank << 16) | (sdispls[peer]+i) */
    std::vector<uint32_t> h_sbuf(sncounts);
    std::vector<uint32_t> h_rbuf(rncounts, 0);

    for (int peer = 0; peer < nranks; peer++) {
        for (int i = 0; i < scounts[peer]; i++) {
            int pos = sdispls[peer] + i;
            h_sbuf[pos] = (uint32_t)(((uint32_t)rank << 16) | (uint32_t)(pos & 0xFFFF));
        }
    }

    void *d_sbuf, *d_rbuf;
    CUDA_CHECK(cudaMalloc(&d_sbuf, sbytes));
    CUDA_CHECK(cudaMalloc(&d_rbuf, rbytes));
    CUDA_CHECK(cudaMemcpy(d_sbuf, h_sbuf.data(), sbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_rbuf, 0, rbytes));

    /* Build global_memh for dst */
    GlobalMemh gmemh = build_global_memh(s.ctx, d_rbuf, rbytes, comm);

    /* UCC args */
    ucc_coll_args_t args = {};
    args.mask                        = UCC_COLL_ARGS_FIELD_FLAGS    |
                                       UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;
    args.flags                       = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                                       UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER |
                                       UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL;
    args.coll_type                   = UCC_COLL_TYPE_ALLTOALLV;
    args.src.info_v.buffer           = d_sbuf;
    args.src.info_v.counts           = (ucc_count_t *)scounts.data();
    args.src.info_v.displacements    = (ucc_aint_t  *)sdispls.data();
    args.src.info_v.datatype         = UCC_DT_UINT32;
    args.src.info_v.mem_type         = UCC_MEMORY_TYPE_CUDA;
    args.dst.info_v.buffer           = d_rbuf;
    args.dst.info_v.counts           = (ucc_count_t *)rcounts.data();
    args.dst.info_v.displacements    = (ucc_aint_t  *)rdispls.data();
    args.dst.info_v.datatype         = UCC_DT_UINT32;
    args.dst.info_v.mem_type         = UCC_MEMORY_TYPE_CUDA;
    args.dst_memh.global_memh        = gmemh.handles;

    ucc_coll_req_h req;
    ucc_status_t   init_status = ucc_collective_init(&args, &req, s.team);
    if (init_status == UCC_ERR_NOT_SUPPORTED ||
        init_status == UCC_ERR_NOT_IMPLEMENTED) {
        if (g_verbose && rank == 0)
            printf("  alltoallv %zu B  SKIP (not supported)\n", msg_bytes);
        free_global_memh(gmemh);
        CUDA_CHECK(cudaFree(d_sbuf));
        CUDA_CHECK(cudaFree(d_rbuf));
        g_skip_count++;
        return true;
    }
    UCC_CHECK(init_status);

    run_collective(s.ctx, req);

    /* Validate: for each peer, data in rbuf at rdispls[peer]..rdispls[peer]+rcounts[peer]
     * must equal what that peer sent for us (at peer's sdispls[rank]..+scounts[rank]).
     * We need the peer's sdispl for rank -- gather all sdispls. */
    std::vector<int> all_sdispls(nranks * nranks);
    MPI_CHECK(MPI_Allgather(sdispls.data(), nranks, MPI_INT,
                            all_sdispls.data(), nranks, MPI_INT, comm));

    CUDA_CHECK(cudaMemcpy(h_rbuf.data(), d_rbuf, rbytes, cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int peer = 0; peer < nranks && ok; peer++) {
        /* peer's sdispls[rank] is where peer stored data for us */
        int peer_sdispl_for_rank = all_sdispls[(size_t)peer * nranks + rank];
        for (int i = 0; i < rcounts[peer] && ok; i++) {
            int      peer_pos = peer_sdispl_for_rank + i;
            uint32_t expected = (uint32_t)(((uint32_t)peer << 16) |
                                           (uint32_t)(peer_pos & 0xFFFF));
            uint32_t got      = h_rbuf[(size_t)rdispls[peer] + i];
            if (got != expected) {
                fprintf(stderr, "[rank %d] alltoallv FAIL size=%zu: "
                        "rbuf[peer=%d][%d] = 0x%08x, expected 0x%08x\n",
                        rank, msg_bytes, peer, i, got, expected);
                ok = false;
            }
        }
    }

    if (g_verbose && rank == 0)
        printf("  alltoallv %7zu B  %s\n", msg_bytes, ok ? "PASS" : "FAIL");

    free_global_memh(gmemh);
    CUDA_CHECK(cudaFree(d_sbuf));
    CUDA_CHECK(cudaFree(d_rbuf));
    return ok;
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) g_verbose = true;
    }

    /* Assign GPUs round-robin by rank */
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        if (g_rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(g_rank % ndev));
    CUDA_CHECK(cudaFree(0)); /* force context creation */

    UccState s = setup_ucc(MPI_COMM_WORLD);

    /* Message sizes: 4 B to 4 MB in powers of 4 */
    std::vector<size_t> sizes;
    for (size_t m = 4; m <= 4 * 1024 * 1024; m *= 4)
        sizes.push_back(m);

    if (g_rank == 0 && g_verbose)
        printf("=== alltoall (nranks=%d) ===\n", g_nranks);

    for (size_t m : sizes) {
        bool ok = test_alltoall(s, MPI_COMM_WORLD, m);
        /* Reduce result across ranks — any failure counts */
        int local_ok = ok ? 1 : 0, global_ok;
        MPI_CHECK(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                                MPI_COMM_WORLD));
        if (local_ok && global_ok) {
            g_pass_count++;
            if (!g_verbose && g_rank == 0)
                printf("alltoall  %7zu B  PASS\n", m);
        } else if (!global_ok) {
            g_fail_count++;
            if (g_rank == 0)
                printf("alltoall  %7zu B  FAIL\n", m);
        }
    }

    if (g_rank == 0 && g_verbose)
        printf("\n=== alltoallv (nranks=%d) ===\n", g_nranks);

    for (size_t m : sizes) {
        bool ok = test_alltoallv(s, MPI_COMM_WORLD, m);
        int local_ok = ok ? 1 : 0, global_ok;
        MPI_CHECK(MPI_Allreduce(&local_ok, &global_ok, 1, MPI_INT, MPI_MIN,
                                MPI_COMM_WORLD));
        if (local_ok && global_ok) {
            g_pass_count++;
            if (!g_verbose && g_rank == 0)
                printf("alltoallv %7zu B  PASS\n", m);
        } else if (!global_ok) {
            g_fail_count++;
            if (g_rank == 0)
                printf("alltoallv %7zu B  FAIL\n", m);
        }
    }

    teardown_ucc(s);

    if (g_rank == 0) {
        printf("\n%d passed, %d failed, %d skipped\n",
               g_pass_count, g_fail_count, g_skip_count);
    }

    MPI_Finalize();
    return (g_fail_count > 0) ? 1 : 0;
}
