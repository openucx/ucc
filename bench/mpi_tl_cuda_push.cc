/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

/*
 * MPI port of the TL/CUDA push-algorithm gtest
 * (test/gtest/coll/test_tl_cuda_push.cc).
 *
 * The gtest runs every rank inside a single process using the UccJob/UccTeam
 * harness and exchanges destination-buffer handles in-process.  This program
 * runs each rank as its own MPI process with its own GPU and performs the same
 * exchange over MPI:
 *   1. Export each rank's CUDA dst buffer via ucc_mem_map(EXPORT).
 *   2. MPI_Allgather the serialized handle bytes.
 *   3. Import every peer's handle via ucc_mem_map(IMPORT).
 *   4. Set dst_memh.global_memh + UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL — the path
 *      that triggers alltoall_push / alltoallv_push in TL/CUDA.
 *
 * The data patterns, test matrix, and validation are kept identical to the
 * gtest:
 *   - alltoallx byte fill/validate pattern
 *   - dtypes {UINT8, FLOAT32} x counts {1, 64, 1024}
 *   - alltoall push (single + persistent)
 *   - alltoallv push (64-bit and 32-bit count/displacement variants) with the
 *     (nprocs + r - i) * count send pattern and forced zero-counts
 *
 * Algorithm selection is forced via the same env the gtest sets:
 *   UCC_TL_CUDA_TUNE = "alltoall:cuda:@push:0-inf:inf" (and alltoallv)
 *   UCC_CL_BASIC_TUNE = "inf"
 *
 * Usage:
 *   mpirun -n <N> ./mpi_tl_cuda_push [-v]
 *     -v   verbose: print pass/fail per (dtype,count) case
 *
 * Exits 0 on success, 1 on any correctness failure.  N must be >= 2.
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cassert>
#include <vector>
#include <string>
#include <mpi.h>
#include <cuda_runtime.h>
#include <ucc/api/ucc.h>

/* ------------------------------------------------------------------ */
/* Globals + error-check helpers                                       */
/* ------------------------------------------------------------------ */

static int  g_rank    = 0;
static int  g_nranks  = 0;
static bool g_verbose = false;

static int  g_pass_count = 0;
static int  g_fail_count = 0;

#define CUDA_CHECK(call)                                                     \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "[rank %d] CUDA error %s:%d: %s\n",             \
                    g_rank, __FILE__, __LINE__, cudaGetErrorString(_e));    \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

#define UCC_CHECK(call)                                                     \
    do {                                                                    \
        ucc_status_t _s = (call);                                          \
        if (_s != UCC_OK) {                                                 \
            fprintf(stderr, "[rank %d] UCC error %s:%d: %s (%d)\n",         \
                    g_rank, __FILE__, __LINE__, ucc_status_string(_s),      \
                    (int)_s);                                               \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

#define MPI_CHECK(call)                                                     \
    do {                                                                    \
        int _e = (call);                                                    \
        if (_e != MPI_SUCCESS) {                                            \
            fprintf(stderr, "[rank %d] MPI error %s:%d code=%d\n",          \
                    g_rank, __FILE__, __LINE__, _e);                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                   \
        }                                                                   \
    } while (0)

/* ------------------------------------------------------------------ */
/* alltoallx data pattern (identical to test/gtest/common/test_ucc.h)  */
/* ------------------------------------------------------------------ */

static void alltoallx_init_buf(int src_rank, int dst_rank, uint8_t *buf,
                               size_t len)
{
    for (size_t i = 0; i < len; i++) {
        buf[i] = (uint8_t)(((src_rank + len - i) *
                            (dst_rank + 1)) % UINT8_MAX);
    }
}

static int alltoallx_validate_buf(int src_rank, int dst_rank, uint8_t *buf,
                                  size_t len)
{
    int err = 0;
    for (size_t i = 0; i < len; i++) {
        uint8_t expected = (uint8_t)(((dst_rank + len - i) *
                                      (src_rank + 1)) % UINT8_MAX);
        if (buf[i] != expected) {
            err++;
        }
    }
    return err;
}

/* ------------------------------------------------------------------ */
/* UCC OOB allgather (required for context / team creation)            */
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
    UccState     s;
    ucc_status_t status;
    int          rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    ucc_lib_config_h lib_cfg;
    UCC_CHECK(ucc_lib_config_read(NULL, NULL, &lib_cfg));

    ucc_lib_params_t lib_params = {};
    lib_params.mask        = UCC_LIB_PARAM_FIELD_THREAD_MODE;
    lib_params.thread_mode = UCC_THREAD_SINGLE;
    UCC_CHECK(ucc_init(&lib_params, lib_cfg, &s.lib));
    ucc_lib_config_release(lib_cfg);

    ucc_context_config_h ctx_cfg;
    UCC_CHECK(ucc_context_config_read(s.lib, NULL, &ctx_cfg));

    ucc_context_params_t ctx_params = {};
    ctx_params.mask          = UCC_CONTEXT_PARAM_FIELD_OOB;
    ctx_params.oob.allgather = oob_allgather;
    ctx_params.oob.req_test  = oob_allgather_test;
    ctx_params.oob.req_free  = oob_allgather_free;
    ctx_params.oob.coll_info = (void *)(uintptr_t)comm;
    ctx_params.oob.n_oob_eps = (uint32_t)size;
    ctx_params.oob.oob_ep    = (uint32_t)rank;
    UCC_CHECK(ucc_context_create(s.lib, &ctx_params, ctx_cfg, &s.ctx));
    ucc_context_config_release(ctx_cfg);

    ucc_team_params_t team_params = {};
    team_params.mask          = UCC_TEAM_PARAM_FIELD_EP       |
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
/* Build global_memh: export dst buffer, allgather + import peers      */
/* ------------------------------------------------------------------ */

struct GlobalMemh {
    ucc_mem_map_mem_h  local_memh;  /* our own exported handle */
    ucc_mem_map_mem_h *handles;     /* [i]: exported (self) / imported (peer) */
    size_t             memh_size;
    int                nranks;
    int                rank;
};

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
    MPI_CHECK(MPI_Allgather(g.local_memh, (int)g.memh_size, MPI_BYTE,
                            all_bytes,    (int)g.memh_size, MPI_BYTE, comm));

    g.handles = (ucc_mem_map_mem_h *)malloc((size_t)g.nranks *
                                            sizeof(ucc_mem_map_mem_h));
    assert(g.handles);

    for (int i = 0; i < g.nranks; i++) {
        if (i == g.rank) {
            g.handles[i] = g.local_memh; /* already valid */
        } else {
            /* Copy peer bytes into a fresh buffer and import in-place.
             * ucc_mem_map(IMPORT) reads the packed TL data serialized by the
             * exporter and rebuilds the handle in this process's address
             * space; the raw bytes alone carry an exporter-heap pointer and
             * cannot be used directly. */
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
/* Progress a request to completion                                    */
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
}

/* ------------------------------------------------------------------ */
/* alltoall push                                                       */
/* ------------------------------------------------------------------ */

/*
 * Mirrors test_alltoall_push.  single_rank_count = per-peer element count.
 * n_calls > 1 exercises the persistent path (UCC_COLL_ARGS_FLAG_PERSISTENT),
 * re-posting the same request and clearing the dst buffer between calls, just
 * like the gtest's `persistent` case.
 */
static bool test_alltoall_push(UccState &s, MPI_Comm comm,
                               ucc_datatype_t dtype, size_t single_rank_count,
                               int n_calls)
{
    int    nranks   = g_nranks;
    int    rank     = g_rank;
    size_t dt_size  = ucc_dt_size(dtype);
    size_t buf_size = dt_size * single_rank_count * (size_t)nranks;

    /* Host init buffer: chunk destined for peer i = init(rank, i) */
    std::vector<uint8_t> h_init(buf_size);
    for (int i = 0; i < nranks; i++) {
        alltoallx_init_buf(rank, i,
                           h_init.data() + (size_t)i * single_rank_count * dt_size,
                           single_rank_count * dt_size);
    }

    void *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, buf_size));
    CUDA_CHECK(cudaMalloc(&d_dst, buf_size));
    CUDA_CHECK(cudaMemcpy(d_src, h_init.data(), buf_size,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, buf_size));

    GlobalMemh gmemh = build_global_memh(s.ctx, d_dst, buf_size, comm);

    ucc_coll_args_t args = {};
    args.mask                 = UCC_COLL_ARGS_FIELD_FLAGS |
                                UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;
    args.flags                = UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL;
    args.coll_type            = UCC_COLL_TYPE_ALLTOALL;
    args.src.info.buffer      = d_src;
    args.src.info.count       = (uint64_t)(single_rank_count * (size_t)nranks);
    args.src.info.datatype    = dtype;
    args.src.info.mem_type    = UCC_MEMORY_TYPE_CUDA;
    args.dst.info.buffer      = d_dst;
    args.dst.info.count       = (uint64_t)(single_rank_count * (size_t)nranks);
    args.dst.info.datatype    = dtype;
    args.dst.info.mem_type    = UCC_MEMORY_TYPE_CUDA;
    args.dst_memh.global_memh = gmemh.handles;

    if (n_calls > 1)
        args.flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;

    ucc_coll_req_h req;
    UCC_CHECK(ucc_collective_init(&args, &req, s.team));

    std::vector<uint8_t> h_dst(buf_size);
    bool ok = true;

    for (int call = 0; call < n_calls && ok; call++) {
        run_collective(s.ctx, req);

        CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, buf_size,
                              cudaMemcpyDeviceToHost));

        size_t per_rank = single_rank_count;
        for (int i = 0; i < nranks; i++) {
            size_t rank_sz = dt_size * per_rank;
            /* slot i holds what peer i sent us: validate(rank,i) == init(i,rank) */
            if (alltoallx_validate_buf(rank, i,
                                       h_dst.data() + rank_sz * (size_t)i,
                                       rank_sz) != 0) {
                fprintf(stderr, "[rank %d] alltoall FAIL dtype=%d count=%zu "
                        "call=%d peer=%d\n", rank, (int)dtype,
                        single_rank_count, call, i);
                ok = false;
                break;
            }
        }

        /* Clear dst between persistent calls */
        if (call + 1 < n_calls)
            CUDA_CHECK(cudaMemset(d_dst, 0, buf_size));
    }

    UCC_CHECK(ucc_collective_finalize(req));
    free_global_memh(gmemh);
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return ok;
}

/* ------------------------------------------------------------------ */
/* alltoallv push                                                      */
/* ------------------------------------------------------------------ */

/*
 * Mirrors test_alltoallv_push<T>.  T selects the count/displacement width:
 *   uint64_t -> COUNT_64BIT | DISPLACEMENTS_64BIT
 *   uint32_t -> 32-bit (no width flags)
 *
 * Rank r sends (nprocs + r - i) * count elements to rank i and receives
 * (nprocs - r + i) * count from rank i; one send-count and the mirrored
 * recv-count are forced to zero for corner-case coverage.
 */
template <typename T>
static bool test_alltoallv_push(UccState &s, MPI_Comm comm,
                                ucc_datatype_t dtype, size_t count)
{
    int    nranks  = g_nranks;
    int    rank    = g_rank;
    size_t dt_size = ucc_dt_size(dtype);

    std::vector<T> scounts(nranks), sdispls(nranks);
    std::vector<T> rcounts(nranks), rdispls(nranks);

    /* Send counts / displacements */
    size_t sbuf_elems = 0;
    for (int i = 0; i < nranks; i++) {
        T sc       = (T)((nranks + rank - i) * (int)count);
        scounts[i] = sc;
        sdispls[i] = (T)sbuf_elems;
        sbuf_elems += sc;
    }
    scounts[(rank + 1) % nranks] = 0; /* forced zero send-count */

    /* Recv counts / displacements */
    size_t rbuf_elems = 0;
    for (int i = 0; i < nranks; i++) {
        T rc       = (T)((nranks - rank + i) * (int)count);
        rcounts[i] = rc;
        rdispls[i] = (T)rbuf_elems;
        rbuf_elems += rc;
    }
    rcounts[(rank - 1 + nranks) % nranks] = 0; /* mirrored zero recv-count */

    size_t sbuf_bytes = sbuf_elems * dt_size;
    size_t rbuf_bytes = rbuf_elems * dt_size;

    /* Host init: chunk destined for peer i = init(rank, i) */
    std::vector<uint8_t> h_init(sbuf_bytes ? sbuf_bytes : 1);
    for (int i = 0; i < nranks; i++) {
        alltoallx_init_buf(rank, i,
                           h_init.data() + (size_t)sdispls[i] * dt_size,
                           (size_t)scounts[i] * dt_size);
    }

    void *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, sbuf_bytes ? sbuf_bytes : 1));
    CUDA_CHECK(cudaMalloc(&d_dst, rbuf_bytes ? rbuf_bytes : 1));
    CUDA_CHECK(cudaMemcpy(d_src, h_init.data(), sbuf_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_dst, 0, rbuf_bytes));

    GlobalMemh gmemh = build_global_memh(s.ctx, d_dst, rbuf_bytes, comm);

    ucc_coll_args_t args = {};
    args.mask  = UCC_COLL_ARGS_FIELD_FLAGS |
                 UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH;
    args.flags = UCC_COLL_ARGS_FLAG_CONTIG_SRC_BUFFER |
                 UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER |
                 UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL;
    if (sizeof(T) == sizeof(uint64_t)) {
        args.flags |= UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                      UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
    }
    args.coll_type                = UCC_COLL_TYPE_ALLTOALLV;
    args.src.info_v.buffer        = d_src;
    args.src.info_v.counts        = (ucc_count_t *)scounts.data();
    args.src.info_v.displacements = (ucc_aint_t *)sdispls.data();
    args.src.info_v.datatype      = dtype;
    args.src.info_v.mem_type      = UCC_MEMORY_TYPE_CUDA;
    args.dst.info_v.buffer        = d_dst;
    args.dst.info_v.counts        = (ucc_count_t *)rcounts.data();
    args.dst.info_v.displacements = (ucc_aint_t *)rdispls.data();
    args.dst.info_v.datatype      = dtype;
    args.dst.info_v.mem_type      = UCC_MEMORY_TYPE_CUDA;
    args.dst_memh.global_memh     = gmemh.handles;

    ucc_coll_req_h req;
    UCC_CHECK(ucc_collective_init(&args, &req, s.team));
    run_collective(s.ctx, req);
    UCC_CHECK(ucc_collective_finalize(req));

    std::vector<uint8_t> h_dst(rbuf_bytes ? rbuf_bytes : 1);
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, rbuf_bytes,
                          cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < nranks; i++) {
        size_t rank_sz  = (size_t)rcounts[i] * dt_size;
        size_t rank_off = (size_t)rdispls[i] * dt_size;
        if (alltoallx_validate_buf(rank, i, h_dst.data() + rank_off,
                                   rank_sz) != 0) {
            fprintf(stderr, "[rank %d] alltoallv FAIL dtype=%d count=%zu "
                    "width=%zu peer=%d\n", rank, (int)dtype, count,
                    sizeof(T) * 8, i);
            ok = false;
            break;
        }
    }

    free_global_memh(gmemh);
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    return ok;
}

/* ------------------------------------------------------------------ */
/* Case runner: reduce local result across ranks, tally, report        */
/* ------------------------------------------------------------------ */

static void record_case(const char *name, bool local_ok)
{
    int lo = local_ok ? 1 : 0, go = 0;
    MPI_CHECK(MPI_Allreduce(&lo, &go, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD));
    if (go) {
        g_pass_count++;
        if (g_verbose && g_rank == 0)
            printf("  %-40s PASS\n", name);
    } else {
        g_fail_count++;
        if (g_rank == 0)
            printf("  %-40s FAIL\n", name);
    }
}

static const char *dt_name(ucc_datatype_t dt)
{
    switch (dt) {
    case UCC_DT_UINT8:   return "uint8";
    case UCC_DT_FLOAT32: return "float32";
    default:             return "dt?";
    }
}

/* ------------------------------------------------------------------ */
/* main                                                                */
/* ------------------------------------------------------------------ */

int main(int argc, char **argv)
{
    /* Force the push algorithm, matching the gtest env. */
    setenv("UCC_TL_CUDA_TUNE",
           "alltoall:cuda:@push:0-inf:inf#alltoallv:cuda:@push:0-inf:inf", 1);
    setenv("UCC_CL_BASIC_TUNE", "inf", 1);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nranks);

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0)
            g_verbose = true;
    }

    if (g_nranks < 2) {
        if (g_rank == 0)
            fprintf(stderr, "This test requires at least 2 ranks.\n");
        MPI_Finalize();
        return 1;
    }

    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        if (g_rank == 0)
            fprintf(stderr, "No CUDA devices found.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(g_rank % ndev));
    CUDA_CHECK(cudaFree(0)); /* force context creation */

    UccState s = setup_ucc(MPI_COMM_WORLD);

    const ucc_datatype_t dtypes[] = { UCC_DT_UINT8, UCC_DT_FLOAT32 };
    const size_t         counts[] = { 1, 64, 1024 };
    char                 name[128];

    if (g_rank == 0)
        printf("=== alltoall push (nranks=%d) ===\n", g_nranks);
    for (ucc_datatype_t dt : dtypes) {
        for (size_t c : counts) {
            snprintf(name, sizeof(name), "alltoall single  %-7s count=%zu",
                     dt_name(dt), c);
            record_case(name, test_alltoall_push(s, MPI_COMM_WORLD, dt, c, 1));

            snprintf(name, sizeof(name), "alltoall persist %-7s count=%zu",
                     dt_name(dt), c);
            record_case(name, test_alltoall_push(s, MPI_COMM_WORLD, dt, c, 3));
        }
    }

    if (g_rank == 0)
        printf("=== alltoallv push (nranks=%d) ===\n", g_nranks);
    for (ucc_datatype_t dt : dtypes) {
        for (size_t c : counts) {
            snprintf(name, sizeof(name), "alltoallv 64bit  %-7s count=%zu",
                     dt_name(dt), c);
            record_case(name,
                        test_alltoallv_push<uint64_t>(s, MPI_COMM_WORLD, dt, c));

            snprintf(name, sizeof(name), "alltoallv 32bit  %-7s count=%zu",
                     dt_name(dt), c);
            record_case(name,
                        test_alltoallv_push<uint32_t>(s, MPI_COMM_WORLD, dt, c));
        }
    }

    teardown_ucc(s);

    if (g_rank == 0)
        printf("\n%d passed, %d failed\n", g_pass_count, g_fail_count);

    MPI_Finalize();
    return (g_fail_count > 0) ? 1 : 0;
}
