/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoall.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_malloc.h"
#include "tl_ucp_sendrecv.h"
#include <math.h>

#define CONGESTION_THRESHOLD 8

/* Message fragmentation: when the pacing budget (tokens) drops to one whole
 * per-peer message or below, one message no longer fits the inflight window,
 * so the per-peer message is split into fragments and a fragment becomes the
 * inflight unit instead. */
#define ALLTOALL_ONESIDED_MAX_FRAGS   64      /* keep nfrags * gsize bounded */
#define ALLTOALL_ONESIDED_MIN_FRAG_SZ 8192    /* don't fragment below this   */

typedef struct alltoall_onesided_frag {
    ucc_rank_t peer;    /* target peer                             */
    size_t     offset;  /* byte offset within the per-peer block   */
    size_t     len;     /* byte length of this fragment            */
    int        last;    /* nonzero if last fragment for this peer  */
} alltoall_onesided_frag_t;

/* Map a linear op index onto a (peer, fragment) pair. Fragment-major order:
 * the outer loop is the fragment index and the inner loop is the position in
 * peer_order, so fragment k of every peer is issued before any peer's fragment
 * k+1. This round-robins the shm/IB lanes at sub-message granularity. */
static inline void
alltoall_onesided_get_fragment(const ucc_rank_t *order, ucc_rank_t gsize,
                               uint32_t nfrags, size_t peer_nbytes,
                               uint32_t op_idx, alltoall_onesided_frag_t *f)
{
    uint32_t frag = op_idx / gsize;   /* fragment-major: outer = fragment */
    uint32_t pos  = op_idx % gsize;   /* inner = position in peer_order   */

    f->peer   = order[pos];
    f->offset = ucc_buffer_block_offset(peer_nbytes, nfrags, frag);
    f->len    = ucc_buffer_block_count(peer_nbytes, nfrags, frag);
    f->last   = (frag == nfrags - 1);
}

static inline ucc_rank_t alltoall_onesided_gcd(ucc_rank_t a, ucc_rank_t b)
{
    ucc_rank_t t;

    while (b) {
        t = b;
        b = a % b;
        a = t;
    }
    return a;
}

static inline ucc_rank_t
alltoall_onesided_coprime_stride(ucc_rank_t size)
{
    ucc_rank_t stride;

    if (size <= 2) {
        return 1;
    }

    for (stride = size / 2 + 1; stride < size; stride++) {
        if (alltoall_onesided_gcd(stride, size) == 1) {
            return stride;
        }
    }

    return 1;
}

static inline int alltoall_onesided_peer_is_local(ucc_sbgp_t *sbgp,
                                                  ucc_rank_t  grank,
                                                  ucc_rank_t  peer)
{
    ucc_rank_t i;

    if (peer == grank) {
        return 1;
    }

    if (!sbgp || sbgp->status != UCC_SBGP_ENABLED) {
        return 0;
    }

    for (i = 0; i < sbgp->group_size; i++) {
        if (ucc_ep_map_eval(sbgp->map, i) == peer) {
            return 1;
        }
    }

    return 0;
}

static void alltoall_onesided_rotate_left(ucc_rank_t *peers, ucc_rank_t n,
                                          ucc_rank_t shift)
{
    ucc_rank_t gcd, i, j, k, tmp;

    if (n <= 1) {
        return;
    }

    shift %= n;
    if (!shift) {
        return;
    }

    gcd = alltoall_onesided_gcd(shift, n);
    for (i = 0; i < gcd; i++) {
        tmp = peers[i];
        j   = i;
        while (1) {
            k = j + shift;
            if (k >= n) {
                k -= n;
            }
            if (k == i) {
                break;
            }
            peers[j] = peers[k];
            j        = k;
        }
        peers[j] = tmp;
    }
}

static void alltoall_onesided_rotate_remote(ucc_rank_t *remote,
                                            ucc_rank_t  n_remote,
                                            ucc_sbgp_t *sbgp)
{
    ucc_rank_t node_rank, node_size, shift;

    if (n_remote <= 1 || !sbgp || sbgp->status != UCC_SBGP_ENABLED) {
        return;
    }

    node_rank = sbgp->group_rank;
    node_size = sbgp->group_size;
    if (node_size > 1 && n_remote % node_size == 0) {
        shift = (node_rank * node_size) % n_remote;
    } else {
        shift = node_rank % n_remote;
    }
    alltoall_onesided_rotate_left(remote, n_remote, shift);
}

static ucc_status_t
alltoall_onesided_build_peer_order(ucc_tl_ucp_task_t *task,
                                   ucc_tl_ucp_team_t *team,
                                   ucc_sbgp_t        *sbgp,
                                   ucc_tl_ucp_alltoall_onesided_alg_t alg)
{
    ucc_tl_ucp_alltoall_onesided_order_t order =
        UCC_TL_UCP_TEAM_LIB(team)->cfg.alltoall_onesided_order;
    ucc_rank_t  grank  = UCC_TL_TEAM_RANK(team);
    ucc_rank_t  gsize  = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t  stride = 1;
    ucc_rank_t *peers;
    ucc_rank_t *tmp;
    ucc_rank_t *local;
    ucc_rank_t *remote;
    ucc_rank_t  n_local, n_remote;
    ucc_rank_t  i, peer, out;

    task->alltoall_onesided.peer_order = NULL;
    peers = ucc_malloc(gsize * sizeof(*peers), "alltoall_onesided_peer_order");
    if (!peers) {
        return UCC_ERR_NO_MEMORY;
    }
    task->alltoall_onesided.peer_order = peers;
    task->alltoall_onesided.n_local    = 0;

    if (order == UCC_TL_UCP_ALLTOALL_ONESIDED_ORDER_STRIDE ||
        order == UCC_TL_UCP_ALLTOALL_ONESIDED_ORDER_FULL) {
        stride = alltoall_onesided_coprime_stride(gsize);
    }

    /* Mixed progress classifies local vs. remote by position in the order
     * (the first/last n_local slots), so it requires the local/remote split
     * even under seq/stride. The split keeps the requested walk order within
     * each class; only FULL additionally rotates the remote class. */
    if ((order == UCC_TL_UCP_ALLTOALL_ONESIDED_ORDER_SEQ ||
         order == UCC_TL_UCP_ALLTOALL_ONESIDED_ORDER_STRIDE) &&
        alg != UCC_TL_UCP_ALLTOALL_ONESIDED_MIXED) {
        for (i = 0; i < gsize; i++) {
            peers[i] = (grank + 1 + i * stride) % gsize;
        }
        return UCC_OK;
    }

    tmp = ucc_malloc(2 * gsize * sizeof(*tmp), "alltoall_onesided_peer_tmp");
    if (!tmp) {
        ucc_free(peers);
        task->alltoall_onesided.peer_order = NULL;
        return UCC_ERR_NO_MEMORY;
    }
    local    = tmp;
    remote   = tmp + gsize;
    n_local  = 0;
    n_remote = 0;

    for (i = 0; i < gsize; i++) {
        peer = (grank + 1 + i * stride) % gsize;
        if (alltoall_onesided_peer_is_local(sbgp, grank, peer)) {
            local[n_local++] = peer;
        } else {
            remote[n_remote++] = peer;
        }
    }

    if (order == UCC_TL_UCP_ALLTOALL_ONESIDED_ORDER_FULL) {
        alltoall_onesided_rotate_remote(remote, n_remote, sbgp);
    }

    /* Odd local rank: remote-first so half the node's ranks hit the NIC
     * immediately while even ranks saturate SHM, halving peak NIC pressure.
     * Even local rank: local-first. Both then drain the remaining set. */
    out = 0;
    if ((sbgp && sbgp->status == UCC_SBGP_ENABLED ? sbgp->group_rank
                                                   : grank) % 2) {
        for (i = 0; i < n_remote; i++) peers[out++] = remote[i];
        for (i = 0; i < n_local;  i++) peers[out++] = local[i];
    } else {
        for (i = 0; i < n_local;  i++) peers[out++] = local[i];
        for (i = 0; i < n_remote; i++) peers[out++] = remote[i];
    }

    task->alltoall_onesided.n_local = n_local;
    ucc_free(tmp);
    return UCC_OK;
}

/* Common helper function to check completion and handle polling */
static inline int alltoall_onesided_handle_completion(
    ucc_tl_ucp_task_t *task, uint32_t *posted, uint32_t *completed,
    uint32_t nreqs, int64_t npolls)
{
    int64_t polls = 0;

    if ((*posted - *completed) >= nreqs) {
        while (polls < npolls) {
            ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
            ++polls;
            if ((*posted - *completed) < nreqs) {
                break;
            }
        }
        if (polls >= npolls) {
            return 0; /* Return 0 to indicate should return */
        }
    }
    return 1; /* Return 1 to indicate should continue */
}

/* Common helper function to wait for all operations to complete */
static inline void alltoall_onesided_wait_completion(ucc_tl_ucp_task_t *task,
                                                     int64_t npolls)
{
    int64_t polls = 0;

    if (!UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
        while (polls++ < npolls) {
            ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
            if (UCC_TL_UCP_TASK_ONESIDED_P2P_COMPLETE(task)) {
                task->super.status = UCC_OK;
                return;
            }
        }
        return;
    }
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_sched_start(ucc_coll_task_t *ctask)
{
    return ucc_schedule_start(ctask);
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_sched_finalize(ucc_coll_task_t *ctask)
{
    ucc_schedule_t *schedule = ucc_derived_of(ctask, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(ctask);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_status_t       status;

    if (task->alltoall_onesided.peer_order) {
        ucc_free(task->alltoall_onesided.peer_order);
        task->alltoall_onesided.peer_order = NULL;
    }

    status = ucc_tl_ucp_coll_finalize(coll_task);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(coll_task), "failed to finalize collective");
    }
    return status;
}

void ucc_tl_ucp_alltoall_onesided_get_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ptrdiff_t          src       = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  mtype     = TASK_ARGS(task).dst.info.mem_type;
    ucc_rank_t         grank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize     = UCC_TL_TEAM_SIZE(team);
    uint32_t           ntokens   = task->alltoall_onesided.tokens;
    int64_t            npolls    = task->alltoall_onesided.npolls;
    /* To resolve remote virtual addresses, the dst_memh is the one that must
     * have the rkey information. For this algorithm, we need to swap the
     * src and dst handles to operate correctly */
    ucc_mem_map_mem_h *dst_memh  = TASK_ARGS(task).src_memh.global_memh;
    uint32_t          *posted    = &task->onesided.get_posted;
    uint32_t          *completed = &task->onesided.get_completed;
    ucc_rank_t        *order     = task->alltoall_onesided.peer_order;
    uint32_t           nfrags    = task->alltoall_onesided.nfrags;
    ucc_mem_map_mem_h  src_memh;
    size_t             nelems;
    uint32_t           total;
    alltoall_onesided_frag_t frag;

    nelems   = TASK_ARGS(task).src.info.count;
    nelems   = (nelems / gsize) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    total    = nfrags * gsize;
    src_memh = (TASK_ARGS(task).flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL)
                   ? TASK_ARGS(task).dst_memh.global_memh[grank]
                   : TASK_ARGS(task).dst_memh.local_memh;

    for (; *posted < total;) {
        alltoall_onesided_get_fragment(order, gsize, nfrags, nelems, *posted,
                                       &frag);
        UCPCHECK_GOTO(
            ucc_tl_ucp_get_nb(PTR_OFFSET(dest, frag.peer * nelems + frag.offset),
                              PTR_OFFSET(src, grank * nelems + frag.offset),
                              frag.len, mtype, frag.peer, src_memh, dst_memh,
                              team, task),
            task, out);

        if (!alltoall_onesided_handle_completion(task, posted, completed,
                                                 ntokens, npolls)) {
            return;
        }
    }

    alltoall_onesided_wait_completion(task, npolls);
out:
    return;
}

void ucc_tl_ucp_alltoall_onesided_put_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ptrdiff_t          src       = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest      = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  mtype     = TASK_ARGS(task).src.info.mem_type;
    ucc_rank_t         grank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize     = UCC_TL_TEAM_SIZE(team);
    uint32_t           ntokens   = task->alltoall_onesided.tokens;
    int64_t            npolls    = task->alltoall_onesided.npolls;
    ucc_mem_map_mem_h *dst_memh  = TASK_ARGS(task).dst_memh.global_memh;
    uint32_t          *posted    = &task->onesided.put_posted;
    uint32_t          *completed = &task->onesided.put_completed;
    ucc_rank_t        *order     = task->alltoall_onesided.peer_order;
    uint32_t           nfrags    = task->alltoall_onesided.nfrags;
    ucc_mem_map_mem_h  src_memh;
    size_t             nelems;
    uint32_t           total;
    alltoall_onesided_frag_t frag;

    nelems   = TASK_ARGS(task).src.info.count;
    nelems   = (nelems / gsize) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    total    = nfrags * gsize;
    src_memh = (TASK_ARGS(task).flags & UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL)
                   ? TASK_ARGS(task).src_memh.global_memh[grank]
                   : TASK_ARGS(task).src_memh.local_memh;

    for (; *posted < total;) {
        alltoall_onesided_get_fragment(order, gsize, nfrags, nelems, *posted,
                                       &frag);
        UCPCHECK_GOTO(
            ucc_tl_ucp_put_nb(PTR_OFFSET(src, frag.peer * nelems + frag.offset),
                              PTR_OFFSET(dest, grank * nelems + frag.offset),
                              frag.len, mtype, frag.peer, src_memh, dst_memh,
                              team, task),
            task, out);
        /* fragment-major order posts every peer's last fragment in the final
         * pass, so flush each peer exactly once when its last fragment lands */
        if (frag.last) {
            UCPCHECK_GOTO(ucc_tl_ucp_ep_flush(frag.peer, team, task), task, out);
        }

        if (!alltoall_onesided_handle_completion(task, posted, completed,
                                                 ntokens, npolls)) {
            return;
        }
    }

    alltoall_onesided_wait_completion(task, npolls);
out:
    return;
}

void ucc_tl_ucp_alltoall_onesided_mixed_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ptrdiff_t          src        = (ptrdiff_t)TASK_ARGS(task).src.info.buffer;
    ptrdiff_t          dest       = (ptrdiff_t)TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  src_mtype  = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  dst_mtype  = TASK_ARGS(task).dst.info.mem_type;
    ucc_rank_t         grank      = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize      = UCC_TL_TEAM_SIZE(team);
    uint32_t           ntokens    = task->alltoall_onesided.tokens;
    int64_t            npolls     = task->alltoall_onesided.npolls;
    ucc_rank_t         n_local    = task->alltoall_onesided.n_local;
    ucc_rank_t        *order      = task->alltoall_onesided.peer_order;
    uint32_t           nfrags     = task->alltoall_onesided.nfrags;
    /* GET: rkey resolves peer's src buffer; PUT: rkey resolves peer's dst buffer */
    ucc_mem_map_mem_h *get_rkey   = TASK_ARGS(task).src_memh.global_memh;
    ucc_mem_map_mem_h *put_rkey   = TASK_ARGS(task).dst_memh.global_memh;
    ucc_mem_map_mem_h  get_lmemh  =
        (TASK_ARGS(task).flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL)
            ? TASK_ARGS(task).dst_memh.global_memh[grank]
            : TASK_ARGS(task).dst_memh.local_memh;
    ucc_mem_map_mem_h  put_lmemh  =
        (TASK_ARGS(task).flags & UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL)
            ? TASK_ARGS(task).src_memh.global_memh[grank]
            : TASK_ARGS(task).src_memh.local_memh;
    ucc_sbgp_t        *sbgp       = ucc_topo_get_sbgp(team->topo, UCC_SBGP_NODE);
    int                local_first =
        ((sbgp && sbgp->status == UCC_SBGP_ENABLED ? sbgp->group_rank
                                                    : grank) % 2 == 0);
    size_t             nelems;
    uint32_t           total, total_posted, inflight;
    int64_t            polls;
    uint32_t           pos;
    int                is_local;
    int                is_local_peer;
    alltoall_onesided_frag_t frag;

    nelems       = TASK_ARGS(task).src.info.count;
    nelems       = (nelems / gsize) * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    total        = nfrags * gsize;
    total_posted = task->onesided.get_posted + task->onesided.put_posted;

    for (; total_posted < total;) {
        pos      = total_posted % gsize;
        is_local = local_first ? (pos < n_local)
                               : (pos >= (gsize - n_local));

        alltoall_onesided_get_fragment(order, gsize, nfrags, nelems, total_posted,
                                       &frag);
        if (is_local) {
            UCPCHECK_GOTO(
                ucc_tl_ucp_get_nb(
                    PTR_OFFSET(dest, frag.peer * nelems + frag.offset),
                    PTR_OFFSET(src,  grank * nelems + frag.offset),
                    frag.len, dst_mtype, frag.peer,
                    get_lmemh, get_rkey, team, task),
                task, out);
        } else {
            UCPCHECK_GOTO(
                ucc_tl_ucp_put_nb(
                    PTR_OFFSET(src,  frag.peer * nelems + frag.offset),
                    PTR_OFFSET(dest, grank * nelems + frag.offset),
                    frag.len, src_mtype, frag.peer,
                    put_lmemh, put_rkey, team, task),
                task, out);
        }

        total_posted = task->onesided.get_posted + task->onesided.put_posted;
        inflight     = total_posted -
                       (task->onesided.get_completed + task->onesided.put_completed);
        if (inflight >= ntokens) {
            polls = 0;
            while (polls++ < npolls) {
                ucp_worker_progress(TASK_CTX(task)->worker.ucp_worker);
                inflight = (task->onesided.get_posted + task->onesided.put_posted) -
                           (task->onesided.get_completed +
                            task->onesided.put_completed);
                if (inflight < ntokens) {
                    break;
                }
            }
            if (inflight >= ntokens) {
                return;
            }
        }
    }

    /* Batch-flush all remote (PUT) peers after all ops are posted so the NIC
     * can process them concurrently. Per-peer flush inside the loop serialised
     * them when peers were laid out node-by-node.
     * flush_posted == 0 guards against re-issuing on subsequent progress calls. */
    if (task->flush_posted == 0) {
        for (pos = 0; pos < (uint32_t)gsize; pos++) {
            is_local_peer = local_first ? (int)(pos < (uint32_t)n_local)
                                        : (int)(pos >= (uint32_t)(gsize - n_local));
            if (!is_local_peer) {
                UCPCHECK_GOTO(ucc_tl_ucp_ep_flush(order[pos], team, task),
                              task, out);
            }
        }
    }

    alltoall_onesided_wait_completion(task, npolls);
out:
    return;
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_alltoall_onesided_init(ucc_base_coll_args_t *coll_args,
                                               ucc_base_team_t *team,
                                               ucc_coll_task_t **task_h)
{
    ucc_schedule_t              *schedule = NULL;
    ucc_tl_ucp_team_t           *tl_team  =
        ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t         barrier_coll_args = {
        .team = team->params.team,
        .args.coll_type = UCC_COLL_TYPE_BARRIER,
    };
    size_t                       perc_bw     =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoall_onesided_percent_bw;
    uint32_t                     nfrags_cfg  =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoall_onesided_nfrags;
    size_t                       frag_size   =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoall_onesided_frag_size;
    ucc_tl_ucp_alltoall_onesided_alg_t alg   =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoall_onesided_alg;
    ucc_tl_ucp_alltoall_onesided_order_t order =
        UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.alltoall_onesided_order;
    ucc_tl_ucp_schedule_t       *tl_schedule = NULL;
    ucc_rank_t                   group_size  = 1;
    ucc_coll_task_t             *barrier_task;
    ucc_coll_task_t             *a2a_task;
    ucc_tl_ucp_task_t           *task;
    ucc_status_t                 status;
    size_t                       nelems;
    double                       rate;
    size_t                       ratio;
    double                       tokens_raw;
    size_t                       per_peer_bytes;
    uint32_t                     nfrags;
    ucp_ep_h                     ep;
    ucp_ep_evaluate_perf_param_t param;
    ucp_ep_evaluate_perf_attr_t  attr;
    int64_t                      npolls;
    ucc_sbgp_t                  *sbgp;

    ALLTOALL_TASK_CHECK(coll_args->args, tl_team);
    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) ||
        (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS &&
            (!(coll_args->args.flags &
               UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)))) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "non memory mapped buffers are not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        return status;
    }

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH)) {
        coll_args->args.src_memh.global_memh = NULL;
    } else {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_SRC_MEMH_GLOBAL)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                "onesided alltoall requires global memory handles for src buffers");
            status = UCC_ERR_INVALID_PARAM;
            return status;
        }
    }

    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH)) {
        coll_args->args.dst_memh.global_memh = NULL;
    } else {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_DST_MEMH_GLOBAL)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                "onesided alltoall requires global memory handles for dst buffers");
            status = UCC_ERR_INVALID_PARAM;
            return status;
        }
    }
    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                     (ucc_tl_ucp_schedule_t **)&tl_schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    schedule = &tl_schedule->super.super;
    ucc_schedule_init(schedule, coll_args, team);
    schedule->super.post     = ucc_tl_ucp_alltoall_onesided_sched_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_tl_ucp_alltoall_onesided_sched_finalize;

    sbgp = ucc_topo_get_sbgp(tl_team->topo, UCC_SBGP_NODE);
    if (sbgp->status == UCC_SBGP_NOT_EXISTS) {
        /* 1 PPN, use put */
        if (alg == UCC_TL_UCP_ALLTOALL_ONESIDED_AUTO) {
            alg = UCC_TL_UCP_ALLTOALL_ONESIDED_PUT;
        }
    } else {
        group_size = sbgp->group_size;
    }

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    if (!task) {
        status = UCC_ERR_NO_MEMORY;
        goto out;
    }
    task->super.finalize = ucc_tl_ucp_alltoall_onesided_finalize;
    a2a_task             = &task->super;

    status = alltoall_onesided_build_peer_order(task, tl_team, sbgp, alg);
    if (status != UCC_OK) {
        goto out_task;
    }

    status = ucc_tl_ucp_coll_init(&barrier_coll_args, team, &barrier_task);
    if (status != UCC_OK) {
        goto out_task;
    }
    if (perc_bw > 100) {
        perc_bw = 100;
    } else if (perc_bw == 0) {
        perc_bw = 1;
    }

    nelems             = TASK_ARGS(task).src.info.count;
    nelems             = nelems / UCC_TL_TEAM_SIZE(tl_team);
    param.field_mask   = UCP_EP_PERF_PARAM_FIELD_MESSAGE_SIZE;
    attr.field_mask    = UCP_EP_PERF_ATTR_FIELD_ESTIMATED_TIME;
    param.message_size = nelems * ucc_dt_size(TASK_ARGS(task).src.info.datatype);;
    ucc_tl_ucp_get_ep(
        tl_team, (UCC_TL_TEAM_RANK(tl_team) + 1) % UCC_TL_TEAM_SIZE(tl_team),
        &ep);
    ucp_ep_evaluate_perf(ep, &param, &attr);

    rate  = (1 / attr.estimated_time) * (double)(perc_bw / 100.0);
    ratio = (nelems > 0) ? nelems * group_size : 1;

    tokens_raw     = (ratio > 0) ? rate / (double)ratio : 1.0;
    per_peer_bytes = nelems * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    nfrags         = 1;
    if (nfrags_cfg == 0) {
        if (frag_size > 0) {
            if (per_peer_bytes > frag_size) {
                nfrags = (uint32_t)((per_peer_bytes + frag_size - 1) /
                                    frag_size);
            }
        } else if (tokens_raw <= 1.0 &&
                   per_peer_bytes > ALLTOALL_ONESIDED_MIN_FRAG_SZ) {
            /* tokens_raw <= 0 (unusable perf estimate) => fragment maximally
             * and let the size floor below bound it */
            nfrags = (tokens_raw > 0.0) ? (uint32_t)ceil(1.0 / tokens_raw)
                                        : ALLTOALL_ONESIDED_MAX_FRAGS;
        }
    } else {
        /* explicit override (1 => disabled, >1 => forced fragment count) */
        nfrags = nfrags_cfg;
    }
    if (nfrags > 1 && per_peer_bytes > ALLTOALL_ONESIDED_MIN_FRAG_SZ) {
        nfrags = ucc_min(nfrags, ALLTOALL_ONESIDED_MAX_FRAGS);
        nfrags = ucc_min(nfrags,
                         (uint32_t)(per_peer_bytes / ALLTOALL_ONESIDED_MIN_FRAG_SZ));
        nfrags = ucc_max(nfrags, 1u);
    } else {
        nfrags = 1;
    }
    if ((uint64_t)nfrags * UCC_TL_TEAM_SIZE(tl_team) > UINT32_MAX) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "onesided alltoall fragment count overflow (nfrags %u)",
                 nfrags);
        status = UCC_ERR_NOT_SUPPORTED;
        goto out_task;
    }
    task->alltoall_onesided.nfrags = nfrags;
    task->alltoall_onesided.tokens = (uint32_t)ucc_max(1.0, tokens_raw * nfrags);
    tl_debug(UCC_TL_TEAM_LIB(tl_team),
             "onesided alltoall: per_peer_bytes %zu tokens_raw %.3f nfrags %u "
             "tokens %u (nfrags_cfg %u frag_size %zu)",
             per_peer_bytes, tokens_raw, nfrags,
             task->alltoall_onesided.tokens, nfrags_cfg, frag_size);
    task->super.post = ucc_tl_ucp_alltoall_onesided_start;
    npolls           = task->n_polls;
    if (alg == UCC_TL_UCP_ALLTOALL_ONESIDED_MIXED ||
        (alg == UCC_TL_UCP_ALLTOALL_ONESIDED_AUTO &&
         order == UCC_TL_UCP_ALLTOALL_ONESIDED_ORDER_FULL &&
         sbgp->status == UCC_SBGP_ENABLED &&
         task->alltoall_onesided.n_local > 0 &&
         task->alltoall_onesided.n_local < (ucc_rank_t)UCC_TL_TEAM_SIZE(tl_team))) {
        npolls = nelems * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
        if (npolls < task->n_polls) {
            npolls = task->n_polls;
        }
        task->super.progress = ucc_tl_ucp_alltoall_onesided_mixed_progress;
    } else if (alg == UCC_TL_UCP_ALLTOALL_ONESIDED_GET ||
               (alg == UCC_TL_UCP_ALLTOALL_ONESIDED_AUTO &&
                group_size >= CONGESTION_THRESHOLD)) {
        npolls = nelems * ucc_dt_size(TASK_ARGS(task).src.info.datatype);
        if (npolls < task->n_polls) {
            npolls = task->n_polls;
        }
        task->super.progress = ucc_tl_ucp_alltoall_onesided_get_progress;
    } else {
        task->super.progress = ucc_tl_ucp_alltoall_onesided_put_progress;
    }
    task->alltoall_onesided.npolls = npolls;

    ucc_schedule_add_task(schedule, a2a_task);
    ucc_task_subscribe_dep(&schedule->super, a2a_task,
                           UCC_EVENT_SCHEDULE_STARTED);

    ucc_schedule_add_task(schedule, barrier_task);
    ucc_task_subscribe_dep(a2a_task, barrier_task,
                           UCC_EVENT_COMPLETED);
    *task_h = &schedule->super;
    return status;
out_task:
    ucc_tl_ucp_alltoall_onesided_finalize(&task->super);
out:
    if (tl_schedule) {
        ucc_tl_ucp_put_schedule(&tl_schedule->super.super);
    }
    return status;
}
