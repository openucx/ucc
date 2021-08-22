/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "core/ucc_mc.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"

#define SAVE_STATE(_phase)                                                    \
    do {                                                                      \
        task->scatter_kn.phase = _phase;                                      \
    } while (0)

#define GET_BASE_PEER(_radix, _rank, _dist, _peer)                            \
    do {                                                                      \
        _peer = _rank - ((_rank / _dist) % _radix) * _dist;                   \
} while (0)

enum {
    UCC_SCATTER_KN_PHASE_INIT,
    UCC_SCATTER_KN_PHASE_LOOP, /* main loop of recursive k-ing */
};

ucc_rank_t calc_recv_dist(ucc_rank_t team_size, ucc_rank_t rank,
                                     ucc_rank_t radix, ucc_rank_t root)
{
	if (rank == root) {
		return 0;
	}
    ucc_rank_t root_base;
    ucc_rank_t dist = 1;
    GET_BASE_PEER(radix, root, dist, root_base);
    while (dist <= team_size) {
        if (rank >= root_base && rank < root_base + radix * dist) {
            break;
        }
        dist *= radix;
        GET_BASE_PEER(radix, root_base, dist, root_base);
    }
    return dist;
}

ucc_status_t
ucc_tl_ucp_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args = &coll_task->args;
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucc_kn_radix_t         radix     = task->scatter_kn.p.radix;
    uint8_t                node_type = task->scatter_kn.p.node_type;
    ucc_knomial_pattern_t *p         = &task->scatter_kn.p;
    void                  *sbuf      = args->src.info.buffer;
    void                  *rbuf      = args->dst.info.buffer;
    ucc_memory_type_t      mem_type  = args->src.info.mem_type;
    size_t                 count     = args->src.info.count;
    ucc_datatype_t         dt        = args->src.info.datatype;
    size_t                 dt_size   = ucc_dt_size(dt);
    ucc_rank_t             size      = team->size;
    ucc_rank_t             rank      = team->rank;
    ucc_rank_t             root      = (ucc_rank_t)args->root;
    ucc_rank_t             team_size = team->size - p->n_extra;
    ucc_rank_t             peer, vroot, vpeer, peer_recv_dist;
    ucc_rank_t             step_radix, peer_seg_index, local_seg_index;
    ptrdiff_t              peer_seg_offset, offset;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;

//    UCC_SCATTER_KN_GOTO_PHASE(task->scatter_kn.phase);
    if (task->scatter_kn.phase == UCC_SCATTER_KN_PHASE_LOOP) {
    	goto UCC_SCATTER_KN_PHASE_LOOP;
    }

    if (KN_NODE_EXTRA == node_type || KN_NODE_PROXY == node_type) {
        goto out;
    }

    while (!ucc_knomial_pattern_loop_done(p)) {
        step_radix  = ucc_sra_kn_compute_step_radix(rank, size, p);
        block_count = ucc_sra_kn_compute_block_count(count, rank, p);
        sbuf        = (rank == root)
                           ? args->src.info.buffer : args->dst.info.buffer;
        rbuf        = args->dst.info.buffer;
        local_seg_index = ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
        local_seg_count = ucc_sra_kn_compute_seg_size(block_count, step_radix,
                                                      local_seg_index);

        if (rank != root && task->scatter_kn.recv_dist == p->radix_pow &&
                                                      task->recv_posted == 0) {
            for (loop_step = 1; loop_step < radix; loop_step++) {
                peer = ucc_knomial_pattern_get_loop_peer(p, rank, size,
                                                             loop_step);
                if (peer == UCC_KN_PEER_NULL)
                    continue;
                vpeer = ucc_knomial_pattern_loop_rank(p, peer);
                vroot = ucc_knomial_pattern_loop_rank(p, root);
                peer_recv_dist = calc_recv_dist(team_size, vpeer, radix, vroot);
                if (peer_recv_dist < task->scatter_kn.recv_dist) {
//                	printf("inside recv, rank = %d, root = %d, task->scatter_kn.recv_dist = %d, peer_recv_dist = %d, peer = %d, vpeer = %d, p->iter = %d \n",
//                	        			rank, root, task->scatter_kn.recv_dist, peer_recv_dist, peer, vpeer, p->iteration);
                    UCPCHECK_GOTO(
                        ucc_tl_ucp_recv_nb(rbuf, local_seg_count * dt_size,
                                       mem_type, peer, team, task), task, out);
                }
                rbuf = PTR_OFFSET(rbuf, local_seg_count * dt_size);
            }
        }

        if (root == rank || (task->recv_posted > 0 &&
                             task->recv_posted == task->recv_completed)) {
            for (loop_step = 1; loop_step < radix; loop_step++) {
                peer = ucc_knomial_pattern_get_loop_peer(p, rank, size,
                                                             loop_step);
                if (peer == UCC_KN_PEER_NULL)
                    continue;
                peer_seg_index =
                    ucc_sra_kn_compute_seg_index(peer, p->radix_pow, p);
                peer_seg_count = ucc_sra_kn_compute_seg_size(
                    block_count, step_radix, peer_seg_index);
                peer_seg_offset = ucc_sra_kn_compute_seg_offset(
                    block_count, step_radix, peer_seg_index);
//                printf("inside send, rank = %d, root = %d, task->recv_posted = %d, peer = %d, p->iter = %d \n",
//                                	        			rank, root, task->recv_posted, peer, p->iteration);
                UCPCHECK_GOTO(
                    ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf,
                                       peer_seg_offset * dt_size),
                                       peer_seg_count * dt_size, mem_type,
                                       peer, team, task), task, out);
            }
        }

UCC_SCATTER_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_SCATTER_KN_PHASE_LOOP);
            return task->super.super.status;
        }
//        SAVE_STATE(UCC_SCATTER_KN_PHASE_INIT);
//        task->scatter_kn.dist *= radix;
        ucc_knomial_pattern_next_iteration(p);
    }

    step_radix      = ucc_sra_kn_compute_step_radix(rank, size, p);
    block_count     = ucc_sra_kn_compute_block_count(count, rank, p);
    local_seg_index = ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
    local_seg_count = ucc_sra_kn_compute_seg_size(block_count, step_radix,
                                                          local_seg_index);
    offset          = ucc_sra_kn_get_offset(count, dt_size, rank, size, radix);
    // check of need to do memcpy at root? and not needed if offset == 0
    if (rank != root && offset != 0) {
        status = ucc_mc_memcpy(PTR_OFFSET(args->dst.info.buffer, offset), rbuf,
                               local_seg_count * dt_size, mem_type, mem_type);
        if (UCC_OK != status) {
            return status;
        }
    }
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_scatter_kn_done", 0);
    task->super.super.status = UCC_OK;
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_scatter_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucc_knomial_pattern_t *p    = &task->scatter_kn.p;
    ucc_rank_t             vrank, vroot;
    ucc_status_t           status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_scatter_kn_start", 0);
    ucc_tl_ucp_task_reset(task);

    ucc_knomial_pattern_init(team->size, team->rank,
                             task->scatter_kn.p.radix,
                             &task->scatter_kn.p);
    task->scatter_kn.phase = UCC_SCATTER_KN_PHASE_INIT;
//    task->scatter_kn.dist  = 1;
    vroot = ucc_knomial_pattern_loop_rank(p, coll_task->args.root);
    vrank = ucc_knomial_pattern_loop_rank(p, team->rank);
    task->scatter_kn.recv_dist = calc_recv_dist(team->size - p->n_extra, vrank,
                                                p->radix, vroot);
//    printf("rank = %d, vrank = %d, root = %ld, vroot = %d, recv_dst = %d, radix = %d \n",
//    		team->rank, vrank, coll_task->args.root, vroot, task->scatter_kn.recv_dist, p->radix);

    status = ucc_tl_ucp_scatter_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}

ucc_status_t
ucc_tl_ucp_scatter_knomial_finalize(ucc_coll_task_t *coll_task)
{
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size      = tl_team->size;
    ucc_rank_t         rank      = tl_team->rank;
    ucc_tl_ucp_task_t *task;

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_scatter_knomial_finalize;

    ucc_assert(coll_args->args.src.info.mem_type ==
               coll_args->args.dst.info.mem_type);
    ucc_knomial_pattern_init(size, rank, radix, &task->scatter_kn.p);

    /* In place currently not supported */

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *     team,
                                       ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size    = tl_team->size;
    size_t             count   = coll_args->args.src.info.count;
    ucc_kn_radix_t     radix;

    radix = ucc_min(UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.bcast_kn_radix, size);
    radix = 2;
    if (((count + radix - 1) / radix * (radix - 1) > count) ||
        ((radix - 1) > count)) {
        radix = 2;
    }
    return ucc_tl_ucp_scatter_knomial_init_r(coll_args, team, task_h,
                                                    radix);
}
