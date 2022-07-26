/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "core/ucc_progress_queue.h"
#include "components/mc/ucc_mc.h"
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

/* Calculates for each rank at which distance it should recieve */
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

void ucc_tl_ucp_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucc_kn_radix_t         radix     = task->scatter_kn.p.radix;
    uint8_t                node_type = task->scatter_kn.p.node_type;
    ucc_knomial_pattern_t *p         = &task->scatter_kn.p;
    void                  *rbuf      = args->dst.info.buffer;
    ucc_memory_type_t      mem_type  = args->src.info.mem_type;
    size_t                 count     = args->src.info.count;
    ucc_datatype_t         dt        = args->src.info.datatype;
    size_t                 dt_size   = ucc_dt_size(dt);
    ucc_rank_t             root      = (ucc_rank_t)args->root;
    ucc_rank_t             size      = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t             rank      = VRANK(UCC_TL_TEAM_RANK(team), root, size);
    ucc_rank_t             team_size = size - p->n_extra;
    void                  *sbuf;
    ucc_rank_t             peer, vroot, vpeer, peer_recv_dist;
    ucc_rank_t             step_radix, peer_seg_index, local_seg_index;
    ptrdiff_t              peer_seg_offset, offset;
    ucc_status_t           status;
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;

    root = VRANK(root, root, size);

    if (task->scatter_kn.phase == UCC_SCATTER_KN_PHASE_LOOP) {
        goto UCC_SCATTER_KN_PHASE_LOOP;
    }

    if (KN_NODE_EXTRA == node_type) {
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

        /*
         Each rank's recieve (beside's root) must only happen once,
         and at its correct distance which is previously calclulated and saved
         in task->scatter_kn.recv_dist.
         Receive will only occur in the following iteration to that of
         it's parent's send.
        */
        if (rank != root && task->scatter_kn.recv_dist == p->radix_pow &&
            task->tagged.recv_posted == 0) {
            for (loop_step = 1; loop_step < radix; loop_step++) {
                peer = ucc_knomial_pattern_get_loop_peer(p, rank, size,
                                                             loop_step);
                if (peer == UCC_KN_PEER_NULL)
                    continue;
                vpeer = ucc_knomial_pattern_loop_rank(p, peer);
                vroot = ucc_knomial_pattern_loop_rank(p, root);
                peer_recv_dist =
                    calc_recv_dist(team_size, vpeer, radix, vroot);
                if (peer_recv_dist < task->scatter_kn.recv_dist) {
                    UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(rbuf,
                                  local_seg_count * dt_size, mem_type,
                                  INV_VRANK(peer, (ucc_rank_t)args->root,
                                  size), team, task), task, out);
                    goto UCC_SCATTER_KN_PHASE_LOOP;
                }
            }
        }

        /*
         Each non leaf rank will send per iteration to up to radix - 1
         "children" who are within it's current distance.
         Distance is initialized to 1 and each iteration is multiplied by radix.
         Each rank's send (besides leaf ranks) happens only after it's recieve
         from previous iteration has completed.
        */
        if (root == rank ||
            (task->tagged.recv_posted > 0 &&
             task->tagged.recv_posted == task->tagged.recv_completed)) {
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
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf,
                    peer_seg_offset * dt_size + task->scatter_kn.send_offset),
                    peer_seg_count * dt_size, mem_type, INV_VRANK(peer,
                    (ucc_rank_t)args->root, size), team, task), task, out);
            }
            local_seg_index =
                ucc_sra_kn_compute_seg_index(rank, p->radix_pow, p);
            offset = ucc_sra_kn_compute_seg_offset(
                block_count, step_radix, local_seg_index);
            task->scatter_kn.send_offset += offset * dt_size;
        }

UCC_SCATTER_KN_PHASE_LOOP:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_SCATTER_KN_PHASE_LOOP);
            return;
        }
        ucc_knomial_pattern_next_iteration(p);
    }

    ucc_sra_kn_get_offset_and_seglen(count, dt_size, rank, size, radix,
                                     &offset, &local_seg_count);
    if (offset != 0) {
        status = ucc_mc_memcpy(PTR_OFFSET(args->dst.info.buffer, offset),
                               PTR_OFFSET(rbuf, task->scatter_kn.send_offset),
                               local_seg_count * dt_size, mem_type, mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            task->super.status = status;
            return;
        }
    }
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_scatter_kn_done", 0);
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tl_ucp_scatter_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t     *team = TASK_TEAM(task);
    ucc_rank_t             size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t             rank = UCC_TL_TEAM_RANK(team);
    ucc_knomial_pattern_t *p    = &task->scatter_kn.p;
    ucc_rank_t             vrank, vroot, root;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_scatter_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    root = coll_task->bargs.args.root;
    ucc_knomial_pattern_init(size, VRANK(rank, root, size),
                             task->scatter_kn.p.radix, &task->scatter_kn.p);
    task->scatter_kn.phase = UCC_SCATTER_KN_PHASE_INIT;
    vroot = ucc_knomial_pattern_loop_rank(p, VRANK(root, root, size));
    vrank = ucc_knomial_pattern_loop_rank(p, VRANK(rank, root, size));
    task->scatter_kn.recv_dist = calc_recv_dist(size - p->n_extra, vrank,
                                                p->radix, vroot);
    task->scatter_kn.send_offset = 0;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
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
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         size    = UCC_TL_TEAM_SIZE(tl_team);
    ucc_rank_t         rank    = UCC_TL_TEAM_RANK(tl_team);
    ucc_tl_ucp_task_t *task;

    /* In place currently not supported */
    if (UCC_IS_INPLACE(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post     = ucc_tl_ucp_scatter_knomial_start;
    task->super.progress = ucc_tl_ucp_scatter_knomial_progress;
    task->super.finalize = ucc_tl_ucp_scatter_knomial_finalize;

    ucc_assert(coll_args->args.src.info.mem_type ==
               coll_args->args.dst.info.mem_type);

    ucc_knomial_pattern_init(size, rank, radix, &task->scatter_kn.p);

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *     team,
                                       ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    size_t             count   = coll_args->args.src.info.count;
    ucc_kn_radix_t     radix, cfg_radix;

    cfg_radix = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.scatter_kn_radix;
    radix = ucc_knomial_pattern_get_min_radix(cfg_radix,
                                              UCC_TL_TEAM_SIZE(tl_team), count);
    return ucc_tl_ucp_scatter_knomial_init_r(coll_args, team, task_h, radix);
}
