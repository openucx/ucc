/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->scatter_kn.phase = _phase;                                       \
    } while (0)

enum {
    UCC_SCATTER_KN_PHASE_INIT,
    UCC_SCATTER_KN_PHASE_LOOP, /* main loop of recursive k-ing */
};

void ucc_tl_ucp_scatter_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task      = ucc_derived_of(coll_task,
                                                      ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team      = TASK_TEAM(task);
    ucc_kn_radix_t         radix     = task->scatter_kn.p.radix;
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
    ucc_kn_radix_t         loop_step;
    size_t                 block_count, peer_seg_count, local_seg_count;
    ucc_status_t           status;

    root = VRANK(root, root, size);
    sbuf = (rank == root) ? args->src.info.buffer : args->dst.info.buffer;

    if (task->scatter_kn.phase == UCC_SCATTER_KN_PHASE_LOOP) {
        goto UCC_SCATTER_KN_PHASE_LOOP;
    }

    if (KN_NODE_EXTRA == p->node_type) {
        goto out;
    }

    while (!ucc_knomial_pattern_loop_done(p)) {
        step_radix  = ucc_kn_compute_step_radix(p);
        block_count = ucc_sra_kn_compute_block_count(count, rank, p);
        local_seg_index = ucc_kn_compute_seg_index(rank, p->radix_pow, p);
        /*
         Each rank's receive (beside's root) must only happen once,
         and at its correct distance which is previously calclulated and saved
         in task->scatter_kn.recv_dist.
         Receive will only occur in the following iteration to that of
         it's parent's send.
        */
        if ((rank != root) && (task->scatter_kn.recv_dist == p->radix_pow)) {
            ucc_assert(task->tagged.recv_posted == 0);
            local_seg_count = ucc_sra_kn_compute_seg_size(block_count,
                                                          step_radix,
                                                          local_seg_index);
            for (loop_step = 1; loop_step < radix; loop_step++) {
                peer = ucc_knomial_pattern_get_loop_peer(p, rank, loop_step);
                if (peer == UCC_KN_PEER_NULL)
                    continue;
                vpeer = ucc_knomial_pattern_loop_rank(p, peer);
                vroot = ucc_knomial_pattern_loop_rank(p, root);
                peer_recv_dist = ucc_knomial_calc_recv_dist(team_size, vpeer,
                                                            radix, vroot);
                task->scatter_kn.recv_size = local_seg_count * dt_size;
                if (peer_recv_dist < task->scatter_kn.recv_dist) {
                    UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(
                        PTR_OFFSET(rbuf, task->scatter_kn.recv_offset),
                        local_seg_count * dt_size, mem_type,
                        INV_VRANK(peer, (ucc_rank_t)args->root, size),
                        team, task), task, out);
                    goto UCC_SCATTER_KN_PHASE_LOOP;
                }
            }
        }

        /*
         Each non leaf rank will send per iteration to up to radix - 1
         "children" who are within it's current distance.
         Distance is initialized to 1 and each iteration is multiplied by radix.
         Each rank's send (besides leaf ranks) happens only after it's receive
         from previous iteration has completed.
        */
        if ((root == rank) || (task->tagged.recv_posted > 0)) {
            ucc_assert(UCC_TL_UCP_TASK_RECV_COMPLETE(task));
            for (loop_step = 1; loop_step < radix; loop_step++) {
                peer = ucc_knomial_pattern_get_loop_peer(p, rank, loop_step);
                if (peer == UCC_KN_PEER_NULL)
                    continue;
                peer_seg_index =
                    ucc_kn_compute_seg_index(peer, p->radix_pow, p);
                peer_seg_count = ucc_sra_kn_compute_seg_size(
                    block_count, step_radix, peer_seg_index);
                peer_seg_offset = ucc_sra_kn_compute_seg_offset(
                    block_count, step_radix, peer_seg_index);
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(PTR_OFFSET(sbuf,
                    task->scatter_kn.recv_offset + peer_seg_offset * dt_size +
                    task->scatter_kn.send_offset),
                    peer_seg_count * dt_size, mem_type, INV_VRANK(peer,
                    (ucc_rank_t)args->root, size), team, task), task, out);
            }
            /*TODO: local_seg_index is always zero since rank that sends is base root? */
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

    if (task->scatter_kn.recv_offset == 0 && (rank != root)) {
        ucc_sra_kn_get_offset_and_seglen(count, dt_size, rank, size, radix,
                                         &offset, &local_seg_count);
        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, offset),
                               rbuf, task->scatter_kn.recv_size, mem_type,
                               mem_type);
        if (ucc_unlikely(status != UCC_OK)) {
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
    ucc_coll_args_t       *args = &TASK_ARGS(task);
    ucc_rank_t             root = args->root;
    ucc_rank_t             vrank, vroot;
    ucc_on_off_auto_value_t is_zcopy;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_scatter_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    ucc_knomial_pattern_init(size, VRANK(rank, root, size),
                             task->scatter_kn.p.radix, &task->scatter_kn.p);
    task->scatter_kn.phase = UCC_SCATTER_KN_PHASE_INIT;
    vroot = ucc_knomial_pattern_loop_rank(p, VRANK(root, root, size));
    vrank = ucc_knomial_pattern_loop_rank(p, VRANK(rank, root, size));
    task->scatter_kn.recv_dist = ucc_knomial_calc_recv_dist(size - p->n_extra,
                                                            vrank, p->radix,
                                                            vroot);
    task->scatter_kn.recv_offset = 0;
    is_zcopy = UCC_TL_UCP_TEAM_LIB(team)->cfg.scatter_kn_enable_recv_zcopy;
    if (((is_zcopy == UCC_CONFIG_AUTO) &&
         (args->src.info.mem_type != UCC_MEMORY_TYPE_HOST)) ||
         (is_zcopy == UCC_CONFIG_ON)) {
        ucc_sra_kn_get_offset_and_seglen(args->src.info.count,
                                         ucc_dt_size(args->src.info.datatype),
                                         VRANK(rank, root, size), size, p->radix,
                                         &task->scatter_kn.recv_offset, NULL);
    }
    task->scatter_kn.send_offset = 0;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_scatter_knomial_finalize(ucc_coll_task_t *coll_task)
{
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix)
{
    ucc_tl_ucp_task_t *task;

    ucc_assert(coll_args->args.src.info.mem_type ==
               coll_args->args.dst.info.mem_type);

    task                     = ucc_tl_ucp_init_task(coll_args, team);
    task->super.post         = ucc_tl_ucp_scatter_knomial_start;
    task->super.progress     = ucc_tl_ucp_scatter_knomial_progress;
    task->super.finalize     = ucc_tl_ucp_scatter_knomial_finalize;
    task->scatter_kn.p.radix = radix;

    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *team,
                                             ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    size_t             count   = coll_args->args.src.info.count;
    ucc_kn_radix_t     radix, cfg_radix;

    cfg_radix = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.scatter_kn_radix;
    radix = ucc_knomial_pattern_get_min_radix(cfg_radix,
                                              UCC_TL_TEAM_SIZE(tl_team), count);
    return ucc_tl_ucp_scatter_knomial_init_r(coll_args, team, task_h, radix);
}
