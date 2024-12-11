/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "gather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "coll_patterns/sra_knomial.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->gather_kn.phase = _phase;                                        \
    } while (0)

static inline uint32_t calc_buffer_size(ucc_rank_t vrank, uint32_t radix,
                                        ucc_rank_t tsize)
{
    uint32_t radix_valuation;

    if (vrank == 0) {
        return tsize;
    }

    radix_valuation = calc_valuation(vrank, radix);
    return (uint32_t)ucc_min(pow(radix, radix_valuation), tsize - vrank);
}

/* gather knomial is used as regular gather collective and as part of reduce SRG */
void ucc_tl_ucp_gather_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t     *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t       *args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t     *team      = TASK_TEAM(task);
    ucc_rank_t             tsize     = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t             rank      = UCC_TL_TEAM_RANK(team);
    ucc_rank_t             root      = (ucc_rank_t)args->root;
    uint32_t               radix     = task->gather_kn.radix;
    ucc_rank_t             vrank     = VRANK(rank, root, tsize);
    ucc_memory_type_t      mtype     = args->src.info.mem_type;
    ucc_status_t           status    = UCC_OK;
    ucc_knomial_pattern_t *p         = &task->gather_kn.p;
    size_t                 dt_size   = ucc_dt_size(args->src.info.datatype);
    size_t                 data_size = args->src.info.count * dt_size;
    ucc_coll_type_t        ct        = args->coll_type;
    size_t msg_size, peer_seg_count;
    void *scratch_offset;
    ucc_rank_t vpeer, peer, vroot_at_level, root_at_level, num_blocks;
    ucc_kn_radix_t loop_step;
    ptrdiff_t peer_seg_offset;

    if (task->gather_kn.p.node_type == KN_NODE_EXTRA) {
        ucc_assert(ct == UCC_COLL_TYPE_REDUCE);
        task->super.status = UCC_OK;
        return;
    }

    UCC_GATHER_KN_GOTO_PHASE(task->gather_kn.phase);
UCC_GATHER_KN_PHASE_INIT:
    while (!ucc_knomial_pattern_loop_done(p)) {
        if (task->tagged.send_posted > 0) {
            goto UCC_GATHER_KN_PHASE_PROGRESS;
        }

        scratch_offset = task->gather_kn.scratch;
        vroot_at_level = ucc_knomial_pattern_get_base_rank(p, vrank);
        if (vroot_at_level == vrank) {
            for (loop_step = 1; loop_step < radix; loop_step++) {
                vpeer = ucc_knomial_pattern_get_loop_peer(p, vrank, loop_step);
                if (vpeer == UCC_KN_PEER_NULL) {
                    continue;
                }
                ucc_kn_g_pattern_peer_seg(vpeer, p, &peer_seg_count,
                                          &peer_seg_offset);
                peer = INV_VRANK(vpeer, root, tsize);
                if (vrank != 0) {
                    msg_size = peer_seg_count * dt_size;
                    if (args->coll_type != UCC_COLL_TYPE_GATHER) {
                        scratch_offset = PTR_OFFSET(task->gather_kn.scratch,
                                                    peer_seg_offset * dt_size);
                    } else {
                        scratch_offset = PTR_OFFSET(scratch_offset,
                                                    data_size *
                                                    task->gather_kn.dist);
                    }
                    UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(scratch_offset,
                                                     msg_size, mtype, peer,
                                                     team, task),
                                  task, out);
                } else {
                    /*
                     the root is a particular case because it must aggregate
                     the data sorted by ranks
                    */
                    scratch_offset = PTR_OFFSET(task->gather_kn.scratch,
                                                data_size * peer);
                    num_blocks = ucc_min(task->gather_kn.dist, tsize - vpeer);
                    /* check if received data correspond to contiguous ranks */
                    if ((ct == UCC_COLL_TYPE_REDUCE) ||
                        (num_blocks <= tsize - peer)) {
                        msg_size = peer_seg_count * dt_size;
                        if (args->coll_type != UCC_COLL_TYPE_GATHER) {
                            scratch_offset = PTR_OFFSET(task->gather_kn.scratch,
                                                        peer_seg_offset * dt_size);
                        }
                        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(scratch_offset,
                                                         msg_size, mtype,
                                                         peer, team, task),
                                        task, out);
                    } else {
                        /*
                         in this case, data must be split in two
                         at the destination buffer
                        */
                        msg_size = data_size * (tsize - peer);
                        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(scratch_offset,
                                                         msg_size, mtype,
                                                         peer, team, task),
                                      task, out);
                        msg_size = data_size * (num_blocks - (tsize - peer));
                        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(task->gather_kn.scratch,
                                                         msg_size, mtype,
                                                         peer, team, task),
                                      task, out);
                    }
                }
            }

            if ((ct != UCC_COLL_TYPE_REDUCE)  &&
                ucc_knomial_pattern_loop_first_iteration(p)) {
                msg_size = data_size;
                if (rank != root) {
                    status = ucc_mc_memcpy(task->gather_kn.scratch,
                                           args->src.info.buffer, msg_size,
                                           args->src.info.mem_type, mtype);
                } else if (!UCC_IS_INPLACE(*args)) {
                    status = ucc_mc_memcpy(
                        PTR_OFFSET(task->gather_kn.scratch, data_size * rank),
                        args->src.info.buffer, msg_size,
                        args->src.info.mem_type, mtype);
                }

                if (ucc_unlikely(UCC_OK != status)) {
                    task->super.status = status;
                    return;
                }
            } else {
                if (rank == root && ucc_knomial_pattern_loop_first_iteration(p) && !UCC_IS_INPLACE(*args)) {
                    ucc_kn_g_pattern_peer_seg(vrank, p, &peer_seg_count,
                                              &peer_seg_offset);
                    status = ucc_mc_memcpy(
                        PTR_OFFSET(task->gather_kn.scratch, peer_seg_offset * dt_size),
                        PTR_OFFSET(args->src.info.buffer, peer_seg_offset * dt_size), peer_seg_count * dt_size,
                        args->src.info.mem_type, mtype);
                }
            }
        } else {
            root_at_level = INV_VRANK(vroot_at_level, root, tsize);
            num_blocks    = ucc_min(task->gather_kn.dist, tsize - vrank);
            if ((ct == UCC_COLL_TYPE_REDUCE) ||
                (root_at_level != root) ||
                (num_blocks <= tsize - rank)) {
                ucc_kn_g_pattern_peer_seg(vrank, p, &peer_seg_count,
                                          &peer_seg_offset);
                msg_size = peer_seg_count * dt_size;
                if (args->coll_type == UCC_COLL_TYPE_GATHER) {
                    scratch_offset = task->gather_kn.scratch;
                } else {
                    scratch_offset = PTR_OFFSET(task->gather_kn.scratch,
                                                peer_seg_offset * dt_size);
                }
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(scratch_offset,
                                                 msg_size, mtype,
                                                 root_at_level, team, task),
                                task, out);
            } else {
                // need to split in this case due to root and tree topology
                msg_size = data_size * (tsize - rank);
                UCPCHECK_GOTO(ucc_tl_ucp_send_nb(task->gather_kn.scratch,
                                                 msg_size, mtype,
                                                 root_at_level, team, task),
                                task, out);
                msg_size = data_size * (num_blocks - (tsize - rank));
                UCPCHECK_GOTO(
                    ucc_tl_ucp_send_nb(PTR_OFFSET(task->gather_kn.scratch,
                                                  data_size * (tsize - rank)),
                                       msg_size, mtype, root_at_level, team,
                                       task),
                    task, out);
            }
        }

UCC_GATHER_KN_PHASE_PROGRESS:
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            SAVE_STATE(UCC_GATHER_KN_PHASE_PROGRESS);
            return;
        }
        task->gather_kn.dist *= radix;
        ucc_kn_g_pattern_next_iter(p);
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_gather_kn_done", 0);
out:
    return;
}

ucc_status_t ucc_tl_ucp_gather_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         root  = (ucc_rank_t)args->root;
    ucc_rank_t         trank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size  = UCC_TL_TEAM_SIZE(team);

    if (root == trank && UCC_IS_INPLACE(*args)) {
        args->src.info       = args->dst.info;
        args->src.info.count = args->dst.info.count / size;
    }

    if (args->coll_type == UCC_COLL_TYPE_GATHER) {
        ucc_kn_g_pattern_init(size, VRANK(trank, root, size),
                              task->gather_kn.radix, args->src.info.count * size,
                              &task->gather_kn.p);
    } else {
        /* reduce srg */
        ucc_assert(args->coll_type == UCC_COLL_TYPE_REDUCE);
        task->gather_kn.scratch = args->dst.info.buffer;
        ucc_kn_gx_pattern_init(size, VRANK(trank, root, size),
                               task->gather_kn.radix, args->dst.info.count,
                               &task->gather_kn.p);
    }

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_gather_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    task->gather_kn.dist  = 1;
    task->gather_kn.phase = UCC_GATHER_KN_PHASE_INIT;

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}

ucc_status_t ucc_tl_ucp_gather_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    if (task->gather_kn.scratch_mc_header) {
        ucc_mc_free(task->gather_kn.scratch_mc_header);
    }
    return ucc_tl_ucp_coll_finalize(coll_task);
}

ucc_status_t ucc_tl_ucp_gather_knomial_init_common(ucc_tl_ucp_task_t *task,
                                                   ucc_kn_radix_t radix)
{
    ucc_tl_ucp_team_t *team   = TASK_TEAM(task);
    ucc_rank_t         trank  = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize  = UCC_TL_TEAM_SIZE(team);
    ucc_coll_args_t   *args   = &TASK_ARGS(task);
    ucc_rank_t         root   = args->root;
    ucc_rank_t         vrank  = VRANK(trank, root, tsize);
    ucc_status_t       status = UCC_OK;
    ucc_memory_type_t  mtype;
    ucc_datatype_t     dt;
    size_t             count, data_size;
    uint32_t           buffer_size;
    int                is_leaf;

    if (UCC_IS_ROOT(*args, trank)) {
        count = args->dst.info.count;
        dt    = args->dst.info.datatype;
        mtype = args->dst.info.mem_type;
    } else {
        count = args->src.info.count;
        dt    = args->src.info.datatype;
        mtype = args->src.info.mem_type;
    }
    data_size             = count * ucc_dt_size(dt);
    task->super.post      = ucc_tl_ucp_gather_knomial_start;
    task->super.progress  = ucc_tl_ucp_gather_knomial_progress;
    task->super.finalize  = ucc_tl_ucp_gather_knomial_finalize;
    task->gather_kn.radix = radix;
    CALC_KN_TREE_DIST(tsize, task->gather_kn.radix,
                      task->gather_kn.max_dist);
    task->gather_kn.scratch_mc_header = NULL;

    if (args->coll_type == UCC_COLL_TYPE_REDUCE) {
        task->gather_kn.scratch = args->dst.info.buffer;
    } else {
        ucc_assert(args->coll_type == UCC_COLL_TYPE_GATHER);
        is_leaf = ((vrank % radix != 0) || (vrank == tsize - 1));
        if (vrank == 0) {
            task->gather_kn.scratch = args->dst.info.buffer;
        } else if (is_leaf) {
            task->gather_kn.scratch = args->src.info.buffer;
        } else {
            buffer_size = calc_buffer_size(vrank, task->gather_kn.radix, tsize);
            status      = ucc_mc_alloc(&task->gather_kn.scratch_mc_header,
                                       buffer_size * data_size, mtype);
            task->gather_kn.scratch = task->gather_kn.scratch_mc_header->addr;
        }
    }

    return status;
}

ucc_status_t ucc_tl_ucp_gather_knomial_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *team,
                                            ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t         tsize    = UCC_TL_TEAM_SIZE(tl_team);
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;
    ucc_kn_radix_t radix;

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    radix = ucc_min(UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.gather_kn_radix, tsize);

    status = ucc_tl_ucp_gather_knomial_init_common(task, radix);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_gather_knomial_init_r(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *team,
                                              ucc_coll_task_t **task_h,
                                              ucc_kn_radix_t radix)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    if (ucc_unlikely(!task)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_tl_ucp_gather_knomial_init_common(task, radix);
    if (ucc_unlikely(status != UCC_OK)) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
