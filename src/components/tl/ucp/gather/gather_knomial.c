/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "gather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"

#define SAVE_STATE(_phase)                                                     \
    do {                                                                       \
        task->gather_kn.phase = _phase;                                        \
    } while (0)

void ucc_tl_ucp_gather_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t *  args      = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         team_size = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         rank      = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size      = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         root      = (ucc_rank_t)args->root;
    uint32_t           radix     = task->gather_kn.radix;
    ucc_rank_t         vrank     = (rank - root + size) % size;
    ucc_memory_type_t  mtype     = args->src.info.mem_type;
    ucc_status_t       status    = UCC_OK;
    size_t             data_size =
        args->src.info.count * ucc_dt_size(args->src.info.datatype);
    size_t     msg_size, msg_count;
    void *     scratch_offset;
    ucc_rank_t vpeer, peer, vroot_at_level, root_at_level, pos;
    uint32_t   i;

UCC_GATHER_KN_PHASE_PROGRESS:
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    UCC_GATHER_KN_GOTO_PHASE(task->gather_kn.phase);

UCC_GATHER_KN_PHASE_INIT:
    while (task->gather_kn.dist <= task->gather_kn.max_dist) {
        scratch_offset = task->gather_kn.scratch;
        if (vrank % task->gather_kn.dist == 0) {
            pos = (vrank / task->gather_kn.dist) % radix;
            if (pos == 0) {
                for (i = 1; i < radix; i++) {
                    vpeer   = vrank + i * task->gather_kn.dist;
                    msg_count = ucc_min(task->gather_kn.dist, team_size - vpeer);
                    if (vpeer >= size) {
                        break;
                    } else if (vrank != 0) {
                        msg_size       = data_size * msg_count;
                        scratch_offset = PTR_OFFSET(
                            scratch_offset, data_size * task->gather_kn.dist);
                        peer = (vpeer + root) % size;
                        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(scratch_offset,
                                                         msg_size, mtype, peer,
                                                         team, task),
                                      task, out);
                    } else { //The root is a particular case because it must aggregate the data sorted by ranks
                        peer           = (vpeer + root) % size;
                        scratch_offset = PTR_OFFSET(task->gather_kn.scratch,
                                                    data_size * peer);
                        // check if received data correspond to contiguous ranks
                        if (msg_count <= team_size - peer) {
                            msg_size = data_size * msg_count;
                            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(scratch_offset,
                                                             msg_size, mtype,
                                                             peer, team, task),
                                          task, out);
                        } else { // in this case, data must be split in two at the destination buffer
                            msg_size = data_size * (team_size - peer);
                            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(scratch_offset,
                                                             msg_size, mtype,
                                                             peer, team, task),
                                          task, out);

                            msg_size =
                                data_size * (msg_count - (team_size - peer));
                            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(
                                              task->gather_kn.scratch, msg_size,
                                              mtype, peer, team, task),
                                          task, out);
                        }
                    }
                }

                if (task->gather_kn.dist == 1) { //check if first passage
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
                }
            } else {
                vroot_at_level = vrank - pos * task->gather_kn.dist;
                root_at_level  = (vroot_at_level + root) % size;
                msg_count      = ucc_min(task->gather_kn.dist,
                                                            team_size - vrank);
                msg_size       = data_size * msg_count;
                if (root_at_level != root || msg_count <= team_size - rank) {
                    UCPCHECK_GOTO(ucc_tl_ucp_send_nb(task->gather_kn.scratch,
                                                     msg_size, mtype,
                                                     root_at_level, team, task),
                                  task, out);
                } else {
                    msg_size = data_size * (team_size - rank);
                    UCPCHECK_GOTO(ucc_tl_ucp_send_nb(task->gather_kn.scratch,
                                                     msg_size, mtype,
                                                     root_at_level, team, task),
                                  task, out);
                    msg_size = data_size * (msg_count - (team_size - rank));
                    UCPCHECK_GOTO(
                        ucc_tl_ucp_send_nb(
                            PTR_OFFSET(task->gather_kn.scratch,
                                       data_size * (team_size - rank)),
                            msg_size, mtype, root_at_level, team, task),
                        task, out);
                }
            }
        }
        task->gather_kn.dist *= radix;
        SAVE_STATE(UCC_GATHER_KN_PHASE_INIT);
        goto UCC_GATHER_KN_PHASE_PROGRESS;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_gather_kn_done", 0);
out:
    return;
}

ucc_status_t ucc_tl_ucp_gather_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t *  args = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_rank_t         root = (ucc_rank_t)args->root;
    ucc_rank_t         rank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size = UCC_TL_TEAM_SIZE(team);

    if (root == rank && UCC_IS_INPLACE(*args)) {
        args->src.info       = args->dst.info;
        args->src.info.count = args->dst.info.count / size;
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
