/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "bcast.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"

void ucc_tl_ucp_bcast_knomial_progress(ucc_coll_task_t *coll_task)
{
    uint32_t           i;

    ucc_rank_t         vrank;
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         rank      = task->subset.myrank;
    ucc_rank_t         size      = (ucc_rank_t)task->subset.map.ep_num;

    uint32_t           radix     = task->bcast_kn.radix;
    ucc_rank_t         root      = (ucc_rank_t)TASK_ARGS(task).root;
    ucc_rank_t         dist      = task->bcast_kn.dist;
    void              *buffer    = TASK_ARGS(task).src.info.buffer;
    ucc_memory_type_t  mtype     = TASK_ARGS(task).src.info.mem_type;
    size_t             data_size = TASK_ARGS(task).src.info.count *
                       ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    ucc_rank_t vpeer, peer, vroot_at_level, root_at_level, pos;

    if (UCC_COLL_ARGS_ACTIVE_SET(&(TASK_ARGS(task)))) {
        root = ucc_ep_map_local_rank(task->subset.map, root);
    }

    vrank = (rank - root + size) % size;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }
    while (dist >= 1) {
        if (vrank % dist == 0) {
            pos = (vrank / dist) % radix;
            if (pos == 0) {
                for (i = radix - 1; i >= 1; i--) {
                    vpeer = vrank + i * dist;
                    if (vpeer < size) {
                        peer = ucc_ep_map_eval(task->subset.map,
                                               (vpeer + root) % size);
                        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(buffer, data_size,
                                                         mtype, peer, team,
                                                         task),
                                      task, out);
                    }
                }
            } else {
                vroot_at_level = vrank - pos * dist;
                root_at_level  = (vroot_at_level + root) % size;
                peer = ucc_ep_map_eval(task->subset.map,
                                       root_at_level);
                UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(buffer, data_size, mtype,
                                                 peer, team, task),
                              task, out);
            }
        }
        dist /= radix;
        if (UCC_INPROGRESS == ucc_tl_ucp_test_recv(task)) {
            task->bcast_kn.dist = dist;
            return;
        }
    }
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        task->bcast_kn.dist = dist;
        return;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_bcast_kn_done", 0);
out:
    return;
}

ucc_status_t ucc_tl_ucp_bcast_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_rank_t         size = (ucc_rank_t)task->subset.map.ep_num;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_bcast_kn_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    CALC_KN_TREE_DIST(size, task->bcast_kn.radix, task->bcast_kn.dist);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}
