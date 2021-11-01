/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "bcast.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"

ucc_status_t ucc_tl_ucp_bcast_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         rank       = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         size       = UCC_TL_TEAM_SIZE(team);
    ucc_rank_t         root      = (uint32_t)TASK_ARGS(task).root;
    uint32_t           radix     = task->bcast_kn.radix;
    ucc_rank_t         vrank     = (rank - root + size) % size;
    ucc_rank_t         dist      = task->bcast_kn.dist;
    void              *buffer    = TASK_ARGS(task).src.info.buffer;
    ucc_memory_type_t  mtype     = TASK_ARGS(task).src.info.mem_type;
    size_t             data_size = TASK_ARGS(task).src.info.count *
                       ucc_dt_size(TASK_ARGS(task).src.info.datatype);
    ucc_rank_t vpeer, peer, vroot_at_level, root_at_level, pos;
    uint32_t i;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return task->super.super.status;
    }
    while (dist >= 1) {
        if (vrank % dist == 0) {
            pos = (vrank / dist) % radix;
            if (pos == 0) {
                for (i = radix - 1; i >= 1; i--) {
                    vpeer = vrank + i * dist;
                    if (vpeer < size) {
                        peer = (vpeer + root) % size;
                        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(buffer, data_size,
                                                         mtype, peer, team,
                                                         task),
                                      task, out);
                    }
                }
            } else {
                vroot_at_level = vrank - pos * dist;
                root_at_level  = (vroot_at_level + root) % size;
                UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(buffer, data_size, mtype,
                                                 root_at_level, team, task),
                              task, out);
            }
        }
        dist /= radix;
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            task->bcast_kn.dist = dist;
            return task->super.super.status;
        }
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.super.status = UCC_OK;
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_bcast_kn_done", 0);
out:
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_bcast_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_rank_t         size = UCC_TL_TEAM_SIZE(team);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_bcast_kn_start", 0);
    ucc_tl_ucp_task_reset(task);

    task->bcast_kn.radix =
        ucc_min(UCC_TL_UCP_TEAM_LIB(team)->cfg.bcast_kn_radix, size);
    CALC_KN_TREE_DIST(size, task->bcast_kn.radix, task->bcast_kn.dist);

    status = ucc_tl_ucp_bcast_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}
