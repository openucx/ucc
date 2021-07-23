/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "core/ucc_mc.h"

ucc_status_t ucc_tl_ucp_allgather_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ucc_rank_t         group_rank = team->rank;
    ucc_rank_t         group_size = team->size;
    void              *rbuf       = coll_task->args.dst.info.buffer;
    ucc_memory_type_t  rmem       = coll_task->args.dst.info.mem_type;
    size_t             count      = coll_task->args.dst.info.count;
    ucc_datatype_t     dt         = coll_task->args.dst.info.datatype;
    size_t             data_size  = count * ucc_dt_size(dt);
    ucc_rank_t         sendto     = (group_rank + 1) % group_size;
    ucc_rank_t         recvfrom   = (group_rank - 1 + group_size) % group_size;
    int                step;
    void              *buf;
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return task->super.super.status;
    }

    while (task->send_posted < group_size - 1) {
        step = task->send_posted;
        buf  = (void *)((ptrdiff_t)rbuf +
                       ((group_rank - step + group_size) % group_size) *
                           data_size);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(buf, data_size, rmem, sendto, team, task),
            task, out);
        buf = (void *)((ptrdiff_t)rbuf +
                       ((group_rank - step - 1 + group_size) % group_size) *
                           data_size);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(buf, data_size, rmem, recvfrom, team, task),
            task, out);
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return task->super.super.status;
        }
    }
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.super.status = UCC_OK;
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_ring_done", 0);
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_allgather_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    size_t             count     = coll_task->args.dst.info.count;
    void              *sbuf      = coll_task->args.src.info.buffer;
    void              *rbuf      = coll_task->args.dst.info.buffer;
    ucc_memory_type_t  smem      = coll_task->args.src.info.mem_type;
    ucc_memory_type_t  rmem      = coll_task->args.dst.info.mem_type;
    ucc_datatype_t     dt        = coll_task->args.dst.info.datatype;
    size_t             data_size = count * ucc_dt_size(dt);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_ring_start", 0);
    ucc_tl_ucp_task_reset(task);

    if (!UCC_IS_INPLACE(coll_task->args)) {
        status = ucc_mc_memcpy((void*)((ptrdiff_t)rbuf + data_size * team->rank),
                               sbuf, data_size, rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    status = ucc_tl_ucp_allgather_ring_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
        return UCC_OK;
    }
    return ucc_task_complete(coll_task);
}
