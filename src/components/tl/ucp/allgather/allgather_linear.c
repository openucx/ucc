/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "components/mc/ucc_mc.h"

ucc_status_t ucc_tl_ucp_allgather_linear_start(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_ucp_allgather_linear_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t      *team,
                                              ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_task_t *task = ucc_tl_ucp_init_task(coll_args, team);

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        ucc_tl_ucp_put_task(task);
        return UCC_ERR_NOT_SUPPORTED;
    }

    task->super.post     = ucc_tl_ucp_allgather_linear_start;
    task->super.progress = ucc_tl_ucp_allgather_linear_progress;
    *task_h              = &task->super;

    return UCC_OK;
}

void ucc_tl_ucp_allgather_linear_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize     = UCC_TL_TEAM_SIZE(team);
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             count     = TASK_ARGS(task).dst.info.count;
    size_t             data_size = (count / tsize) * ucc_dt_size(dt);
    void              *tmprecv, *tmpsend;
    ucc_rank_t         peer;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    while (task->tagged.send_posted < tsize - 1) {
        peer    = (trank + 1 + task->tagged.send_posted) % tsize;
        tmpsend = PTR_OFFSET(rbuf, trank * data_size);
        tmprecv = PTR_OFFSET(rbuf, peer * data_size);

        /* Send my data to peer */
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(tmpsend, data_size, rmem, peer, team, task),
            task, out);
        /* Receive peer's data at peer's offset */
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(tmprecv, data_size, rmem, peer, team, task),
            task, out);
    }

    /* check if ucp task is complete if it is not complete, yield task */
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;

out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_linear_done", 0);
}

ucc_status_t ucc_tl_ucp_allgather_linear_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    size_t             count     = TASK_ARGS(task).dst.info.count;
    void              *sbuf      = TASK_ARGS(task).src.info.buffer;
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  smem      = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    ucc_rank_t         trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize     = UCC_TL_TEAM_SIZE(team);
    size_t             data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_linear_start",
                                     0);

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    /* Copy local data to the receive buffer if not in-place */
    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, data_size * trank), sbuf,
                               data_size, rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}
