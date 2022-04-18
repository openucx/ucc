/**
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "tl_ucp_coll.h"
#include "tl_ucp_sendrecv.h"
#include "send.h"

ucc_status_t ucc_tl_ucp_send_init(ucc_tl_ucp_task_t *task)
{
    task->super.post     = ucc_tl_ucp_send_start;
    task->super.progress = ucc_tl_ucp_send_progress;

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_send_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t   *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t   *team       = TASK_TEAM(task);
    ucc_coll_args_t     *args       = &TASK_ARGS(task);
    uint64_t             peer       = args->root;
    void                *sbuf       = args->src.info.buffer;
    ucc_memory_type_t    mem_type   = args->src.info.mem_type;
    size_t               count      = args->src.info.count;
    ucc_datatype_t       dt         = args->src.info.datatype;
    size_t               dt_size    = ucc_dt_size(dt);
    size_t               data_size  = count * dt_size;

    UCPCHECK_GOTO(ucc_tl_ucp_send_nb(sbuf, data_size, mem_type, peer, team, task), task, out);

    task->super.status = UCC_INPROGRESS;
    ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
    return UCC_OK;

out:
    return task->super.status;
}

void ucc_tl_ucp_send_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t   *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);

    task->super.status = ucc_tl_ucp_test(task);
}
