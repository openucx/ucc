#include "config.h"
#include "tl_ucp.h"
#include "barrier.h"
#include "core/ucc_progress_queue.h"

ucc_status_t ucc_tl_ucp_barrier_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    /* TODO implement main logic */
    task->super.super.status = UCC_OK;
    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_barrier_knomial_start(ucc_coll_req_t *req)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(req, ucc_tl_ucp_task_t);
    /* TODO implement main logic */
    ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(task->team)->pq,
                         &task->super);
    return UCC_OK;
}
