/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/recursive_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "core/ucc_mc.h"

ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    task->super.super.status = UCC_OK;
    fprintf(stderr,"[%d][%s] -- [%d]\n",getpid(),__FUNCTION__,__LINE__);
    return task->super.super.status;
}

ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team = task->team;
    ucc_status_t       status;
    task->super.super.status = UCC_INPROGRESS;
    fprintf(stderr,"[%d][%s] -- [%d]\n",getpid(),__FUNCTION__,__LINE__);
    status = ucc_tl_ucp_allreduce_sra_knomial_progress(&task->super);
    if (UCC_INPROGRESS == status) {
        ucc_progress_enqueue(UCC_TL_UCP_TEAM_CORE_CTX(team)->pq, &task->super);
    } else if (status < 0) {
        return status;
    }
    return UCC_OK;
}
