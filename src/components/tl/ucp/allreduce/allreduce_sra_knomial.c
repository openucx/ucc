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
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "core/ucc_mc.h"
#include "../reduce_scatter/reduce_scatter.h"
#include "../allgather/allgather.h"

/* SRA - scatter-reduce-allgather knomial algorithm
   1. The algorithm performs collective allreduce operation as a sequence of
      K-nomial Reduce-Scatter followed by K-nomial (with the same radix K)
      allgather.
   2. In essence this is an extension of the Bi-nomial SRA algorithm algorithm
      proposed by Rabenseifner2004 (https://doi.org/10.1007/978-3-540-24685-5_1).
      The extension adds the support for arbitrary radix.
   3. The algorithm targets Large message sizes (ie. optimized for max bandwidth).
   4. If number of ranks in the team can not form a full radix subtree
      (for radix=2 this means the team size is not power of 2) then there will be
      "extra" ranks which don't participate in the main exchange loop. They
      will send the data to their "proxy" ranks in the beginning and then wait
      for the response with the final data.
   5. The knomial reduce-scatter and allgather primitives can be used separately.
      However, if they are used together as part of SRA allreduce one has to
      provide the same radix for both routines.
   6. If the allreduce is INPLACE or if a rank serves as a PROXY then the algorithm
      requires allocation of a scratch buffer of the size equal to input buffer.
   7. After the completion of reduce-scatter phase the local result (at non EXTRA
      ranks) will be located in dst buffer at offset the can be commputed by the
      routine from coll_patterns/sra_knomial.h: ucc_sra_kn_get_offset.
 */
ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    return ucc_schedule_start(schedule);
}

ucc_status_t
ucc_tl_ucp_allreduce_sra_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_tl_ucp_put_schedule(schedule);
    return UCC_OK;
}

ucc_status_t
ucc_tl_ucp_allreduce_sra_knomial_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t      *team,
                                      ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_schedule_t      *schedule = ucc_tl_ucp_get_schedule(tl_team);
    size_t               count    = coll_args->args.src.info.count;
    ucc_base_coll_args_t args     = *coll_args;
    ucc_coll_task_t     *task, *rs_task;
    ucc_status_t         status;
    ucc_kn_radix_t       radix;
    ALLREDUCE_TASK_CHECK(coll_args->args, tl_team);
    radix = ucc_min(UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allreduce_sra_kn_radix,
                    tl_team->size);

    if (((count + radix - 1) / radix * (radix - 1) > count) ||
        ((radix - 1) > count)) {
        radix = 2;
    }

    /* 1st step of allreduce: knomial reduce_scatter */
    status = ucc_tl_ucp_reduce_scatter_knomial_init_r(&args, team, &task, radix);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failed to init reduce_scatter_knomial task");
        goto out;
    }
    task->flags = UCC_COLL_TASK_FLAG_INTERNAL;
    ucc_schedule_add_task(schedule, task);
    ucc_event_manager_subscribe(&schedule->super.em, UCC_EVENT_SCHEDULE_STARTED,
                                task);
    task->handlers[UCC_EVENT_SCHEDULE_STARTED] = ucc_task_start_handler;
    rs_task                                    = task;

    /* 2nd step of allreduce: knomial allgather. 2nd task subscribes
     to completion event of reduce_scatter task. */
    args.args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
    args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    status = ucc_tl_ucp_allgather_knomial_init_r(&args, team, &task, radix);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failed to init allgather_knomial task");
        goto out;
    }
    task->flags = UCC_COLL_TASK_FLAG_INTERNAL;
    ucc_schedule_add_task(schedule, task);
    ucc_event_manager_subscribe(&rs_task->em, UCC_EVENT_COMPLETED, task);
    task->handlers[UCC_EVENT_COMPLETED] = ucc_task_start_handler;

    schedule->super.post     = ucc_tl_ucp_allreduce_sra_knomial_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_tl_ucp_allreduce_sra_knomial_finalize;
    *task_h                  = &schedule->super;
    return UCC_OK;
out:
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}
