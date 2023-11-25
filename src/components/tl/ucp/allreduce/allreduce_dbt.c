/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include "../reduce/reduce.h"
#include "../bcast/bcast.h"

/* RB - reduce-bcast dbt algorithm
   1. The algorithm performs collective reduce operation for large messages
      using a double binary tree, followed by double binary tree bcast.
   2. The algorithm targets Large message sizes (ie. optimized for max bandwidth),
      when knomial SRA fails to find a radix which minimizes n_extra ranks.
   5. The dbt reduce and bcast primitives can be used separately.
      However, if they are used together as part of RB allreduce, one has to
      provide the same coll_root for both routines.
   6. After the completion of reduce phase the local result will be located
      in dst buffer of root, which then must be used as root src buffer for bcast.
 */
ucc_status_t ucc_tl_ucp_allreduce_dbt_start(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t  *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_coll_args_t *args     = &schedule->super.bargs.args;
    ucc_coll_task_t *bcast_task, *reduce_task;

    reduce_task                             = schedule->tasks[0];
    reduce_task->bargs.args.src.info.buffer = args->src.info.buffer;
    reduce_task->bargs.args.dst.info.buffer = args->src.info.buffer;
    reduce_task->bargs.args.src.info.count  = args->src.info.count;
    reduce_task->bargs.args.dst.info.count  = args->src.info.count;

    bcast_task                             = schedule->tasks[1];
    bcast_task->bargs.args.dst.info.buffer = args->src.info.buffer;
    bcast_task->bargs.args.dst.info.count  = args->src.info.count;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_dbt_start", 0);
    return ucc_schedule_start(coll_task);
}

ucc_status_t
ucc_tl_ucp_allreduce_dbt_finalize(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_status_t    status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(schedule, "ucp_allreduce_dbt_done", 0);
    status = ucc_schedule_finalize(coll_task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t
ucc_tl_ucp_allreduce_dbt_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t      *team,
                                      ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t   *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t args      = *coll_args;
    ucc_rank_t           rank      = UCC_TL_TEAM_RANK(tl_team);
    ucc_rank_t           size      = UCC_TL_TEAM_SIZE(tl_team);
    ucc_rank_t           tree_root = get_root(size);
    ucc_schedule_t      *schedule;
    ucc_coll_task_t     *task, *rs_task;
    void                *buf;
    // ucc_datatype_t       dtype    = coll_args->args.src.info.datatype;
    // ucc_memory_type_t    mem_type = coll_args->args.src.info.mem_type;
    // size_t               count    = coll_args->args.src.info.count;
    ucc_datatype_t       dtype;
    ucc_memory_type_t    mem_type;
    size_t               count;
    ucc_status_t         status;

    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
        /* ActiveSets currently are only supported with KN alg */
        return ucc_tl_ucp_allreduce_knomial_init(coll_args, team, task_h);
    }

    if (UCC_IS_INPLACE(args.args)) {
        dtype    = args.args.dst.info.datatype;
        mem_type = args.args.dst.info.mem_type;
        count    = args.args.dst.info.count;
        buf      = args.args.dst.info.buffer
    } else {
        dtype    = args.args.src.info.datatype;
        mem_type = args.args.src.info.mem_type;
        count    = args.args.src.info.count;
        buf      = args.args.dst.info.buffer
    }
    args.args.root = tree_root;

    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                        (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    /* 1st step of allreduce: reduce dbt */
    args.args.dst.info.buffer   = args.args.src.info.buffer;
    args.args.dst.info.mem_type = args.args.src.info.mem_type;
    args.args.dst.info.datatype = args.args.src.info.datatype;
    args.args.dst.info.count    = args.args.src.info.count;
    UCC_CHECK_GOTO(ucc_tl_ucp_reduce_dbt_init(&args, team, &task),
                   out, status);

    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, task), out, status);
    UCC_CHECK_GOTO(ucc_event_manager_subscribe(&schedule->super,
                                               UCC_EVENT_SCHEDULE_STARTED, task,
                                               ucc_task_start_handler),
                   out, status);
    rs_task = task;

    /* 2nd step of allreduce: bcast dbt . 2nd task subscribes
     to completion event of reduce task. */
    args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
    args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    UCC_CHECK_GOTO(
        ucc_tl_ucp_bcast_dbt_init(&args, team, &task), out,
        status);

    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, task), out, status);
    UCC_CHECK_GOTO(ucc_event_manager_subscribe(rs_task, UCC_EVENT_COMPLETED,
                                               task, ucc_task_start_handler),
                   out, status);

    schedule->super.post           = ucc_tl_ucp_allreduce_dbt_start;
    schedule->super.progress       = NULL;
    schedule->super.finalize       = ucc_tl_ucp_allreduce_dbt_finalize;
    *task_h                        = &schedule->super;
    return UCC_OK;
out:
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}
