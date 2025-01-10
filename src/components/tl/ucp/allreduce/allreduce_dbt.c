/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allreduce.h"
#include "../reduce/reduce.h"
#include "../bcast/bcast.h"

ucc_status_t ucc_tl_ucp_allreduce_dbt_start(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t  *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_coll_args_t *args     = &schedule->super.bargs.args;
    ucc_coll_task_t *reduce_task, *bcast_task;

    reduce_task = schedule->tasks[0];
    reduce_task->bargs.args.src.info.buffer = args->src.info.buffer;
    reduce_task->bargs.args.dst.info.buffer = args->dst.info.buffer;
    reduce_task->bargs.args.src.info.count  = args->src.info.count;
    reduce_task->bargs.args.dst.info.count  = args->dst.info.count;

    bcast_task = schedule->tasks[1];
    bcast_task->bargs.args.src.info.buffer = args->dst.info.buffer;
    bcast_task->bargs.args.src.info.count  = args->dst.info.count;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allreduce_dbt_start", 0);
    return ucc_schedule_start(coll_task);
}

ucc_status_t ucc_tl_ucp_allreduce_dbt_finalize(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_status_t    status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(schedule, "ucp_allreduce_dbt_done", 0);
    status = ucc_schedule_finalize(coll_task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t ucc_tl_ucp_allreduce_dbt_init(ucc_base_coll_args_t *coll_args,
                                           ucc_base_team_t *team,
                                           ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t args     = *coll_args;
    ucc_schedule_t      *schedule;
    ucc_coll_task_t     *reduce_task, *bcast_task;
    ucc_status_t         status;

    if (UCC_IS_INPLACE(args.args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                     (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    args.args.root = 0;
    UCC_CHECK_GOTO(ucc_tl_ucp_reduce_dbt_init(&args, team, &reduce_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, reduce_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_event_manager_subscribe(&schedule->super,
                                               UCC_EVENT_SCHEDULE_STARTED,
                                               reduce_task,
                                               ucc_task_start_handler),
                   out, status);

    UCC_CHECK_GOTO(ucc_tl_ucp_bcast_dbt_init(&args, team, &bcast_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, bcast_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_event_manager_subscribe(reduce_task, UCC_EVENT_COMPLETED,
                                               bcast_task,
                                               ucc_task_start_handler),
                   out, status);

    schedule->super.post = ucc_tl_ucp_allreduce_dbt_start;
    schedule->super.progress = NULL;
    schedule->super.finalize =  ucc_tl_ucp_allreduce_dbt_finalize;
    *task_h = &schedule->super;

    return UCC_OK;

out:
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}
