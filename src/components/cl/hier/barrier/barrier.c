/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "barrier.h"
#include "../cl_hier_coll.h"

#define MAX_BARRIER_TASKS 3

static ucc_status_t ucc_cl_hier_barrier_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_barrier_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_barrier_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_barrier_finalize",
                                      0);
    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_barrier_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t  *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_coll_task_t     *tasks[MAX_BARRIER_TASKS] = {NULL};
    ucc_schedule_t      *schedule;
    ucc_status_t         status;
    ucc_base_coll_args_t args;
    int                  n_tasks, i;

    schedule = &ucc_cl_hier_get_schedule(cl_team)->super.super;
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }

    memcpy(&args, coll_args, sizeof(args));
    args.args.root = 0; /* TODO: we can select the rank closest to HCA */
    n_tasks        = 0;
    UCC_CHECK_GOTO(ucc_schedule_init(schedule, &args, team), out, status);

    if (SBGP_ENABLED(cl_team, NODE)) {
        ucc_assert(n_tasks == 0);
        if (cl_team->top_sbgp == UCC_HIER_SBGP_NODE) {
            args.args.coll_type = UCC_COLL_TYPE_BARRIER;
        } else {
            args.args.coll_type = UCC_COLL_TYPE_FANIN;
        }
        UCC_CHECK_GOTO(
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]),
            out, status);
        n_tasks++;
    }

    if (SBGP_ENABLED(cl_team, NODE_LEADERS)) {
        ucc_assert(cl_team->top_sbgp == UCC_HIER_SBGP_NODE_LEADERS);
        args.args.coll_type = UCC_COLL_TYPE_BARRIER;
        UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS), &args,
                                     &tasks[n_tasks]),
                       out, status);
        n_tasks++;
    }


    if (SBGP_ENABLED(cl_team, NODE) &&
        cl_team->top_sbgp != UCC_HIER_SBGP_NODE) {
        args.args.coll_type = UCC_COLL_TYPE_FANOUT;
        UCC_CHECK_GOTO(
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]),
            out, status);
        n_tasks++;
    }

    UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                       &schedule->super, UCC_EVENT_SCHEDULE_STARTED, tasks[0],
                       ucc_task_start_handler),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, tasks[0]), out, status);
    for (i = 1; i < n_tasks; i++) {
        UCC_CHECK_GOTO(
            ucc_event_manager_subscribe(tasks[i - 1], UCC_EVENT_COMPLETED,
                                        tasks[i], ucc_task_start_handler),
            out, status);
        UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, tasks[i]), out, status);
    }

    schedule->super.post     = ucc_cl_hier_barrier_start;
    schedule->super.finalize = ucc_cl_hier_barrier_finalize;
    *task                    = &schedule->super;
    return UCC_OK;

out:
    for (i = 0; i < n_tasks; i++) {
        tasks[i]->finalize(tasks[i]);
    }
    ucc_cl_hier_put_schedule(schedule);
    return status;
}
