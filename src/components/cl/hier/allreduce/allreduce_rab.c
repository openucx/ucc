/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"

#define MAX_AR_RAB_TASKS 3

static ucc_status_t ucc_cl_hier_allreduce_rab_start(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allreduce_rab_start", 0);
    return ucc_schedule_start(schedule);
}

static ucc_status_t ucc_cl_hier_allreduce_rab_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allreduce_rab_finalize", 0);
    UCC_CL_HIER_PROFILE_REQUEST_FREE(task);
    status = ucc_schedule_finalize(task);
    ucc_free(schedule);
    return status;
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_allreduce_rab_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args,
                         ucc_base_team_t      *team,
                         ucc_coll_task_t     **task)
{
    ucc_cl_hier_team_t     *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_hier_sbgp_type_t    top_sbgp;
    ucc_schedule_t         *schedule;
    ucc_status_t            status;
    ucc_base_coll_args_t    args;
    ucc_coll_task_t        *tasks[MAX_AR_RAB_TASKS];
    int                     n_tasks, i;

    schedule = ucc_malloc(sizeof(*schedule), "hier ar rab schedule");
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    UCC_CL_HIER_PROFILE_REQUEST_NEW(schedule, "cl_hier_allreduce_rab", 0);

    memcpy(&args, coll_args, sizeof(args));
    args.args.root = 0; /* TODO: we can select the rank closest to HCA */
    n_tasks        = 0;
    status         = ucc_schedule_init(schedule, &args, team);
    if (ucc_unlikely(UCC_OK != status)) {
        goto out;
    }

    if (SBGP_EXISTS(cl_team, NODE_LEADERS)) {
        top_sbgp = UCC_HIER_SBGP_NODE_LEADERS;
    } else {
        ucc_assert(SBGP_EXISTS(cl_team, NODE));
        top_sbgp = UCC_HIER_SBGP_NODE;
    }

    if (SBGP_ENABLED(cl_team, NODE)) {
        ucc_assert(n_tasks == 0);
        /* can have only NODE sbgp, both above have been skipped */
        if (top_sbgp == UCC_HIER_SBGP_NODE) {
            args.args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
        } else {
            args.args.coll_type = UCC_COLL_TYPE_REDUCE;
        }
        status = ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]);
        if (ucc_unlikely(UCC_OK != status)) {
            goto out;
        }
        n_tasks++;
        args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if (SBGP_ENABLED(cl_team, NODE_LEADERS)) {
        ucc_assert(top_sbgp == UCC_HIER_SBGP_NODE_LEADERS);
        args.args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
        status = ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS), &args, &tasks[n_tasks]);
        if (ucc_unlikely(UCC_OK != status)) {
            goto out;
        }
        n_tasks++;
    }

    /* For bcast src.buffer should point to origin dst.buffer of allreduce */
    args.args.src.info.buffer = args.args.dst.info.buffer;

    if (SBGP_ENABLED(cl_team, NODE) && top_sbgp != UCC_HIER_SBGP_NODE) {
        args.args.coll_type = UCC_COLL_TYPE_BCAST;
        status = ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]);
        if (ucc_unlikely(UCC_OK != status)) {
            goto out;
        }
        n_tasks++;
    }

    for (i = 0; i < n_tasks; i++) {
        if (i == 0) {
            ucc_event_manager_subscribe(&schedule->super.em, UCC_EVENT_SCHEDULE_STARTED,
                                        tasks[i], ucc_task_start_handler);
        } else {
            ucc_event_manager_subscribe(&tasks[i - 1]->em, UCC_EVENT_COMPLETED, tasks[i],
                                        ucc_task_start_handler);
        }
        ucc_schedule_add_task(schedule, tasks[i]);
    }

    schedule->super.post     = ucc_cl_hier_allreduce_rab_start;
    schedule->super.finalize = ucc_cl_hier_allreduce_rab_finalize;
    *task                    = &schedule->super;
    return UCC_OK;

out:
    //TODO cleanup tasks
    ucc_free(schedule);
    return status;
}
