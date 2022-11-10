/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "../cl_hier_coll.h"

#define MAX_AR_RAB_TASKS 3

static ucc_status_t ucc_cl_hier_allreduce_rab_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allreduce_rab_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_allreduce_rab_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allreduce_rab_finalize",
                                      0);
    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

static ucc_status_t ucc_cl_hier_ar_rab_schedule_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status;

    status = ucc_schedule_pipelined_finalize(&schedule->super.super.super);
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

static ucc_status_t
ucc_cl_hier_allreduce_rab_frag_setup(ucc_schedule_pipelined_t *schedule_p,
                                     ucc_schedule_t *frag, int frag_num)
{
    ucc_cl_hier_team_t *cl_team =
        ucc_derived_of(schedule_p->super.super.team, ucc_cl_hier_team_t);
    ucc_coll_args_t *args    = &schedule_p->super.super.bargs.args;
    size_t           dt_size = ucc_dt_size(args->dst.info.datatype);
    int              n_frags = schedule_p->super.n_tasks;
    int              inplace = UCC_IS_INPLACE(*args);
    size_t           frag_count, frag_offset;
    ucc_coll_task_t *task;
    int              i;

    frag_count =
        ucc_buffer_block_count(args->dst.info.count, n_frags, frag_num);
    frag_offset =
        ucc_buffer_block_offset(args->dst.info.count, n_frags, frag_num);

    for (i = 0; i < frag->n_tasks; i++) {
        task                             = frag->tasks[i];
        task->bargs.args.src.info.count  = frag_count;
        task->bargs.args.dst.info.count  = frag_count;
        task->bargs.args.dst.info.buffer =
            PTR_OFFSET(args->dst.info.buffer, frag_offset * dt_size);
        if ((task->bargs.args.coll_type == UCC_COLL_TYPE_BCAST) ||
            (task->bargs.args.coll_type == UCC_COLL_TYPE_REDUCE && inplace &&
             (SBGP_RANK(cl_team, NODE) != args->root))) {
            task->bargs.args.src.info.buffer =
                PTR_OFFSET(args->dst.info.buffer, frag_offset * dt_size);
        } else {
            task->bargs.args.src.info.buffer =
                PTR_OFFSET(args->src.info.buffer, frag_offset * dt_size);
        }
    }
    return UCC_OK;
}

static ucc_status_t
ucc_cl_hier_allreduce_rab_init_schedule(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *     team,
                                        ucc_schedule_t **sched_p, int n_frags)
{
    ucc_cl_hier_team_t  *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_coll_task_t     *tasks[MAX_AR_RAB_TASKS] = {NULL};
    ucc_schedule_t      *schedule;
    ucc_status_t         status;
    ucc_base_coll_args_t args;
    int                  n_tasks, i;

    schedule = &ucc_cl_hier_get_schedule(cl_team)->super.super;
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }

    args           = *coll_args;
    args.args.root = 0; /* TODO: we can select the rank closest to HCA */
    n_tasks        = 0;
    UCC_CHECK_GOTO(ucc_schedule_init(schedule, &args, team), out, status);

    if (n_frags > 1) {
        args.max_frag_count =
            ucc_buffer_block_count(args.args.dst.info.count, n_frags, 0);
        args.mask |= UCC_BASE_CARGS_MAX_FRAG_COUNT;
    }
    if (SBGP_ENABLED(cl_team, NODE)) {
        ucc_assert(n_tasks == 0);
        if (cl_team->top_sbgp == UCC_HIER_SBGP_NODE) {
            args.args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
        } else {
            args.args.coll_type = UCC_COLL_TYPE_REDUCE;
            if (UCC_IS_INPLACE(args.args) &&
                (SBGP_RANK(cl_team, NODE) != args.args.root)) {
                args.args.src.info = args.args.dst.info;
            }
        }
        UCC_CHECK_GOTO(
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]),
            out, status);
        n_tasks++;
        args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    if (SBGP_ENABLED(cl_team, NODE_LEADERS)) {
        ucc_assert(cl_team->top_sbgp == UCC_HIER_SBGP_NODE_LEADERS);
        args.args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
        UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS), &args,
                                     &tasks[n_tasks]),
                       out, status);
        n_tasks++;
    }


    if (SBGP_ENABLED(cl_team, NODE) &&
        cl_team->top_sbgp != UCC_HIER_SBGP_NODE) {
        /* For bcast src should point to origin dst of allreduce */
        args.args.src.info  = args.args.dst.info;
        args.args.coll_type = UCC_COLL_TYPE_BCAST;
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

    schedule->super.post           = ucc_cl_hier_allreduce_rab_start;
    schedule->super.progress       = NULL;
    schedule->super.finalize       = ucc_cl_hier_allreduce_rab_finalize;
    schedule->super.triggered_post = ucc_triggered_post;
    *sched_p                       = schedule;
    return UCC_OK;

out:
    for (i = 0; i < n_tasks; i++) {
        tasks[i]->finalize(tasks[i]);
    }
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

static ucc_status_t ucc_cl_hier_allreduce_rab_frag_init(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *sp,
    ucc_base_team_t *team, ucc_schedule_t **frag_p)
{
    int          n_frags = sp->super.n_tasks;
    ucc_status_t status;

    status = ucc_cl_hier_allreduce_rab_init_schedule(coll_args, team, frag_p,
                                                     n_frags);
    return status;
}

static ucc_status_t ucc_cl_hier_rab_allreduce_start(ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule =
        ucc_derived_of(task, ucc_schedule_pipelined_t);

    cl_debug(task->team->context->lib,
             "posting rab ar, sbuf %p, rbuf %p, count %zd, dt %s, op %s, "
             "inplace %d, pdepth %d, frags_total %d",
             task->bargs.args.src.info.buffer, task->bargs.args.dst.info.buffer,
             task->bargs.args.dst.info.count,
             ucc_datatype_str(task->bargs.args.src.info.datatype),
             ucc_reduction_op_str(task->bargs.args.op),
             UCC_IS_INPLACE(task->bargs.args), schedule->n_frags,
             schedule->super.n_tasks);

    return ucc_schedule_pipelined_post(task);
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_allreduce_rab_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t *cl_team   = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_lib_config_t *cfg = &UCC_CL_HIER_TEAM_LIB(cl_team)->cfg;
    ucc_cl_hier_schedule_t *  schedule;
    int                       n_frags, pipeline_depth;
    ucc_status_t              status;

    if (coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    ucc_pipeline_nfrags_pdepth(&cfg->allreduce_rab_pipeline,
                               coll_args->args.dst.info.count *
                               ucc_dt_size(coll_args->args.dst.info.datatype),
                               &n_frags, &pipeline_depth);

    if (n_frags == 1) {
        return ucc_cl_hier_allreduce_rab_init_schedule(
            coll_args, team, (ucc_schedule_t **)task, n_frags);
    }

    schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_schedule_pipelined_init(
        coll_args, team, ucc_cl_hier_allreduce_rab_frag_init,
        ucc_cl_hier_allreduce_rab_frag_setup, pipeline_depth, n_frags,
        cfg->allreduce_rab_pipeline.order, &schedule->super);

    if (ucc_unlikely(status != UCC_OK)) {
        cl_error(team->context->lib,
                 "failed to init pipelined rab ar schedule");
        goto err_pipe_init;
    }

    schedule->super.super.super.post = ucc_cl_hier_rab_allreduce_start;
    schedule->super.super.super.triggered_post = ucc_triggered_post;
    schedule->super.super.super.finalize = ucc_cl_hier_ar_rab_schedule_finalize;
    *task                                = &schedule->super.super.super;
    return UCC_OK;

err_pipe_init:
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}
