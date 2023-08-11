/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "bcast.h"
#include "core/ucc_team.h"
#include "../cl_hier_coll.h"

static ucc_status_t ucc_cl_hier_bcast_2step_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_bcast_2step_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_bcast_2step_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_bcast_2step_finalize",
                                      0);
    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

static ucc_status_t
ucc_cl_hier_bcast_2step_schedule_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status;

    status = ucc_schedule_pipelined_finalize(&schedule->super.super.super);
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

static ucc_status_t
ucc_cl_hier_bcast_2step_frag_setup(ucc_schedule_pipelined_t *schedule_p,
                                   ucc_schedule_t *frag, int frag_num)
{
    ucc_coll_args_t *args    = &schedule_p->super.super.bargs.args;
    size_t           dt_size = ucc_dt_size(args->src.info.datatype);
    int              n_frags = schedule_p->super.n_tasks;
    size_t           frag_count, frag_offset;
    ucc_coll_task_t *task;
    int              i;

    frag_count =
        ucc_buffer_block_count(args->src.info.count, n_frags, frag_num);
    frag_offset =
        ucc_buffer_block_offset(args->src.info.count, n_frags, frag_num);

    for (i = 0; i < frag->n_tasks; i++) {
        task                             = frag->tasks[i];
        task->bargs.args.src.info.count  = frag_count;
        task->bargs.args.src.info.buffer =
            PTR_OFFSET(args->src.info.buffer, frag_offset * dt_size);
    }
    return UCC_OK;
}

static inline ucc_rank_t
find_root_net_rank(ucc_host_id_t root_host_id, ucc_cl_hier_team_t *cl_team)
{
    ucc_sbgp_t *sbgp      = cl_team->sbgps[UCC_HIER_SBGP_NODE_LEADERS].sbgp;
    ucc_team_t *core_team = cl_team->super.super.params.team;
    ucc_rank_t  i, rank;

    for (i = 0; i < sbgp->group_size; i++) {
        rank = ucc_ep_map_eval(sbgp->map, i);
        if (ucc_team_rank_host_id(rank, core_team) == root_host_id) {
            return i;
        }
    }
    return UCC_RANK_INVALID;
}

static inline ucc_rank_t
find_root_node_rank(ucc_rank_t root, ucc_cl_hier_team_t *cl_team)
{
    ucc_sbgp_t *sbgp       = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp;
    ucc_rank_t  i;

    for (i = 0; i < sbgp->group_size; i++) {
        if (ucc_ep_map_eval(sbgp->map, i) == root) {
            return i;
        }
    }
    return UCC_RANK_INVALID;
}

static ucc_status_t
ucc_cl_hier_bcast_2step_init_schedule(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t      *team,
                                      ucc_schedule_t **sched_p, int n_frags)
{
    ucc_cl_hier_team_t *cl_team   = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_team_t         *core_team = team->params.team;
    ucc_coll_task_t    *tasks[2]  = {NULL, NULL};
    ucc_rank_t root               = coll_args->args.root;
    ucc_rank_t rank               = UCC_TL_TEAM_RANK(cl_team);
    int root_on_local_node        = ucc_team_ranks_on_same_node(root, rank,
                                                                core_team);
    ucc_base_coll_args_t args       = *coll_args;
    int                  n_tasks    = 0;
    int                  first_task = 0;
    ucc_schedule_t      *schedule;
    ucc_status_t         status;
    int                  i;

    schedule = &ucc_cl_hier_get_schedule(cl_team)->super.super;
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    status = ucc_schedule_init(schedule, &args, team);
    if (ucc_unlikely(UCC_OK != status)) {
        goto out;
    }

    if (n_frags > 1) {
        args.max_frag_count =
            ucc_buffer_block_count(args.args.src.info.count, n_frags, 0);
        args.mask |= UCC_BASE_CARGS_MAX_FRAG_COUNT;
    }

    ucc_assert(SBGP_ENABLED(cl_team, NODE_LEADERS) ||
               SBGP_ENABLED(cl_team, NODE));
    if (SBGP_ENABLED(cl_team, NODE_LEADERS)) {
        args.args.root = find_root_net_rank(
            ucc_team_rank_host_id(root, core_team), cl_team);
        status = ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS), &args,
                               &tasks[n_tasks]);
        if (ucc_unlikely(UCC_OK != status)) {
            goto out;
        }
        n_tasks++;
        if (root_on_local_node && (root != rank)) {
            first_task = 1;
        }
    }

    if (SBGP_ENABLED(cl_team, NODE)) {
        args.args.root = root_on_local_node
            ? find_root_node_rank(root, cl_team)
            : core_team->topo->node_leader_rank_id;
        status =
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]);
        if (ucc_unlikely(UCC_OK != status)) {
            goto out;
        }
        n_tasks++;
    }

    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, tasks[first_task],
                                          UCC_EVENT_SCHEDULE_STARTED),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, tasks[first_task]),
                   out, status);
    if (n_tasks > 1) {
        if (root == rank) {
            UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super,
                                                  tasks[(first_task + 1) % 2],
                                                  UCC_EVENT_SCHEDULE_STARTED),
                           out, status);
        } else {
            UCC_CHECK_GOTO(ucc_task_subscribe_dep(tasks[first_task],
                                                  tasks[(first_task + 1) % 2],
                                                  UCC_EVENT_COMPLETED),
                            out, status);
        }
        UCC_CHECK_GOTO(ucc_schedule_add_task(schedule,
                                             tasks[(first_task + 1) % 2]),
                       out, status);
    }

    schedule->super.post           = ucc_cl_hier_bcast_2step_start;
    schedule->super.progress       = NULL;
    schedule->super.finalize       = ucc_cl_hier_bcast_2step_finalize;
    *sched_p                       = schedule;
    return UCC_OK;

out:
    for (i = 0; i < n_tasks; i++) {
        tasks[i]->finalize(tasks[i]);
    }
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

static ucc_status_t ucc_cl_hier_bcast_2step_frag_init(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *sp,
    ucc_base_team_t *team,           ucc_schedule_t **frag_p)
{
    int n_frags = sp->super.n_tasks;

    return ucc_cl_hier_bcast_2step_init_schedule(coll_args, team, frag_p,
                                                 n_frags);
}

static ucc_status_t ucc_cl_hier_2step_bcast_start(ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule =
        ucc_derived_of(task, ucc_schedule_pipelined_t);

    cl_debug(task->team->context->lib,
             "posting 2step bcast, buf %p, count %zd, dt %s"
             " pdepth %d, frags_total %d",
             task->bargs.args.src.info.buffer,
             task->bargs.args.src.info.count,
             ucc_datatype_str(task->bargs.args.src.info.datatype),
             schedule->n_frags, schedule->super.n_tasks);

    return ucc_schedule_pipelined_post(task);
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_bcast_2step_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t       *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_lib_config_t *cfg     = &UCC_CL_HIER_TEAM_LIB(cl_team)->cfg;
    ucc_cl_hier_schedule_t *  schedule;
    int                       n_frags, pipeline_depth;
    ucc_status_t              status;

    if (UCC_IS_PERSISTENT(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    ucc_pipeline_nfrags_pdepth(&cfg->bcast_2step_pipeline,
                               coll_args->args.src.info.count *
                               ucc_dt_size(coll_args->args.src.info.datatype),
                               &n_frags, &pipeline_depth);

    if (n_frags == 1) {
        return ucc_cl_hier_bcast_2step_init_schedule(
            coll_args, team, (ucc_schedule_t **)task, n_frags);
    }

    schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_schedule_pipelined_init(
        coll_args, team, ucc_cl_hier_bcast_2step_frag_init,
        ucc_cl_hier_bcast_2step_frag_setup, pipeline_depth, n_frags,
        cfg->bcast_2step_pipeline.order, &schedule->super);

    if (ucc_unlikely(status != UCC_OK)) {
        cl_error(team->context->lib,
                 "failed to init pipelined 2step bcast schedule");
        goto err_pipe_init;
    }

    schedule->super.super.super.post           = ucc_cl_hier_2step_bcast_start;
    schedule->super.super.super.finalize       =
        ucc_cl_hier_bcast_2step_schedule_finalize;
    *task = &schedule->super.super.super;
    return UCC_OK;

err_pipe_init:
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}
