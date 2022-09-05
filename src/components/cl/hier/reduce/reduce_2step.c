/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "reduce.h"
#include "core/ucc_team.h"
#include "../cl_hier_coll.h"

#define MAX_AR_2STEP_TASKS 3

static ucc_status_t ucc_cl_hier_reduce_2step_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_reduce_2step_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_reduce_2step_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule = ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t    status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_reduce_2step_finalize",
                                      0);
    status = ucc_schedule_finalize(task);
    if (schedule->scratch) {
        ucc_mc_free(schedule->scratch);
    }
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

static ucc_status_t ucc_cl_hier_ar_2step_schedule_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status;

    status = ucc_schedule_pipelined_finalize(&schedule->super.super.super);
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

static ucc_status_t
ucc_cl_hier_reduce_2step_frag_setup(ucc_schedule_pipelined_t *schedule_p,
                                     ucc_schedule_t *frag, int frag_num)
{
    ucc_cl_hier_team_t *cl_team =
        ucc_derived_of(schedule_p->super.super.team, ucc_cl_hier_team_t);
    ucc_coll_args_t *args    = &schedule_p->super.super.bargs.args;
    size_t           dt_size = ucc_dt_size(args->src.info.datatype);
    int              n_frags = schedule_p->super.n_tasks;
    ucc_cl_hier_schedule_t * cl_schedule = ucc_derived_of(frag, ucc_cl_hier_schedule_t);
    void             *scratch = cl_schedule->scratch ? cl_schedule->scratch->addr : NULL;
    int root                        = args->root;
    int rank                        = UCC_TL_TEAM_RANK(cl_team);
    size_t    count = (rank == root) ? args->dst.info.count :
        args->src.info.count;

    size_t           frag_count, frag_offset;
    ucc_coll_task_t *task;
    int              i;

    frag_count =
        ucc_buffer_block_count(count, n_frags, frag_num);
    frag_offset =
        ucc_buffer_block_offset(count, n_frags, frag_num);

    for (i = 0; i < frag->n_tasks; i++) {
        task                             = frag->tasks[i];
        task->bargs.args.src.info.count  = frag_count;
        task->bargs.args.dst.info.count  = frag_count;
        if (task->bargs.args.src.info.buffer != scratch) {
            task->bargs.args.src.info.buffer =
                PTR_OFFSET(args->src.info.buffer, frag_offset * dt_size);
        }
        if (task->bargs.args.dst.info.buffer != scratch) {
            task->bargs.args.dst.info.buffer =
                PTR_OFFSET(args->dst.info.buffer, frag_offset * dt_size);
        }
    }
    return UCC_OK;
}

static inline ucc_rank_t
find_root_net_rank(ucc_host_id_t root_host_id, ucc_cl_hier_team_t *cl_team)
{
    ucc_rank_t  net_rank  = UCC_RANK_INVALID;
    ucc_sbgp_t *sbgp      = cl_team->sbgps[UCC_HIER_SBGP_NODE_LEADERS].sbgp;
    ucc_team_t *core_team = cl_team->super.super.params.team;
    ucc_rank_t  i, rank;

    for (i = 0; i < sbgp->group_size; i++) {
        rank = ucc_ep_map_eval(sbgp->map, i);
        if (ucc_team_rank_host_id(rank, core_team) == root_host_id) {
            net_rank = i;
            break;
        }
    }
    return net_rank;
}

static inline ucc_rank_t
find_root_node_rank(ucc_rank_t root, ucc_cl_hier_team_t *cl_team)
{
    ucc_rank_t  node_rank  = UCC_RANK_INVALID;
    ucc_sbgp_t *sbgp       = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp;
    ucc_rank_t  i;

    for (i = 0; i < sbgp->group_size; i++) {
        if (ucc_ep_map_eval(sbgp->map, i) == root) {
            node_rank = i;
            break;
        }
    }
    return node_rank;
}

static ucc_status_t
ucc_cl_hier_reduce_2step_init_schedule(ucc_base_coll_args_t *coll_args,
                                        ucc_base_team_t *     team,
                                        ucc_schedule_t **sched_p, int n_frags)
{
    ucc_cl_hier_team_t *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_team_t         *core_team = team->params.team;
    ucc_coll_task_t    *tasks[2] = {NULL, NULL};
    int root                        = coll_args->args.root;
    int rank                        = UCC_TL_TEAM_RANK(cl_team);
    int root_on_local_node          = ucc_team_ranks_on_same_node(root, rank, core_team);
    ucc_base_coll_args_t args = *coll_args;
    size_t    count = (rank == root) ? args.args.dst.info.count :
        args.args.src.info.count;
    ucc_cl_hier_schedule_t      *cl_schedule;
    ucc_schedule_t      *schedule;
    ucc_status_t         status;

    int                  n_tasks, i, first_task;

    n_tasks        = 0;
    first_task     = 0;

    if (root != rank) {
        args.args.dst.info.count    = args.args.src.info.count;
        args.args.dst.info.mem_type = args.args.src.info.mem_type;
        args.args.dst.info.datatype = args.args.src.info.datatype;
        args.args.mask &= (~UCC_COLL_ARGS_FLAG_IN_PLACE);
    }

    cl_schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!cl_schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    schedule = &cl_schedule->super.super;
    status         = ucc_schedule_init(schedule, &args, team);
    if (ucc_unlikely(UCC_OK != status)) {
        goto out;
    }

    args.max_frag_count = ucc_buffer_block_count(count, n_frags, 0);

    if (n_frags > 1) {
        args.mask |= UCC_BASE_CARGS_MAX_FRAG_COUNT;
    }


    if (SBGP_ENABLED(cl_team, NODE)) {
        args.args.root = root_on_local_node
            ? find_root_node_rank(root, cl_team) : 0;

        if (root != rank && SBGP_ENABLED(cl_team, NODE_LEADERS)) {
            status = ucc_mc_alloc(&cl_schedule->scratch, args.max_frag_count,
                                  args.args.src.info.mem_type);
            if (ucc_unlikely(UCC_OK != status)) {
                goto out;
            }
            args.args.dst.info.buffer = cl_schedule->scratch->addr;
            if (root_on_local_node) {
                first_task = 1;
                args.args.src.info.buffer = cl_schedule->scratch->addr;
            }
        }
        status =
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]);
        if (ucc_unlikely(UCC_OK != status)) {
            goto out;
        }
        n_tasks++;
    }

    if (SBGP_ENABLED(cl_team, NODE_LEADERS)) {
        if (n_tasks == 1) {
            if (root != rank) {
                /* args.args.dst.info.buffer = cl_schedule->scratch->addr; */
                args.args.src.info.buffer = root_on_local_node ?
                    coll_args->args.src.info.buffer : cl_schedule->scratch->addr;
            } else {
                args.args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
                args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
            }
        }
        args.args.root = find_root_net_rank(ucc_team_rank_host_id(root, core_team), cl_team);
        status =
            ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS), &args, &tasks[n_tasks]);
        if (ucc_unlikely(UCC_OK != status)) {
            goto out;
        }
        n_tasks++;
    }

    ucc_task_subscribe_dep(&schedule->super, tasks[first_task], UCC_EVENT_SCHEDULE_STARTED);
    ucc_schedule_add_task(schedule, tasks[first_task]);

    if (n_tasks > 1) {
        ucc_task_subscribe_dep(tasks[first_task],
                               tasks[(first_task + 1) % 2], UCC_EVENT_COMPLETED);
        ucc_schedule_add_task(schedule, tasks[(first_task + 1) % 2]);
    }

    schedule->super.post           = ucc_cl_hier_reduce_2step_start;
    schedule->super.progress       = NULL;
    schedule->super.finalize       = ucc_cl_hier_reduce_2step_finalize;
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

static ucc_status_t ucc_cl_hier_reduce_2step_frag_init(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *sp,
    ucc_base_team_t *team, ucc_schedule_t **frag_p)
{
    int          n_frags = sp->super.n_tasks;
    ucc_status_t status;

    status = ucc_cl_hier_reduce_2step_init_schedule(coll_args, team, frag_p,
                                                     n_frags);
    return status;
}

static ucc_status_t ucc_cl_hier_2step_reduce_start(ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule =
        ucc_derived_of(task, ucc_schedule_pipelined_t);

    cl_debug(task->team->context->lib,
             "posting reduce_2step, count %zd, dt %s"
             " pdepth %d, frags_total %d",
             task->bargs.args.src.info.count,
             ucc_datatype_str(task->bargs.args.src.info.datatype),
             schedule->n_frags, schedule->super.n_tasks);

    return ucc_schedule_pipelined_post(task);
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_reduce_2step_init,
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
    ucc_pipeline_nfrags_pdepth(&cfg->reduce_2step_pipeline,
                               coll_args->args.src.info.count *
                               ucc_dt_size(coll_args->args.src.info.datatype),
                               &n_frags, &pipeline_depth);

    if (n_frags == 1) {
        return ucc_cl_hier_reduce_2step_init_schedule(
            coll_args, team, (ucc_schedule_t **)task, n_frags);
    }

    schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_schedule_pipelined_init(
        coll_args, team, ucc_cl_hier_reduce_2step_frag_init,
        ucc_cl_hier_reduce_2step_frag_setup, pipeline_depth, n_frags,
        cfg->reduce_2step_pipeline.order, &schedule->super);

    if (ucc_unlikely(status != UCC_OK)) {
        cl_error(team->context->lib,
                 "failed to init pipelined 2step ar schedule");
        goto err_pipe_init;
    }

    schedule->super.super.super.post = ucc_cl_hier_2step_reduce_start;
    schedule->super.super.super.triggered_post = ucc_triggered_post;
    schedule->super.super.super.finalize = ucc_cl_hier_ar_2step_schedule_finalize;
    *task                                = &schedule->super.super.super;
    return UCC_OK;

err_pipe_init:
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}
