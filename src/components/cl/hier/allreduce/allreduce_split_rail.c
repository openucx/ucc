/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allreduce.h"
#include "../cl_hier_coll.h"
#include "core/ucc_team.h"

static ucc_status_t
ucc_cl_hier_allreduce_split_rail_frag_finalize(ucc_coll_task_t *task)
{
    ucc_status_t            status = UCC_OK;
    ucc_cl_hier_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);

    status = ucc_schedule_finalize(&schedule->super.super.super);
    ucc_free(schedule->allreduce_split_rail.counts);
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

static ucc_status_t
ucc_cl_hier_ar_split_rail_schedule_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status = UCC_OK;

    if (schedule->scratch) {
        ucc_mc_free(schedule->scratch);
    }
    status = ucc_schedule_pipelined_finalize(&schedule->super.super.super);
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

static ucc_status_t ucc_cl_hier_allreduce_split_rail_frag_setup(
    ucc_schedule_pipelined_t *schedule_p, ucc_schedule_t *frag, int frag_num)
{
    ucc_cl_hier_team_t *cl_team =
        ucc_derived_of(schedule_p->super.super.team, ucc_cl_hier_team_t);
    ucc_cl_hier_schedule_t *sched =
        ucc_derived_of(schedule_p, ucc_cl_hier_schedule_t);
    ucc_coll_args_t *args    = &schedule_p->super.super.bargs.args;
    size_t           dt_size = ucc_dt_size(args->dst.info.datatype);
    int              n_frags = schedule_p->super.n_tasks;
    int              inplace = UCC_IS_INPLACE(*args);
    size_t           frag_count, frag_offset, ar_count, ar_offset;
    ucc_rank_t       node_size, node_rank;
    ucc_coll_task_t *task_rs, *task_ar, *task_ag;
    int              i;
    uint64_t *       counts, *displs;

    node_size = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_size;
    node_rank = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_rank;
    frag_count =
        ucc_buffer_block_count(args->dst.info.count, n_frags, frag_num);
    frag_offset =
        ucc_buffer_block_offset(args->dst.info.count, n_frags, frag_num);
    counts = ucc_derived_of(frag, ucc_cl_hier_schedule_t)
                 ->allreduce_split_rail.counts;
    displs = counts + node_size;
    for (i = 0; i < node_size; i++) {
        counts[i] = ucc_buffer_block_count(frag_count, node_size, i);
        displs[i] = ucc_buffer_block_offset(frag_count, node_size, i);
    }

    ar_count  = counts[node_rank];
    ar_offset = displs[node_rank];

    task_rs = frag->tasks[0];
    task_ar = frag->tasks[1];
    task_ag = frag->tasks[2];

    ucc_assert(task_rs->bargs.args.dst.info_v.counts == counts);

    if (inplace) {
        task_rs->bargs.args.src.info.buffer =
            PTR_OFFSET(args->dst.info.buffer, frag_offset * dt_size);
        task_rs->bargs.args.dst.info_v.buffer = PTR_OFFSET(
            sched->scratch->addr, (frag_offset + ar_offset) * dt_size);
    } else {
        task_rs->bargs.args.src.info.buffer =
            PTR_OFFSET(args->src.info.buffer, frag_offset * dt_size);
        task_rs->bargs.args.dst.info_v.buffer = PTR_OFFSET(
            args->dst.info.buffer, (frag_offset + ar_offset) * dt_size);
        task_rs->bargs.args.src.info.count = frag_count;
    }

    task_ar->bargs.args.src.info.count = ar_count;
    task_ar->bargs.args.dst.info.count = ar_count;
    if (!inplace) {
        task_ar->bargs.args.dst.info.buffer =
            task_rs->bargs.args.dst.info_v.buffer;
    } else {
        task_ar->bargs.args.src.info.buffer =
            task_rs->bargs.args.dst.info_v.buffer;
        task_ar->bargs.args.dst.info.buffer = PTR_OFFSET(
            args->dst.info.buffer, (frag_offset + ar_offset) * dt_size);
    }

    ucc_assert(UCC_IS_INPLACE(task_ag->bargs.args));
    task_ag->bargs.args.dst.info_v.buffer = PTR_OFFSET(
        args->dst.info.buffer, frag_offset * dt_size); //only dst since inplace
    task_ag->bargs.args.src.info.count = frag_count;
    ucc_assert(task_ag->bargs.args.dst.info_v.counts == counts);
    ucc_assert(task_ag->bargs.args.dst.info_v.displacements == displs);
    return UCC_OK;
}

static ucc_status_t ucc_cl_hier_allreduce_split_rail_frag_init(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *sp,
    ucc_base_team_t *team, ucc_schedule_t **frag_p)
{
    ucc_cl_hier_team_t *    cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_schedule_t *sched = ucc_derived_of(sp, ucc_cl_hier_schedule_t);
    size_t           dt_size = ucc_dt_size(coll_args->args.dst.info.datatype);
    ucc_status_t     status  = UCC_OK;
    int              inplace = UCC_IS_INPLACE(coll_args->args);
    ucc_coll_task_t *task_rs, *task_ag, *task_ar;
    ucc_base_coll_args_t    rs_args, ar_args, ag_args;
    ucc_cl_hier_schedule_t *cl_schedule;
    ucc_schedule_t *        schedule;
    size_t                  total_count;
    ucc_rank_t              node_size, node_rank;
    int       i;
    uint64_t *counts, *displs;

    cl_schedule = ucc_cl_hier_get_schedule(cl_team);

    if (ucc_unlikely(!cl_schedule)) {
        return UCC_ERR_NO_MEMORY;
    }

    schedule = &cl_schedule->super.super;
    status   = ucc_schedule_init(schedule, coll_args, team);
    if (UCC_OK != status) {
        return status;
    }

    node_size   = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_size;
    node_rank   = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_rank;
    total_count = coll_args->args.dst.info.count;

    cl_schedule->allreduce_split_rail.counts =
        ucc_malloc(node_size * 2 * sizeof(uint64_t), "counts");
    if (ucc_unlikely(!cl_schedule->allreduce_split_rail.counts)) {
        cl_error(team->context->lib,
                 "failed to allocate %zd bytes for counts array",
                 node_size * 2 * sizeof(uint64_t));
        goto err_rs;
    }
    counts = cl_schedule->allreduce_split_rail.counts;
    displs = counts + node_size;
    for (i = 0; i < node_size; i++) {
        counts[i] = ucc_buffer_block_count(total_count, node_size, i);
        displs[i] = ucc_buffer_block_offset(total_count, node_size, i);
    }
    memcpy(&rs_args, coll_args, sizeof(rs_args));
    memcpy(&ar_args, coll_args, sizeof(ar_args));
    memcpy(&ag_args, coll_args, sizeof(ag_args));

    rs_args.args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
    rs_args.args.flags &= (~UCC_COLL_ARGS_FLAG_IN_PLACE);
    rs_args.args.flags |= (UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                           UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT);

    /* REDUCE-SCATTER */
    rs_args.args.coll_type           = UCC_COLL_TYPE_REDUCE_SCATTERV;
    rs_args.args.dst.info_v.counts   = counts;
    rs_args.args.dst.info_v.mem_type = coll_args->args.dst.info.mem_type;
    rs_args.args.dst.info_v.datatype = coll_args->args.dst.info.datatype;
    if (inplace) {
        rs_args.args.src.info.buffer   = coll_args->args.dst.info.buffer;
        rs_args.args.src.info.datatype = coll_args->args.dst.info.datatype;
        rs_args.args.dst.info_v.buffer =
            PTR_OFFSET(sched->scratch->addr, displs[node_rank] * dt_size);
    } else {
        rs_args.args.dst.info_v.buffer = PTR_OFFSET(
            coll_args->args.dst.info.buffer, displs[node_rank] * dt_size);
        rs_args.args.src.info.count = coll_args->args.dst.info.count;
    }

    status = ucc_coll_init(SCORE_MAP(cl_team, NODE), &rs_args, &task_rs);
    if (ucc_unlikely(UCC_OK != status)) {
        cl_error(team->context->lib, "failed to init rs task");
        goto err_rs;
    }

    /* ALLREDUCE */
    ar_args.args.coll_type      = UCC_COLL_TYPE_ALLREDUCE;
    ar_args.args.src.info.count = counts[node_rank];
    if (!inplace) {
        ar_args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        ar_args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
        ar_args.args.dst.info.count = counts[node_rank];
    } else {
        ar_args.args.flags &= (~UCC_COLL_ARGS_FLAG_IN_PLACE);
        ar_args.args.src.info.buffer = rs_args.args.dst.info.buffer;
        ar_args.args.dst.info.buffer = PTR_OFFSET(
            coll_args->args.dst.info.buffer, displs[node_rank] * dt_size);
    }

    status = ucc_coll_init(SCORE_MAP(cl_team, NET), &ar_args, &task_ar);
    if (ucc_unlikely(UCC_OK != status)) {
        cl_error(team->context->lib, "failed to init ar task");
        goto err_ar;
    }

    /* ALLGATHER */
    ag_args.args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
    ag_args.args.flags |= (UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                           UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT);
    ag_args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    ag_args.args.coll_type                = UCC_COLL_TYPE_ALLGATHERV;
    ag_args.args.dst.info_v.buffer        = coll_args->args.dst.info.buffer;
    ag_args.args.dst.info_v.mem_type      = coll_args->args.dst.info.mem_type;
    ag_args.args.dst.info_v.datatype      = coll_args->args.dst.info.datatype;
    ag_args.args.dst.info_v.counts        = counts;
    ag_args.args.dst.info_v.displacements = displs;

    status = ucc_coll_init(SCORE_MAP(cl_team, NODE), &ag_args, &task_ag);
    if (ucc_unlikely(UCC_OK != status)) {
        cl_error(team->context->lib, "failed to init ag task");
        goto err_ag;
    }

    task_rs->n_deps = 1;
    ucc_schedule_add_task(schedule, task_rs);
    ucc_event_manager_subscribe(&schedule->super.em, UCC_EVENT_SCHEDULE_STARTED,
                                task_rs, ucc_dependency_handler);

    task_ar->n_deps = 1;
    ucc_schedule_add_task(schedule, task_ar);
    ucc_event_manager_subscribe(&task_rs->em, UCC_EVENT_COMPLETED, task_ar,
                                ucc_dependency_handler);

    task_ag->n_deps = 1;
    ucc_schedule_add_task(schedule, task_ag);
    ucc_event_manager_subscribe(&task_ar->em, UCC_EVENT_COMPLETED, task_ag,
                                ucc_dependency_handler);

    schedule->super.post     = ucc_schedule_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_cl_hier_allreduce_split_rail_frag_finalize;

    *frag_p = schedule;
    return status;

err_ag:
    if (task_ar) {
        ucc_collective_finalize(&task_ar->super);
    }
err_ar:
    if (task_rs) {
        ucc_collective_finalize(&task_rs->super);
    }
err_rs:
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

static inline void get_n_frags(ucc_base_coll_args_t *coll_args,
                               ucc_cl_hier_team_t *team, int *n_frags,
                               int *pipeline_depth)
{
    ucc_cl_hier_lib_config_t *cfg     = &UCC_CL_HIER_TEAM_LIB(team)->cfg;
    size_t                    msgsize = coll_args->args.dst.info.count *
                     ucc_dt_size(coll_args->args.dst.info.datatype);
    int min_num_frags;

    *n_frags = 1;
    if (msgsize > cfg->allreduce_split_rail_frag_thresh) {
        min_num_frags = msgsize / cfg->allreduce_split_rail_frag_size;
        *n_frags = ucc_max(min_num_frags, cfg->allreduce_split_rail_n_frags);
    }
    *pipeline_depth =
        ucc_min(*n_frags, cfg->allreduce_split_rail_pipeline_depth);
}

static ucc_status_t
ucc_cl_hier_split_rail_allreduce_start(ucc_coll_task_t *task)
{
    ucc_schedule_pipelined_t *schedule =
        ucc_derived_of(task, ucc_schedule_pipelined_t);

    cl_info(task->team->context->lib,
            "posting split_rail ar, sbuf %p, rbuf %p, count %zd, dt %s, op %s, "
            "inplace %d, pdepth %d, frags_total %d",
            task->bargs.args.src.info.buffer, task->bargs.args.dst.info.buffer,
            task->bargs.args.dst.info.count,
            ucc_datatype_str(task->bargs.args.src.info.datatype),
            ucc_reduction_op_str(task->bargs.args.op),
            UCC_IS_INPLACE(task->bargs.args), schedule->n_frags,
            schedule->super.n_tasks);

    return ucc_schedule_pipelined_post(task);
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_allreduce_split_rail_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_lib_config_t *cfg   = &UCC_CL_HIER_TEAM_LIB(cl_team)->cfg;
    size_t                    count = coll_args->args.dst.info.count;
    size_t data_size = count * ucc_dt_size(coll_args->args.dst.info.datatype);
    ucc_cl_hier_schedule_t *schedule;
    int                 n_frags, pipeline_depth;
    ucc_status_t status;

    if (coll_args->args.op == UCC_OP_AVG) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!SBGP_ENABLED(cl_team, NODE) || !SBGP_ENABLED(cl_team, NET)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!ucc_topo_isoppn(team->params.team->topo)) {
        cl_debug(team->context->lib, "split_rail algorithm does not support "
                                     "teams with non-uniform ppn across nodes");
        return UCC_ERR_NOT_SUPPORTED;
    }

    schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }

    if (UCC_IS_INPLACE(coll_args->args)) {
        status = ucc_mc_alloc(&schedule->scratch, data_size,
                              coll_args->args.dst.info.mem_type);
        if (ucc_unlikely(UCC_OK != status)) {
            cl_error(team->context->lib,
                     "failed to allocate %zd bytes for inplace scratch",
                     data_size);
            goto err_scratch;
        }
    }

    get_n_frags(coll_args, cl_team, &n_frags, &pipeline_depth);

    status = ucc_schedule_pipelined_init(
        coll_args, team, ucc_cl_hier_allreduce_split_rail_frag_init,
        ucc_cl_hier_allreduce_split_rail_frag_setup, pipeline_depth, n_frags,
        cfg->allreduce_split_rail_seq, &schedule->super);

    if (ucc_unlikely(status != UCC_OK)) {
        cl_error(team->context->lib,
                 "failed to init pipelined split_rail ar schedule");
        goto err_pipe_init;
    }

    schedule->super.super.super.post = ucc_cl_hier_split_rail_allreduce_start;
    schedule->super.super.super.triggered_post = ucc_triggered_post;
    schedule->super.super.super.finalize =
        ucc_cl_hier_ar_split_rail_schedule_finalize;
    *task = &schedule->super.super.super;
    return UCC_OK;

err_pipe_init:
    if (schedule->scratch) {
        ucc_mc_free(schedule->scratch);
    }
err_scratch:
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}
