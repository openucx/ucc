/**
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv.h"
#include "../cl_hier_coll.h"
#include "core/ucc_team.h"

#define MAX_ALLGATHERV_TASKS 3

ucc_base_coll_alg_info_t
    ucc_cl_hier_allgatherv_algs[UCC_CL_HIER_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLGATHERV_ALG_NODE_SPLIT] =
            {.id   = UCC_CL_HIER_ALLGATHERV_ALG_NODE_SPLIT,
             .name = "node_split",
             .desc = "splitting allgatherv into three consecutive calls, first "
                     "a gatherv"
                     " inside the node, then an allgatherv between the leaders "
                     "and then a bcast."},
        [UCC_CL_HIER_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

static ucc_status_t ucc_cl_hier_allgatherv_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_allgatherv_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *cl_schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_finalize", 0);

    ucc_assert(cl_schedule->super.super.n_tasks <= 3);

    if (cl_schedule->scratch) {
        ucc_mc_free(cl_schedule->scratch);
    }

    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(&cl_schedule->super.super);
    return status;
}

// Question: Do i need this function ? 
ucc_status_t ucc_cl_hier_allgatherv_triggered_post_setup(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status  = UCC_OK;
    int          n_tasks = schedule->super.super.n_tasks;
    int          i       = 0;

    for (i = 0; i < n_tasks; ++i) {
        ucc_coll_task_t *sub_task = schedule->super.super.tasks[i];
        if (sub_task->triggered_post_setup != NULL) {
            sub_task->ee = task->ee;
            sub_task->triggered_post_setup(sub_task);
        }
    }
    return status;
}

static ucc_status_t ucc_cl_hier_allgatherv_node_split_init_schedule(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_schedule_t **sched_p, int n_frags)
{
    ucc_cl_hier_team_t     *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_schedule_t *cl_schedule;
    ucc_schedule_t         *schedule;
    ucc_status_t            status;
    ucc_base_coll_args_t    args;
    ucc_coll_task_t        *tasks[MAX_ALLGATHERV_TASKS] = {NULL};
    int                     n_tasks                     = 0;
    void                    *gv_rc, *gv_displ;  // Gatherv args buffers
    void                    *agv_rc, *agv_displ;    // Allgatherv args buffers
    int                     i, c64, d64, rank, nrank, host_id;
    ucc_rank_t              full_size, node_size, leaders_size;
    size_t                  elem_size;
    ucc_rank_t              node_root = 0;

    rank = cl_team->sbgps[UCC_HIER_SBGP_FULL].sbgp->group_rank;
    nrank = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_rank;
    host_id = ucc_team_rank_host_id(rank, coll_args->team);

    int is_root = nrank == node_root;

    c64 = UCC_COLL_ARGS_COUNT64(&coll_args->args);
    d64 = UCC_COLL_ARGS_DISPL64(&coll_args->args);

    if (c64 ^ d64) {
        cl_debug(team->context->lib,
                 "mixed 64 bit count/displ mode is not supported\n");
        return UCC_ERR_NOT_SUPPORTED;
    }

    cl_schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!cl_schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    schedule = &cl_schedule->super.super;
    memcpy(&args, coll_args, sizeof(args)); 

    UCC_CHECK_GOTO(ucc_schedule_init(schedule, &args, team), out, status);

    // Question: What is this ? 
    if (n_frags > 1) {
        args.max_frag_count =
            ucc_buffer_block_count(args.args.src.info.count, n_frags, 0);
        args.mask |= UCC_BASE_CARGS_MAX_FRAG_COUNT;
    }

    full_size    = cl_team->sbgps[UCC_HIER_SBGP_FULL].sbgp->group_size;
    node_size    = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_size;
    leaders_size = cl_team->sbgps[UCC_HIER_SBGP_NODE_LEADERS].sbgp->group_size;
    elem_size    = c64 ? 8 : 4;

    // if (!cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->preserves_order){
    //     printf("Reordering node ranking is not supported");
    //     return UCC_ERR_NOT_SUPPORTED;
    // }

    // Init buffers for collectives arguments
    size_t scratch_size = elem_size * (node_size * 2 + leaders_size * 2);
    status = ucc_mc_alloc(&cl_schedule->scratch, scratch_size, UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(UCC_OK != status)) {
        cl_error(team->context->lib,
                 "failed to allocate %zd bytes for full counts",
                 scratch_size);
        goto out;
    }
    gv_rc    = cl_schedule->scratch->addr; /* +node_size */
    gv_displ = PTR_OFFSET(gv_rc, node_size*elem_size); /* +node_size*/
    agv_rc   = PTR_OFFSET(gv_displ, node_size*elem_size); /* +leaders_size*/
    agv_displ   = PTR_OFFSET(agv_rc, leaders_size*elem_size); /* +leaders_size*/

    // Gatherv in the node
    // src.info.buffer  -> dst.info_v.buffer
    if (node_size > 1){

        do { // This section need to be rewritten to support both uint32_t and uint64_t once the logic is approved
            int _i;
            uint32_t _scount, _displ;
            
            _displ = 0;
            
            /* For every rank in the node, add his count as is and the displacement 
            *    to be the running sum of the counts 
            */
            for (_i = 0; is_root && _i < node_size; _i++) {
                ucc_rank_t r = ucc_ep_map_eval((*cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp).map, _i);
                _scount = ((uint32_t *)coll_args->args.dst.info_v.counts)[r];
                ((uint32_t *)gv_rc)[_i] = _scount;
                ((uint32_t *)gv_displ)[_i] = _displ;
                
                _displ += _scount;
            }
        } while (0);

        args.args.coll_type = UCC_COLL_TYPE_GATHERV;
        args.args.root      = node_root;
        args.args.dst.info_v.counts = (ucc_count_t *)gv_rc;
        args.args.dst.info_v.displacements = (ucc_aint_t *)gv_displ;

        UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]), out, status);
        UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                        &schedule->super, UCC_EVENT_SCHEDULE_STARTED,
                        tasks[n_tasks], ucc_task_start_handler),
                    out, status);
        UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, tasks[n_tasks]), out,
                    status);

        // The result of this collective is in dst.info_v.buffer, this should be the source buffer of the next collective
        args.args.src.info.buffer = args.args.dst.info_v.buffer; 
        n_tasks++;
    }

    // Allgatherv in the full net
    // src.info.buffer  -> dst.info_v.buffer
    if (is_root) {

        do { // This section need to be rewritten to support both uint32_t and uint64_t once the logic is approved
            int _i;
            for (_i = 0; _i < leaders_size; _i++) {
                ((uint32_t *) agv_rc)[_i] = 0;
            }

            int _hid, _count, _displ;
            /* For every rank in the communicator, 
            *   - find his host id (which is also the index of the leader -- Sergey ?)
            *   - Add his count to the leader count
            */
            for (_i = 0; _i < full_size; _i++){
                _hid = ucc_team_rank_host_id(_i, coll_args->team);
                _count = ((uint32_t *)coll_args->args.dst.info_v.counts)[_i];
                // Add the count of this rank to the count of the leader
                ((uint32_t *) agv_rc)[_hid] += _count;
            }
            /*
                For every leader, add the sum of the previous counts to his displacements
            */
            _displ = 0;
            for (_i = 0; _i < leaders_size; _i++){
                ((uint32_t *) agv_displ)[_i] = _displ;
                _displ += ((uint32_t *) agv_rc)[_i];
            }
        } while (0);

        args.args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
        args.args.dst.info_v.counts = (ucc_count_t *)agv_rc;
        args.args.dst.info_v.displacements = (ucc_aint_t *)agv_displ;

        UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS), &args, &tasks[n_tasks]), out, status);

        if (n_tasks > 1){ // TODO optimize
            UCC_CHECK_GOTO(
                ucc_event_manager_subscribe(tasks[n_tasks - 1], UCC_EVENT_COMPLETED,
                                            tasks[n_tasks], ucc_task_start_handler),
                out, status);
        }
        else{
            UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                            &schedule->super, UCC_EVENT_SCHEDULE_STARTED,
                            tasks[n_tasks], ucc_task_start_handler),
                        out, status);
        }
        UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, tasks[n_tasks]), out,
                       status);

        // The result of this collective is in dst.info_v.buffer, this should be the source buffer of the next collective
        args.args.src.info.buffer = args.args.dst.info_v.buffer; 
        n_tasks++;
    }

    // BCAST in the node
    // src.info.buffer -> src.info.buffer
    if (node_size > 1){

        args.args.coll_type = UCC_COLL_TYPE_BCAST;
        args.args.root      = node_root;
        args.args.src.info.count = 0;

        do { // This section need to be rewritten to support both uint32_t and uint64_t once the logic is approved
            int _i;
            for (_i = 0; _i < full_size; _i++) {
                args.args.src.info.count += ((uint32_t *)coll_args->args.dst.info_v.counts)[_i];
            }
        } while (0);

        UCC_CHECK_GOTO(
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]), out,
            status);
        UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, tasks[n_tasks]), out,
                    status);

        if (n_tasks > 1){ // TODO optimize
            UCC_CHECK_GOTO(
                ucc_event_manager_subscribe(tasks[n_tasks - 1], UCC_EVENT_COMPLETED,
                                            tasks[n_tasks], ucc_task_start_handler),
                out, status);
        }
        else{
            UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                            &schedule->super, UCC_EVENT_SCHEDULE_STARTED,
                            tasks[n_tasks], ucc_task_start_handler),
                        out, status);
        }

        // This collective writes to src.info.buffer, this should be the output buffer (dst.info_v.buffer)
        args.args.dst.info_v.buffer = args.args.src.info.buffer;
        n_tasks++;
    }

    schedule->super.post     = ucc_cl_hier_allgatherv_start;
    schedule->super.finalize = ucc_cl_hier_allgatherv_finalize;
    //schedule->super.triggered_post_setup = ucc_cl_hier_allgatherv_triggered_post_setup;
    *sched_p = schedule;

    return UCC_OK;

out:
    for (i = 0; i < n_tasks; i++) {
        tasks[i]->finalize(tasks[i]);
    }
    ucc_mc_free(cl_schedule->scratch);
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

static ucc_status_t ucc_cl_hier_allgatherv_node_split_frag_init(
    ucc_base_coll_args_t *coll_args, ucc_schedule_pipelined_t *sp,
    ucc_base_team_t *team, ucc_schedule_t **frag_p)
{
    int n_frags = sp->super.n_tasks;

    return ucc_cl_hier_allgatherv_node_split_init_schedule(coll_args, team,
                                                           frag_p, n_frags);
}

static ucc_status_t ucc_cl_hier_allgatherv_node_split_frag_setup(
    ucc_schedule_pipelined_t *schedule_p, ucc_schedule_t *frag, int frag_num)
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
        task                            = frag->tasks[i];
        task->bargs.args.src.info.count = frag_count;
        task->bargs.args.src.info.buffer =
            PTR_OFFSET(args->src.info.buffer, frag_offset * dt_size);
    }
    return UCC_OK;
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_allgatherv_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t *cl_team   = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_lib_config_t *cfg = &UCC_CL_HIER_TEAM_LIB(cl_team)->cfg;
    ucc_cl_hier_schedule_t   *schedule;
    int                       n_frags, pipeline_depth;
    ucc_status_t              status;

    if (UCC_IS_PERSISTENT(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }
    n_frags        = 1;
    pipeline_depth = 1;

    // Question: How to use this ? 
    // ucc_pipeline_nfrags_pdepth(&cfg->allgatherv_node_split_pipeline,
    //                           coll_args->args.src.info.count *
    //                           ucc_dt_size(coll_args->args.src.info.datatype),
    //                           &n_frags, &pipeline_depth);

    if (n_frags == 1) {
        return ucc_cl_hier_allgatherv_node_split_init_schedule(
            coll_args, team, (ucc_schedule_t **)task, n_frags);
    }

    schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }

    status = ucc_schedule_pipelined_init(
        coll_args, team, ucc_cl_hier_allgatherv_node_split_frag_init,
        ucc_cl_hier_allgatherv_node_split_frag_setup, pipeline_depth, n_frags,
        cfg->allgatherv_node_split_pipeline.order, &schedule->super);

    if (ucc_unlikely(status != UCC_OK)) {
        cl_error(team->context->lib,
                 "failed to init pipelined node split allgatherv schedule");
        goto err_pipe_init;
    }

    schedule->super.super.super.post     = ucc_cl_hier_allgatherv_start;
    schedule->super.super.super.finalize = ucc_cl_hier_allgatherv_finalize;
    *task                                = &schedule->super.super.super;
    return UCC_OK;

err_pipe_init:
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}
