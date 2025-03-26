/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "unpack.h"

ucc_status_t ucc_cl_hier_allgatherv_unpack_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *cl_schedule = ucc_derived_of(task,
                                                         ucc_cl_hier_schedule_t);

    ucc_mc_free(cl_schedule->scratch);

    return UCC_OK;
}

void ucc_cl_hier_allgatherv_unpack_progress(ucc_coll_task_t *task)
{
    ucc_schedule_t          *schedule    = ucc_derived_of(task, ucc_schedule_t);
    ucc_cl_hier_schedule_t  *cl_schedule = ucc_derived_of(schedule,
                                                          ucc_cl_hier_schedule_t);
    ucc_rank_t              *n_tasks     = cl_schedule->scratch->addr;
    ucc_ee_executor_task_t **tasks       = PTR_OFFSET(
                                                cl_schedule->scratch->addr,
                                                sizeof(ucc_rank_t));
    ucc_status_t             st          = UCC_OK;
    ucc_rank_t               i;
    ucc_ee_executor_task_t  *etask;

    for (i = 0; i < *n_tasks; i++) {
        etask = tasks[i];
        if (etask != NULL) {
            st = ucc_ee_executor_task_test(etask);
            if (st == UCC_OK) {
                ucc_ee_executor_task_finalize(etask);
                tasks[i] = NULL;
            } else {
                if (ucc_likely(st > 0)) {
                    st = UCC_INPROGRESS;
                }
                goto out;
            }
        }
    }

out:
    schedule->super.status       = st;
}

ucc_status_t ucc_cl_hier_allgatherv_unpack_start(ucc_coll_task_t *task)
{
    ucc_schedule_t             *schedule    = ucc_derived_of(task,
                                                             ucc_schedule_t);
    ucc_cl_hier_team_t         *cl_team     = ucc_derived_of(task->team,
                                                             ucc_cl_hier_team_t);
    ucc_rank_t                  team_size   = UCC_CL_TEAM_SIZE(cl_team);
    ucc_coll_args_t            *args        = &task->bargs.args;
    ucc_ee_executor_task_args_t eargs       = {0};
    ucc_cl_hier_schedule_t     *cl_schedule = ucc_derived_of(schedule,
                                                    ucc_cl_hier_schedule_t);
    ucc_rank_t                 *n_tasks     = cl_schedule->scratch->addr;
    ucc_ee_executor_task_t    **tasks       = PTR_OFFSET(
                                                cl_schedule->scratch->addr,
                                                sizeof(ucc_rank_t));
    size_t                      src_dt_size = ucc_dt_size(
                                                args->src.info_v.datatype);
    size_t                      dst_dt_size = ucc_dt_size(
                                                args->dst.info_v.datatype);
    ucc_topo_t                 *topo = task->team->params.team->topo;
    ucc_ee_executor_t          *exec;
    ucc_status_t                status;
    ucc_rank_t                  i;
    size_t                      src_rank_count;
    size_t                      dst_rank_count;
    size_t                      src_rank_disp;
    size_t                      dst_rank_disp;
    ucc_rank_t                 *node_leaders = NULL;
    ucc_rank_t                 *per_node_leaders = NULL;
    ucc_sbgp_t                 *all_nodes = NULL;
    int                         n_nodes;
    ucc_rank_t                  node_id;
    ucc_rank_t                  node_rank_offset;
    ucc_rank_t                  team_rank;

    UCC_CHECK_GOTO(
        ucc_coll_task_get_executor(&schedule->super, &exec),
        out, status);
    eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;

    *n_tasks = 0;

    // Get the node leaders and all nodes information
    UCC_CHECK_GOTO(
        ucc_topo_get_node_leaders(topo, &node_leaders, &per_node_leaders),
        out, status);
    
    UCC_CHECK_GOTO(
        ucc_topo_get_all_nodes(topo, &all_nodes, &n_nodes),
        out, status);

    for (i = 0; i < team_size; i++) {
        // Find which node this rank belongs to
        for (node_id = 0; node_id < n_nodes; node_id++) {
            if (all_nodes[node_id].status == UCC_SBGP_ENABLED) {
                for (node_rank_offset = 0; node_rank_offset < all_nodes[node_id].group_size; node_rank_offset++) {
                    team_rank = ucc_ep_map_eval(all_nodes[node_id].map, node_rank_offset);
                    if (team_rank == i) {
                        // Found the node and position for this team rank
                        goto found_rank;
                    }
                }
            }
        }
found_rank:
        // Get counts and displacements for destination (team order)
        dst_rank_count = ucc_coll_args_get_count(args, args->dst.info_v.counts, i);
        dst_rank_disp = ucc_coll_args_get_displacement(args, args->dst.info_v.displacements, i);
        
        // Source displacement needs to account for node-based ordering
        // Calculate offset in source buffer based on node_id and rank's position within node
        src_rank_disp = 0;
        // Add up counts for all preceding nodes
        for (ucc_rank_t prev_node = 0; prev_node < node_id; prev_node++) {
            if (all_nodes[prev_node].status == UCC_SBGP_ENABLED) {
                for (ucc_rank_t prev_node_rank = 0; prev_node_rank < all_nodes[prev_node].group_size; prev_node_rank++) {
                    ucc_rank_t prev_team_rank = ucc_ep_map_eval(all_nodes[prev_node].map, prev_node_rank);
                    src_rank_disp += ucc_coll_args_get_count(args, args->dst.info_v.counts, prev_team_rank);
                }
            }
        }
        
        // Add offset within current node
        for (ucc_rank_t j = 0; j < node_rank_offset; j++) {
            ucc_rank_t curr_team_rank = ucc_ep_map_eval(all_nodes[node_id].map, j);
            src_rank_disp += ucc_coll_args_get_count(args, args->dst.info_v.counts, curr_team_rank);
        }
        
        src_rank_count = dst_rank_count;
        ucc_assert(src_rank_count * src_dt_size == dst_rank_count * dst_dt_size);
        
        eargs.copy.src = PTR_OFFSET(args->src.info_v.buffer, src_rank_disp * src_dt_size);
        eargs.copy.dst = PTR_OFFSET(args->dst.info_v.buffer, dst_rank_disp * dst_dt_size);
        eargs.copy.len = dst_rank_count * dst_dt_size;
        
        if (eargs.copy.src != eargs.copy.dst) {
            UCC_CHECK_GOTO(
                ucc_ee_executor_task_post(exec, &eargs, &tasks[*n_tasks]),
                out, status);
            (*n_tasks)++;
        }
    }

    schedule->super.status = UCC_INPROGRESS;

    ucc_progress_queue_enqueue(cl_team->super.super.context->ucc_context->pq,
                               task);

    return UCC_OK;
out:
    return status;
}

ucc_status_t ucc_cl_hier_allgatherv_unpack_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h)
{
    ucc_cl_hier_team_t     *cl_team   = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_rank_t              team_size = UCC_CL_TEAM_SIZE(cl_team);
    ucc_status_t            status;
    ucc_schedule_t         *schedule;
    ucc_cl_hier_schedule_t *cl_schedule;
    size_t                  scratch_size;

    schedule = &ucc_cl_hier_get_schedule(cl_team)->super.super;
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    cl_schedule = ucc_derived_of(schedule, ucc_cl_hier_schedule_t);

    UCC_CHECK_GOTO(
        ucc_schedule_init(schedule, coll_args, team), free_schedule, status);

    /* Holds n_tasks and n_tasks # of ucc_ee_executor_task_t pointers */
    scratch_size = sizeof(ucc_rank_t) + team_size * sizeof(ucc_ee_executor_task_t*);
    UCC_CHECK_GOTO(
        ucc_mc_alloc(&cl_schedule->scratch, scratch_size, UCC_MEMORY_TYPE_HOST),
        free_schedule, status);

    schedule->super.flags   |= UCC_COLL_TASK_FLAG_EXECUTOR;
    schedule->super.post     = ucc_cl_hier_allgatherv_unpack_start;
    schedule->super.progress = ucc_cl_hier_allgatherv_unpack_progress;
    schedule->super.finalize = ucc_cl_hier_allgatherv_unpack_finalize;

    *task_h = &schedule->super;

    return UCC_OK;

free_schedule:
    ucc_cl_hier_put_schedule(schedule);
    return status;
}
