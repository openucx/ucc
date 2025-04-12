/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv.h"
#include "unpack.h"
#include "../cl_hier_coll.h"
#include "core/ucc_team.h"

#define MAX_ALLGATHERV_TASKS 4

ucc_base_coll_alg_info_t
    ucc_cl_hier_allgatherv_algs[UCC_CL_HIER_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLGATHERV_ALG_GAB] =
            {.id   = UCC_CL_HIER_ALLGATHERV_ALG_GAB,
             .name = "gab",
             .desc = "gatherv + allgatherv + bcast"},
        [UCC_CL_HIER_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

static ucc_status_t ucc_cl_hier_allgatherv_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_allgatherv_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t         *schedule    = ucc_derived_of(task, ucc_schedule_t);
    ucc_cl_hier_schedule_t *cl_schedule = ucc_derived_of(task,
                                                         ucc_cl_hier_schedule_t);
    ucc_status_t            status;

    ucc_mc_free(cl_schedule->scratch);

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_finalize", 0);
    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

/* Return team_rank's node leader in team space */
static inline ucc_status_t find_leader_rank(ucc_base_team_t *team,
                                            ucc_rank_t       team_rank,
                                            ucc_rank_t      *rank_out)
{
    ucc_cl_hier_team_t *cl_team      = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_team_t         *core_team    = cl_team->super.super.params.team;
    ucc_rank_t         *node_leaders = NULL;
    ucc_status_t        status;

    ucc_assert(team_rank >= 0 && team_rank < UCC_CL_TEAM_SIZE(cl_team));
    ucc_assert(SBGP_EXISTS(cl_team, NODE_LEADERS));

    status = ucc_topo_get_node_leaders(core_team->topo, &node_leaders, NULL);
    if (UCC_OK != status) {
        cl_error(team->context->lib, "Could not get node leaders");
        return status;
    }

    *rank_out = node_leaders[team_rank];
    return UCC_OK;
}

/* Check if the ranks are block ordered. If they aren't, we'll need to 
   unpack the data after the allgatherv into the right position, even if the
   dst buffer is contiguous */
static inline ucc_status_t is_block_ordered(ucc_cl_hier_team_t *cl_team, int *ordered)
{
    ucc_topo_t *topo = cl_team->super.super.params.team->topo;
    ucc_sbgp_t *sbgp;
    int         is_block_ordered;

    if (cl_team->is_block_ordered != -1) {
        is_block_ordered = cl_team->is_block_ordered;
    } else {
        sbgp = ucc_topo_get_sbgp(topo, UCC_SBGP_FULL_HOST_ORDERED);
        is_block_ordered = ucc_ep_map_is_identity(&sbgp->map) ? 1 : 0;
        cl_team->is_block_ordered = is_block_ordered;
    }

    *ordered = is_block_ordered;

    return UCC_OK;
}

/* Node leader subgroup is always ordered by ascending host_id. If the team's ranks are
   not in the same order, then node leader subgroup allgatherv won't be enough, we'll
   have to use a staging buffer and unpack to reorder each leader's contribution.
   So, this func will check that the team ranks in the ldr sbgp are ascending */
static inline ucc_status_t is_host_ordered(ucc_cl_hier_team_t *cl_team, int *ordered)
{
    int         is_host_ordered;
    ucc_rank_t  max_rank, i, team_rank;

    if (cl_team->is_host_ordered != -1) {
        is_host_ordered = cl_team->is_host_ordered;
    } else {
        if (SBGP_EXISTS(cl_team, NODE_LEADERS)) {
            is_host_ordered = 1;
            max_rank = ucc_ep_map_eval(SBGP_MAP(cl_team, NODE_LEADERS), 0);
            for (i = 1; i < SBGP_SIZE(cl_team, NODE_LEADERS); i++) {
                team_rank = ucc_ep_map_eval(SBGP_MAP(cl_team, NODE_LEADERS), i);
                if (team_rank < max_rank) {
                    is_host_ordered = 0;
                    break;
                }
                max_rank = team_rank;
            }
        } else {
            is_host_ordered = 1;
        }
        cl_team->is_host_ordered = is_host_ordered;
    }

    *ordered = is_host_ordered;

    return UCC_OK;
}

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_allgatherv_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t     *cl_team          = ucc_derived_of(team,
                                                              ucc_cl_hier_team_t);
    ucc_coll_task_t        *tasks[MAX_ALLGATHERV_TASKS]
                                             = {NULL};
    ucc_rank_t              rank             = UCC_CL_TEAM_RANK(cl_team);
    ucc_rank_t              node_sbgp_size   = SBGP_SIZE(cl_team, NODE);
    ucc_rank_t              leader_sbgp_size = SBGP_SIZE(cl_team, NODE_LEADERS);
    ucc_rank_t              team_size        = UCC_CL_TEAM_SIZE(cl_team);
    ucc_topo_t             *topo             = team->params.team->topo;
    ucc_aint_t             *node_disps       = NULL;
    ucc_count_t            *node_counts      = NULL;
    ucc_aint_t             *leader_disps     = NULL;
    ucc_count_t            *leader_counts    = NULL;
    size_t                  dt_size          = ucc_dt_size(coll_args->args.
                                                           dst.info_v.datatype);
    int                     in_place         = 0;
    int                     is_contig        = 1;
    size_t                  disp_counter     = 0;
    ucc_schedule_t         *schedule;
    ucc_cl_hier_schedule_t *cl_schedule;
    ucc_status_t            status;
    ucc_base_coll_args_t    args, args_old;
    int                     n_tasks, i;
    size_t                  scratch_size;
    size_t                  node_counts_size;
    size_t                  node_disps_size;
    size_t                  leader_counts_size;
    size_t                  leader_disps_size;
    size_t                  total_count;
    void                   *buffer;
    void                   *node_gathered_data;
    ucc_rank_t              leader_team_rank;
    ucc_rank_t              leader_sbgp_rank;
    ucc_rank_t              team_rank;
    size_t                  leader_old_count;
    size_t                  add_count, new_count;
    int                     block_ordered, host_ordered;
    int                     ldr_sbgp_only;

    memcpy(&args,     coll_args, sizeof(args));
    memcpy(&args_old, coll_args, sizeof(args));

    in_place = UCC_IS_INPLACE(args.args);

    if (args.args.dst.info_v.mem_type != UCC_MEMORY_TYPE_HOST ||
        (!in_place && args.args.src.info.mem_type != UCC_MEMORY_TYPE_HOST) ||
        UCC_IS_PERSISTENT(args.args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    schedule = &ucc_cl_hier_get_schedule(cl_team)->super.super;
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    cl_schedule = ucc_derived_of(schedule, ucc_cl_hier_schedule_t);

    n_tasks   = 0;
    UCC_CHECK_GOTO(is_block_ordered(cl_team, &block_ordered), free_sched, status);
    UCC_CHECK_GOTO(is_host_ordered(cl_team, &host_ordered), free_sched, status);
    is_contig = UCC_COLL_IS_DST_CONTIG(&args.args) && block_ordered && host_ordered;
    /* handle the case where this rank may be the only one on this node */
    ldr_sbgp_only = !SBGP_ENABLED(cl_team, NODE) && SBGP_ENABLED(cl_team, NODE_LEADERS);
    
    UCC_CHECK_GOTO(ucc_schedule_init(schedule, &args, team), free_sched, status);

    node_counts_size   = node_sbgp_size * sizeof(ucc_count_t);
    node_disps_size    = node_sbgp_size * sizeof(ucc_aint_t);
    leader_counts_size = leader_sbgp_size * sizeof(ucc_count_t);
    leader_disps_size  = leader_sbgp_size * sizeof(ucc_aint_t);
    total_count        = ucc_coll_args_get_total_count(&args.args,
                            args.args.dst.info_v.counts, team_size);
    scratch_size       = node_counts_size + node_disps_size
                         + leader_counts_size + leader_disps_size;
    /* If the dst buf isn't contig, allocate and work on a contig scratch buffer */
    scratch_size      += (is_contig ? 0 : (total_count * dt_size));

    UCC_CHECK_GOTO(
        ucc_mc_alloc(&cl_schedule->scratch, scratch_size, UCC_MEMORY_TYPE_HOST),
        free_sched, status);
    memset(cl_schedule->scratch->addr, 0, scratch_size);

    node_counts        = PTR_OFFSET(cl_schedule->scratch->addr, 0);
    node_disps         = PTR_OFFSET(node_counts, node_counts_size);
    leader_counts      = PTR_OFFSET(node_disps, node_disps_size);
    leader_disps       = PTR_OFFSET(leader_counts, leader_counts_size);
    if (is_contig) {
        buffer         = args.args.dst.info_v.buffer;
    } else {
        buffer         = PTR_OFFSET(leader_disps, leader_disps_size);
    }
    node_gathered_data = NULL;

    /* If node ldr sbgp exists, calculate leader_counts, leader_disps, and set
       the dst buffer of the gatherv to the right displacements for the in-place
       node-leader allgatherv.
       Calculate this on non-node-leader ranks as well for the unpack phase */
    if((SBGP_ENABLED(cl_team, NODE) && SBGP_EXISTS(cl_team, NODE_LEADERS)) || ldr_sbgp_only) {
        /* Sum up the counts on each node to get the count for each node leader */
        for (i = 0; i < team_size; i++) {
            UCC_CHECK_GOTO(
                find_leader_rank(team, i, &leader_team_rank),
                free_scratch, status);
            leader_sbgp_rank = ucc_ep_map_local_rank(
                                            SBGP_MAP(cl_team, NODE_LEADERS),
                                            leader_team_rank);
            leader_old_count = ucc_coll_args_get_count(
                                            &args.args, leader_counts,
                                            leader_sbgp_rank);
            add_count        = ucc_coll_args_get_count(
                                            &args.args,
                                            args.args.dst.info_v.counts, i);
            new_count        = add_count + leader_old_count;
            ucc_coll_args_set_count(&args.args, leader_counts,
                                    leader_sbgp_rank, new_count);
        }

        /* Calculate leader_disps by adding each count to disp_counter to make
           a contiguous chunk */
        disp_counter = 0;
        for (i = 0; i < leader_sbgp_size; i++) {
            ucc_coll_args_set_displacement(&args.args, leader_disps,
                                            i, disp_counter);
            disp_counter += ucc_coll_args_get_count(&args.args,
                                                    leader_counts,
                                                    i);
        }

        if (SBGP_ENABLED(cl_team, NODE_LEADERS)) {
            node_gathered_data = PTR_OFFSET(buffer,
                                            dt_size *
                                                ucc_coll_args_get_displacement(
                                                    &args.args,
                                                    leader_disps,
                                                    SBGP_RANK(cl_team, NODE_LEADERS))
                                            );
        }
    }

    if (SBGP_ENABLED(cl_team, NODE)) {
        ucc_assert(n_tasks == 0);
        if (cl_team->top_sbgp == UCC_HIER_SBGP_NODE) {
            args.args.coll_type = UCC_COLL_TYPE_ALLGATHERV;
        } else {
            disp_counter = 0;
            for (i = 0; i < node_sbgp_size; i++) {
                team_rank =
                    ucc_ep_map_eval(SBGP_MAP(cl_team, NODE), i);
                ucc_coll_args_set_count(
                    &args.args, node_counts, i,
                    ucc_coll_args_get_count(&args.args,
                                            args.args.dst.info_v.counts,
                                            team_rank));
                ucc_coll_args_set_displacement(&args.args, node_disps,
                                               i, disp_counter);
                disp_counter += ucc_coll_args_get_count(&args.args,
                                                        node_counts, i);
            }

            if (in_place) {
                args.args.src.info.buffer   =
                    PTR_OFFSET(args.args.dst.info_v.buffer,
                               dt_size * ucc_coll_args_get_displacement(
                                            &args.args,
                                            args.args.dst.info_v.displacements,
                                            rank));
                args.args.src.info.count    = 
                    ucc_coll_args_get_count(&args.args,
                                            args.args.dst.info_v.counts,
                                            rank);
                args.args.src.info.datatype = args.args.dst.info_v.datatype;
                args.args.src.info.mem_type = args.args.dst.info_v.mem_type;
            }

            args.args.coll_type                = UCC_COLL_TYPE_GATHERV;
            args.args.root                     = topo->node_leader_rank_id;
            args.args.flags                   &= ~UCC_COLL_ARGS_FLAG_IN_PLACE;
            args.args.dst.info_v.displacements = node_disps;
            args.args.dst.info_v.counts        = node_counts;
            args.args.dst.info_v.buffer        = node_gathered_data;
        }
        UCC_CHECK_GOTO(
            ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]),
            free_scratch, status);
        n_tasks++;
    }

    args = args_old;

    /* Need to pack in case its not inplace or the buf isnt contig and we didnt do the gatherv */
    if (ldr_sbgp_only) {
        if (!in_place) {
            memcpy(node_gathered_data, args.args.src.info.buffer, args.args.src.info.count * ucc_dt_size(args.args.src.info.datatype));
        } else if (!is_contig) {
            memcpy(node_gathered_data,
                PTR_OFFSET(args.args.dst.info_v.buffer,
                            dt_size * ucc_coll_args_get_displacement(&args.args, args.args.dst.info_v.displacements, rank)),
                dt_size * ucc_coll_args_get_count(&args.args, args.args.dst.info_v.counts, rank));
        }
    }

    args = args_old;

    if (SBGP_ENABLED(cl_team, NODE_LEADERS)) {
        ucc_assert(cl_team->top_sbgp == UCC_HIER_SBGP_NODE_LEADERS);
        args.args.coll_type                = UCC_COLL_TYPE_ALLGATHERV;
        args.args.mask                    |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.args.flags                   |= UCC_COLL_ARGS_FLAG_IN_PLACE;
        args.args.flags                   |= UCC_COLL_ARGS_FLAG_CONTIG_DST_BUFFER;
        args.args.dst.info_v.buffer        = buffer;
        args.args.dst.info_v.displacements = leader_disps;
        args.args.dst.info_v.counts        = leader_counts;
        UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS), &args,
                                     &tasks[n_tasks]),
                       free_scratch, status);
        n_tasks++;
    }

    if (cl_team->top_sbgp != UCC_HIER_SBGP_NODE) {
        args                        = args_old;
        if (SBGP_ENABLED(cl_team, NODE)) {
            args.args.coll_type         = UCC_COLL_TYPE_BCAST;
            args.args.mask             |= UCC_COLL_ARGS_FIELD_FLAGS;
            args.args.flags            |= UCC_COLL_ARGS_FLAG_IN_PLACE;
            args.args.root              = topo->node_leader_rank_id;
            args.args.src.info.buffer   = buffer;
            args.args.src.info.count    = total_count;
            args.args.src.info.datatype = args_old.args.dst.info_v.datatype;
            args.args.src.info.mem_type = args_old.args.dst.info_v.mem_type;

            /* If using tl_shm and the shmem segment size is less than total_count,
            this node-level bcast will cause the allgatherv to fail and fall back */
            UCC_CHECK_GOTO(
                ucc_coll_init(SCORE_MAP(cl_team, NODE), &args, &tasks[n_tasks]),
                free_scratch, status);
            n_tasks++;
        }

        if (!is_contig) {
            args                          = args_old;
            args.args.src.info_v.datatype = args.args.dst.info_v.datatype;
            args.args.src.info_v.mem_type = args.args.dst.info_v.mem_type;
            args.args.src.info_v.buffer   = buffer;
            
            // Pass leader_disps and leader_counts through src.info_v for unpack
            args.args.src.info_v.displacements = leader_disps;
            args.args.src.info_v.counts        = leader_counts;

            UCC_CHECK_GOTO(
                ucc_cl_hier_allgatherv_unpack_init(&args, team, &tasks[n_tasks]),
                free_scratch, status);
            n_tasks++;
        }
    }

    UCC_CHECK_GOTO(ucc_event_manager_subscribe(
                       &schedule->super, UCC_EVENT_SCHEDULE_STARTED, tasks[0],
                       ucc_task_start_handler),
                   free_scratch, status);
    UCC_CHECK_GOTO(
        ucc_schedule_add_task(schedule, tasks[0]), free_scratch, status);
    for (i = 1; i < n_tasks; i++) {
        UCC_CHECK_GOTO(
            ucc_event_manager_subscribe(tasks[i - 1], UCC_EVENT_COMPLETED,
                                        tasks[i], ucc_task_start_handler),
            free_scratch, status);
        UCC_CHECK_GOTO(
            ucc_schedule_add_task(schedule, tasks[i]), free_scratch, status);
    }

    schedule->super.flags   |= UCC_COLL_TASK_FLAG_EXECUTOR;
    schedule->super.post     = ucc_cl_hier_allgatherv_start;
    schedule->super.finalize = ucc_cl_hier_allgatherv_finalize;
    *task                    = &schedule->super;
    return UCC_OK;

free_scratch:
    ucc_mc_free(cl_schedule->scratch);
free_sched:
    for (i = 0; i < n_tasks; i++) {
        tasks[i]->finalize(tasks[i]);
    }
    ucc_cl_hier_put_schedule(schedule);
    cl_error(team->context->lib, "failed to init cl hier allgatherv");
    return status;
}
