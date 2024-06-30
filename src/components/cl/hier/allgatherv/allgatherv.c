/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "allgatherv.h"
#include "../cl_hier_coll.h"
#include "core/ucc_team.h"


// Question leave thresholds ?
#define SET_GATHER_COUNTS(_type, _sbgp, _coll_args, _sc_gather, _rc_gather,     \
                            _displs_gather, _is_root)                           \
    do {                                                                        \
        int   _i;                                                               \
        _type _scount, _rcount, _displ;                                         \
        *_sc_gather = _coll_args->args.src.info_v.counts[0]                      \
        _displ = 0;                                                             \
        for (_i = 0; _is_root && _i < (_sbgp)->group_size; _i++) {              \
            ucc_rank_t r = ucc_ep_map_eval((_sbgp)->map, _i);                   \
            _scount      = ((_type *)(_coll_args)->args.src.info_v.counts)[r];  \
            ((_type *)_rc_gather)[_i] = _scount;                                \
            ((_type *)_displs_gather)[_i] = _displ;                             \
            _displ += _scount;                                                  \
        }                                                                       \
    } while (0)


// TODO
ucc_base_coll_alg_info_t ucc_cl_hier_allgatherv_algs[UCC_CL_HIER_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLGATHERV_ALG_NODE_SPLIT] =
            {.id   = UCC_CL_HIER_ALLTOALLV_ALG_NODE_SPLIT,
             .name = "node_split",
             .desc = "splitting allgatherv into two concurrent a2av calls"
                     " withing the node and outside of it"},
        [UCC_CL_HIER_ALLTOALLV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

// END OF TODO
//
static ucc_status_t ucc_cl_hier_allgatherv_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_start", 0);
    return ucc_schedule_start(task);
}


static ucc_status_t ucc_cl_hier_allgatherv_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule = ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_finalize", 0);
    // TODO
    ucc_assert(schedule->super.super.n_tasks == 1 ||
               schedule->super.super.n_tasks == 2);
    if (schedule->scratch) {
        ucc_mc_free(schedule->scratch);
    }
    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

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


UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_allgatherv_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t     *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_lib_t      *cl_lib  = UCC_CL_HIER_TEAM_LIB(cl_team);
    int                     full_only = 0;
    ucc_cl_hier_schedule_t *cl_schedule;
    ucc_schedule_t         *schedule;
    ucc_status_t            status;
    ucc_base_coll_args_t    args;
    ucc_coll_task_t        *task_gather, *task_allgatherv, *task_bcast;
    int                     c64, d64;
    void                    *sc_gather, *rc_gather, *displs_gather;
    ucc_rank_t              full_size, node_size;
    size_t                  sdt_size, rdt_size;
    ucc_sbgp_t             *sbgp;
    size_t                  elem_size;
    ucc_rank_t              node_root = 0; // Question: configure ?
    ucc_rank_t              node_rank, net_rank;    // TODO

    ucc_base_coll_args_t    g_args, agv_args, b_args;

    c64 = UCC_COLL_ARGS_COUNT64(&coll_args->args);
    d64 = UCC_COLL_ARGS_DISPL64(&coll_args->args);

    if (c64 ^ d64) {
        cl_debug(team->context->lib, "mixed 64 bit count/displ mode is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    cl_schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!cl_schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    schedule = &cl_schedule->super.super;
    memcpy(&args, coll_args, sizeof(args));
    UCC_CHECK_GOTO(ucc_schedule_init(schedule, &args, team), error, status);

    full_size = cl_team->sbgps[UCC_HIER_SBGP_FULL].sbgp->group_size;
    node_size = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_size;
    elem_size = c64 ? 8 : 4;

    status = ucc_mc_alloc(&cl_schedule->scratch,
                          elem_size * (1 + 2*node_size),
                          UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(UCC_OK != status)) {
        cl_error(team->context->lib,
                 "failed to allocate %zd bytes for full counts",
                 elem_size * (full_size + node_size) * 4);
        goto error;
    }

    memcpy(&g_args, coll_args, sizeof(g_args));
    memcpy(&ag_args, coll_args, sizeof(ag_args));
    memcpy(&b_args, coll_args, sizeof(b_args));

    // TODO alloc buffers
    sc_gather       = cl_schedule->scratch->addr;     // Size 1
    rc_gather       = PTR_OFFSET(sc_gather, elem_size); // Size: node size
    displs_gather   = PTR_OFFSET(rc_gather, node_size * elem_size); // Size: node size
    //next          = PTR_OFFSET(displs_gather, node_size * elem_size);

    // my code
    // Gather in the node
    ucc_rank_t node_r, net_r;
    ucc_sbgp_t g_sbgp = *cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp;

    if (c64){
        SET_GATHER_COUNTS(uint64_t, sbgp, coll_args, sc_gather,
                rc_gather, displs_gather, node_rank == node_root);
    }
    else{
        SET_GATHER_COUNTS(uint32_t, sbgp, coll_args, sc_gather,
                rc_gather, displs_gather, node_rank == node_root);
    }

    g_args.args.coll_type = UCC_COLL_TYPE_GATHERV;
    // TODO: Ask some1 if root node should be configurable, for example in a situation
    // where rank 0 is not optimal (ex if NIC is close to rank 1)
    g_args.args.root = node_root;
    g_args.args.src.info_v.counts = (ucc_count_t *) sc_gather;
    g_args.args.dst.info_v.counts = (ucc_count_t *) rc_gather;
    g_args.args.dst.info_v.displacements = (ucc_aint_t *) displs_gather;

    UCC_CHECK_GOTO(ucc_coll_init(cl_team->sbgps[UCC_HIER_SBGP_NODE].score_map,
                                 &g_args, &task_gather),
                   err_init_gather, status);

    // Allgatherv in the net

    // BCAST in the node


err_init_gather:
    // TODO
    ucc_mc_free(cl_schedule->scratch);
error:
    ucc_cl_hier_put_schedule(schedule);
    return status;
}
