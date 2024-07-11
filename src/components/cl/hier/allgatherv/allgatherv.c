/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

/**
 * Question: How can uint32_t be casted to ucc_count_t (uint64_t) ? Isnt
 *  there a problem of size there?
 */

#include "allgatherv.h"
#include "../cl_hier_coll.h"
#include "core/ucc_team.h"

ucc_base_coll_alg_info_t ucc_cl_hier_allgatherv_algs[UCC_CL_HIER_ALLGATHERV_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLGATHERV_ALG_NODE_SPLIT] = {
            .id = UCC_CL_HIER_ALLGATHERV_ALG_NODE_SPLIT,
            .name = "node_split",
            .desc = "splitting allgatherv into three consecutive calls, first a gatherv"
                " inside the node, then an allgatherv between the leaders and then a bcast."
        },
        [UCC_CL_HIER_ALLGATHERV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}
};

/** TODO Not sure about this assumption:
* Can we consider the node leader to be the proc with the lowest rank in the node ?
* Otherwise these lines are not valid as the displacements don't "start" with the node leader:
* In SET_GATHER_COUNTS:
* ((_type *)_displs_gather)[_i] = _displ;
* _displ += _scount;
* In SET_ALLGATHERV_COUNTS:
* ((_type *) _displs)[_i] = coll_args->args.dst.info_v.displacements[_i];
*/

#define SET_GATHER_COUNTS(_type, _sbgp, _coll_args, _sc_gather, _rc_gather,     \
                            _displs_gather, _is_root)                           \
    do {                                                                        \
        int   _i;                                                               \
        _type _scount, _displ;                                                  \
        _sc_gather = (ucc_count_t) _coll_args->args.src.info.count;           \
        _displ = 0;                                                             \
        for (_i = 0; _is_root && _i < (_sbgp).group_size; _i++) {               \
            ucc_rank_t r = ucc_ep_map_eval((_sbgp).map, _i);                    \
            _scount      = ( (_type *) _coll_args->args.dst.info_v.counts)[r];   \
            ((_type *)_rc_gather)[_i] = _scount;                                \
            ((_type *)_displs_gather)[_i] = _displ;                             \
            _displ += _scount;                                                  \
        }                                                                       \
    } while (0)

// TODO Fix get_node_ix
#define SET_ALLGATHERV_COUNTS(_type, _sbgp, _coll_args, _sc, _rc, _displs, _full_size)          \
    do {                                                                                        \
        int _i;                                                                                 \
        _sc = 0;                                                                                \
        for (_i = 0; _i < (_sbgp).group_size; _i++){ \
            ((_type *) _rc)[_i] = 0; \
            ((_type *) _displs)[_i] = ((_type *)(coll_args)->args.dst.info_v.displacements)[_i]; \
        } \
        for (_i = 0; _i < _full_size; _i++){ \
            int _is_local = ucc_rank_on_local_node(_i, (team)->params.team->topo); \
            if (_is_local){ \
                _sc += (ucc_count_t) ((_type *)(_coll_args)->args.dst.info_v.counts)[_i]; \
            } \
            else{ \
                int _node_ix = _i % (_sbgp).group_size;  \
                ((_type *) _rc)[_node_ix] += ((_type *)(_coll_args)->args.dst.info_v.counts)[_i]; \
            } \
        } \
    } while (0)

#define SET_BCAST_COUNT(_type, _full_size, _coll_args, _count) \
    do { \
        int _i; \
        _count = 0; \
        for (_i=0; _i < (_full_size); _i++){ \
            _count += ((_type *)(_coll_args)->args.dst.info_v.counts)[_i]; \
        } \
    } while (0)


static ucc_status_t ucc_cl_hier_allgatherv_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_allgatherv_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *cl_schedule = ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_allgatherv_finalize", 0);

    ucc_assert(cl_schedule->super.super.n_tasks == 2 ||
               cl_schedule->super.super.n_tasks == 3);

    if (cl_schedule->scratch) {
        ucc_mc_free(cl_schedule->scratch);
    }

    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(&cl_schedule->super.super);
    return status;
}

// Question: Do i need this function ? -> Tomi/Sergey
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
    printf("Waiting in allgatherv init,  pid=%d\n", getpid());
    int wait = 1;
    while (wait){
        sleep(1);
    }

    ucc_cl_hier_team_t     *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_schedule_t *cl_schedule;
    ucc_schedule_t         *schedule;
    ucc_status_t            status;
    ucc_base_coll_args_t    args;
    ucc_coll_task_t        *task_gather, *task_allgatherv, *task_bcast;
    int                     c64, d64;
    void                    *rc_gather, *displs_gather;
    ucc_count_t             sc_gather, sc_ag;
    void                    *rc_ag, *displs_ag;
    void                    *count_bcast;
    ucc_rank_t              full_size, node_size, leaders_size;
    size_t                  elem_size;
    ucc_rank_t              node_root = 0;

    ucc_base_coll_args_t    g_args, agv_args, b_args;

    int is_root = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_rank == node_root;

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
    memcpy(&args, coll_args, sizeof(args)); // Remove that and just use coll_args
    UCC_CHECK_GOTO(ucc_schedule_init(schedule, &args, team), error, status);

    full_size = cl_team->sbgps[UCC_HIER_SBGP_FULL].sbgp->group_size;
    node_size = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_size;
    leaders_size = cl_team->sbgps[UCC_HIER_SBGP_NODE_LEADERS].sbgp->group_size;
    elem_size = c64 ? 8 : 4;

    // Question: is there a way to init the buffer with zeros ?
    size_t scratch_size = elem_size * (node_size*2 + leaders_size*2 + 1);
    status = ucc_mc_alloc(&cl_schedule->scratch,
                          scratch_size,
                          UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(UCC_OK != status)) {
        cl_error(team->context->lib,
                 "failed to allocate %zd bytes for full counts",
                  scratch_size); // TODO
        goto error;
    }

    memcpy(&g_args, coll_args, sizeof(g_args));

    // alloc buffers
    rc_gather       = cl_schedule->scratch->addr;     // Size node side
    displs_gather   = PTR_OFFSET(rc_gather, node_size * elem_size); // Size: node size
    rc_ag           = PTR_OFFSET(displs_gather, node_size * elem_size); // Size: leaders size
    displs_ag       = PTR_OFFSET(rc_ag, leaders_size * elem_size); // Size: leaders size
    count_bcast     = PTR_OFFSET(displs_ag, leaders_size * elem_size); // Size: 1

    // Gather in the node

    if (c64){
        SET_GATHER_COUNTS(uint64_t, *cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp, coll_args, sc_gather,
                rc_gather, displs_gather, is_root);
    }
    else{
        SET_GATHER_COUNTS(uint32_t, *cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp, coll_args, sc_gather,
                rc_gather, displs_gather, is_root);
    }

    g_args.args.coll_type = UCC_COLL_TYPE_GATHERV;
    // TODO Select optimal root (closest to HCA)
    g_args.args.root = node_root;
    g_args.args.src.info.count = *((ucc_count_t *) sc_gather);
    g_args.args.dst.info_v.counts = (ucc_count_t *) rc_gather;
    g_args.args.dst.info_v.displacements = (ucc_aint_t *) displs_gather;

    UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE),
                                 &g_args, &task_gather),
                   err_init_gather, status);

    // Allgatherv in the net
    if (is_root){
        memcpy(&agv_args, coll_args, sizeof(agv_args));
        agv_args.args.coll_type = UCC_COLL_TYPE_ALLGATHERV;

        if (c64){
            SET_ALLGATHERV_COUNTS(uint64_t, *cl_team->sbgps[UCC_HIER_SBGP_NODE_LEADERS].sbgp, coll_args,
                sc_ag, rc_ag, displs_ag, full_size);
        }
        else{
            SET_ALLGATHERV_COUNTS(uint32_t, *cl_team->sbgps[UCC_HIER_SBGP_NODE_LEADERS].sbgp, coll_args,
                sc_ag, rc_ag, displs_ag, full_size);
        }

        agv_args.args.src.info.count =  *((ucc_count_t *) sc_ag);
        agv_args.args.dst.info_v.counts =  (ucc_count_t *) rc_ag;
        agv_args.args.dst.info_v.displacements =  (ucc_aint_t *) displs_ag;

        UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE_LEADERS),
                                 &agv_args, &task_allgatherv),
                   err_init_allgatherv, status);
    }

    // BCAST in the node
    memcpy(&b_args, coll_args, sizeof(b_args));

    if (c64){
        SET_BCAST_COUNT(uint64_t, full_size, coll_args, *((uint64_t *) count_bcast));
    }
    else {
        SET_BCAST_COUNT(uint32_t, full_size, coll_args, *((uint32_t *) count_bcast));
    }
    b_args.args.root = node_root;
    // Question: why (ucc_count_t *) if it can be uint32_t ? because ucc_count_t is uint64_t
    b_args.args.src.info.count = *((ucc_count_t *) count_bcast);

    UCC_CHECK_GOTO(ucc_coll_init(SCORE_MAP(cl_team, NODE),
                                 &b_args, &task_bcast),
                   err_init_bcast, status);

    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, task_gather), err, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, task_gather, UCC_EVENT_SCHEDULE_STARTED), err, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, task_allgatherv), err, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, task_gather, UCC_EVENT_COMPLETED), err, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, task_bcast), err, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, task_gather, UCC_EVENT_COMPLETED), err, status);

    schedule->super.post = ucc_cl_hier_allgatherv_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_cl_hier_allgatherv_finalize;
    schedule->super.triggered_post_setup = ucc_cl_hier_allgatherv_triggered_post_setup;
    *task = &schedule->super;
    return UCC_OK;

err: ;
err_init_bcast:
    ucc_collective_finalize(&task_bcast->super);
err_init_allgatherv:
    if (is_root){
        ucc_collective_finalize(&task_allgatherv->super);
    }
err_init_gather:
    ucc_collective_finalize(&task_allgatherv->super);
error:
    ucc_mc_free(cl_schedule->scratch);
    ucc_cl_hier_put_schedule(schedule);
    return status;
}

