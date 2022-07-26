/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */

#include "alltoallv.h"
#include "../cl_hier_coll.h"
#include "core/ucc_team.h"

ucc_base_coll_alg_info_t
    ucc_cl_hier_alltoallv_algs[UCC_CL_HIER_ALLTOALLV_ALG_LAST + 1] = {
        [UCC_CL_HIER_ALLTOALLV_ALG_NODE_SPLIT] =
            {.id   = UCC_CL_HIER_ALLTOALLV_ALG_NODE_SPLIT,
             .name = "node_split",
             .desc = "splitting alltoallv into two concurrent a2av calls"
                     " withing the node and outside of it"},
        [UCC_CL_HIER_ALLTOALLV_ALG_LAST] = {
            .id = 0, .name = NULL, .desc = NULL}};

static ucc_status_t ucc_cl_hier_alltoallv_start(ucc_coll_task_t *task)
{
    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_alltoallv_start", 0);
    return ucc_schedule_start(task);
}

static ucc_status_t ucc_cl_hier_alltoallv_finalize(ucc_coll_task_t *task)
{
    ucc_cl_hier_schedule_t *schedule =
        ucc_derived_of(task, ucc_cl_hier_schedule_t);
    ucc_status_t status;

    UCC_CL_HIER_PROFILE_REQUEST_EVENT(task, "cl_hier_alltoallv_finalize", 0);
    ucc_assert(schedule->super.super.n_tasks == 2);
    ucc_mc_free(schedule->scratch);
    status = ucc_schedule_finalize(task);
    ucc_cl_hier_put_schedule(&schedule->super.super);
    return status;
}

#define SET_FULL_COUNTS(_type, _sbgp, _coll_args, _team, _node_thresh,         \
                        _sdt_size, _rdt_size, _sc_full, _sd_full, _rc_full,    \
                        _rd_full)                                              \
    do {                                                                       \
        int   _i, _is_local;                                                   \
        _type _scount, _rcount;                                                \
                                                                               \
        for (_i = 0; _i < (_sbgp)->group_size; _i++) {                         \
            _scount = ((_type *)(_coll_args)->args.src.info_v.counts)[_i];     \
            _rcount = ((_type *)(_coll_args)->args.dst.info_v.counts)[_i];     \
            _is_local =                                                        \
                ucc_rank_on_local_node(_i, (_team)->params.team->topo);        \
            if ((_scount * _sdt_size > (_node_thresh)) && _is_local) {         \
                ((_type *)_sc_full)[_i] = 0;                                   \
            } else {                                                           \
                ((_type *)_sc_full)[_i] = _scount;                             \
                ((_type *)_sd_full)[_i] =                                      \
                    ((_type *)(_coll_args)                                     \
                         ->args.src.info_v.displacements)[_i];                 \
            }                                                                  \
            if ((_rcount * _rdt_size > (_node_thresh)) && _is_local) {         \
                ((_type *)_rc_full)[_i] = 0;                                   \
            } else {                                                           \
                ((_type *)_rc_full)[_i] = _rcount;                             \
                ((_type *)_rd_full)[_i] =                                      \
                    ((_type *)(_coll_args)                                     \
                         ->args.dst.info_v.displacements)[_i];                 \
            }                                                                  \
        }                                                                      \
    } while (0)

#define SET_NODE_COUNTS(_type, _sbgp, _coll_args, _node_thresh, _sdt_size,     \
                        _rdt_size, _sc_node, _sd_node, _rc_node, _rd_node)     \
    do {                                                                       \
        int   _i;                                                              \
        _type _scount, _rcount;                                                \
                                                                               \
        for (_i = 0; _i < (_sbgp)->group_size; _i++) {                         \
            ucc_rank_t r = ucc_ep_map_eval((_sbgp)->map, _i);                  \
            _scount      = ((_type *)(_coll_args)->args.src.info_v.counts)[r]; \
            _rcount      = ((_type *)(_coll_args)->args.dst.info_v.counts)[r]; \
            if (_scount * _sdt_size <= (_node_thresh)) {                       \
                ((_type *)_sc_node)[_i] = 0;                                   \
            } else {                                                           \
                ((_type *)_sc_node)[_i] = _scount;                             \
                ((_type *)_sd_node)[_i] =                                      \
                    ((_type *)(_coll_args)->args.src.info_v.displacements)[r]; \
            }                                                                  \
            if (_rcount * _rdt_size <= (_node_thresh)) {                       \
                ((_type *)_rc_node)[_i] = 0;                                   \
            } else {                                                           \
                ((_type *)_rc_node)[_i] = _rcount;                             \
                ((_type *)_rd_node)[_i] =                                      \
                    ((_type *)(_coll_args)->args.dst.info_v.displacements)[r]; \
            }                                                                  \
        }                                                                      \
    } while (0)

ucc_status_t ucc_cl_hier_alltoallv_triggered_post_setup(ucc_coll_task_t *task)
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

UCC_CL_HIER_PROFILE_FUNC(ucc_status_t, ucc_cl_hier_alltoallv_init,
                         (coll_args, team, task),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task)
{
    ucc_cl_hier_team_t     *cl_team = ucc_derived_of(team, ucc_cl_hier_team_t);
    ucc_cl_hier_lib_t      *cl_lib  = UCC_CL_HIER_TEAM_LIB(cl_team);
    ucc_cl_hier_schedule_t *cl_schedule;
    ucc_schedule_t         *schedule;
    ucc_status_t            status;
    ucc_base_coll_args_t    args;
    ucc_coll_task_t        *task_node, *task_full;
    int                     c64, d64;
    void                   *sc_full, *sd_full, *rc_full, *rd_full;
    void                   *sc_node, *sd_node, *rc_node, *rd_node;
    ucc_rank_t              full_size, node_size;
    size_t                  sdt_size, rdt_size;
    ucc_sbgp_t             *sbgp;
    size_t                  elem_size;

    if (UCC_IS_INPLACE(coll_args->args)) {
        cl_debug(team->context->lib, "inplace alltoallv is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!SBGP_ENABLED(cl_team, NODE) || !SBGP_ENABLED(cl_team, FULL)) {
        cl_debug(team->context->lib, "alltoallv requires NODE and FULL sbgps");
        return UCC_ERR_NOT_SUPPORTED;
    }

    c64 = UCC_COLL_ARGS_COUNT64(&coll_args->args);
    d64 = UCC_COLL_ARGS_DISPL64(&coll_args->args);

    if (c64 ^ d64) {
        cl_debug(team->context->lib,
                 "mixed 64 bit count/displ mode is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    cl_schedule = ucc_cl_hier_get_schedule(cl_team);
    if (ucc_unlikely(!cl_schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    schedule = &cl_schedule->super.super;
    memcpy(&args, coll_args, sizeof(args));
    status = ucc_schedule_init(schedule, &args, team);
    if (ucc_unlikely(UCC_OK != status)) {
        goto error;
    }

    full_size = cl_team->sbgps[UCC_HIER_SBGP_FULL].sbgp->group_size;
    node_size = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp->group_size;

    elem_size = c64 ? 8 : 4;
    status    = ucc_mc_alloc(&cl_schedule->scratch,
                             elem_size * (full_size + node_size) * 4,
                             UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(UCC_OK != status)) {
        cl_error(team->context->lib,
                 "failed to allocate %zd bytes for full counts",
                 elem_size * (full_size + node_size) * 4);
        goto error;
    }

    sc_full = cl_schedule->scratch->addr;
    sd_full = PTR_OFFSET(sc_full, full_size * elem_size);
    rc_full = PTR_OFFSET(sc_full, full_size * elem_size * 2);
    rd_full = PTR_OFFSET(sc_full, full_size * elem_size * 3);

    sc_node = PTR_OFFSET(sc_full, full_size * elem_size * 4);
    sd_node = PTR_OFFSET(sc_node, node_size * elem_size);
    rc_node = PTR_OFFSET(sc_node, node_size * elem_size * 2);
    rd_node = PTR_OFFSET(sc_node, node_size * elem_size * 3);

    /* Duplicate FULL a2av info and alloc task */
    sbgp = cl_team->sbgps[UCC_HIER_SBGP_FULL].sbgp;
    ucc_assert(sbgp->group_size == team->params.size);
    sdt_size = ucc_dt_size(coll_args->args.src.info_v.datatype);
    rdt_size = ucc_dt_size(coll_args->args.dst.info_v.datatype);

    if (c64) {
        SET_FULL_COUNTS(uint64_t, sbgp, coll_args, team,
                        cl_lib->cfg.a2av_node_thresh, sdt_size, rdt_size,
                        sc_full, sd_full, rc_full, rd_full);
    } else {
        SET_FULL_COUNTS(uint32_t, sbgp, coll_args, team,
                        cl_lib->cfg.a2av_node_thresh, sdt_size, rdt_size,
                        sc_full, sd_full, rc_full, rd_full);
    }
    args.args.src.info_v.counts        = (ucc_aint_t *)sc_full;
    args.args.dst.info_v.counts        = (ucc_aint_t *)rc_full;
    args.args.src.info_v.displacements = (ucc_aint_t *)sd_full;
    args.args.dst.info_v.displacements = (ucc_aint_t *)rd_full;

    status = ucc_coll_init(cl_team->sbgps[UCC_HIER_SBGP_FULL].score_map, &args,
                           &task_full);
    if (UCC_OK != status) {
        cl_error(team->context->lib, "failed to init full a2av task");
        goto err_init_1;
    }

    /* Setup NODE a2av */
    sbgp = cl_team->sbgps[UCC_HIER_SBGP_NODE].sbgp;

    if (c64) {
        SET_NODE_COUNTS(uint64_t, sbgp, coll_args, cl_lib->cfg.a2av_node_thresh,
                        sdt_size, rdt_size, sc_node, sd_node, rc_node, rd_node);

    } else {
        SET_NODE_COUNTS(uint32_t, sbgp, coll_args, cl_lib->cfg.a2av_node_thresh,
                        sdt_size, rdt_size, sc_node, sd_node, rc_node, rd_node);
    }

    args.args.src.info_v.counts        = (ucc_aint_t *)sc_node;
    args.args.dst.info_v.counts        = (ucc_aint_t *)rc_node;
    args.args.src.info_v.displacements = (ucc_aint_t *)sd_node;
    args.args.dst.info_v.displacements = (ucc_aint_t *)rd_node;
    status = ucc_coll_init(cl_team->sbgps[UCC_HIER_SBGP_NODE].score_map, &args,
                           &task_node);
    if (UCC_OK != status) {
        cl_error(team->context->lib, "failed to init full a2av task");
        goto err_init_2;
    }
    ucc_schedule_add_task(schedule, task_node);
    ucc_schedule_add_task(schedule, task_full);
    ucc_task_subscribe_dep(&schedule->super, task_node,
                           UCC_EVENT_SCHEDULE_STARTED);
    ucc_task_subscribe_dep(&schedule->super, task_full,
                           UCC_EVENT_SCHEDULE_STARTED);

    schedule->super.post           = ucc_cl_hier_alltoallv_start;
    schedule->super.progress       = NULL;
    schedule->super.finalize       = ucc_cl_hier_alltoallv_finalize;
    schedule->super.triggered_post = ucc_triggered_post;
    schedule->super.triggered_post_setup =
        ucc_cl_hier_alltoallv_triggered_post_setup;
    *task = &schedule->super;
    return UCC_OK;

err_init_2:
    ucc_collective_finalize(&task_full->super);
err_init_1:
    ucc_mc_free(cl_schedule->scratch);
error:
    ucc_cl_hier_put_schedule(schedule);
    return status;
}
