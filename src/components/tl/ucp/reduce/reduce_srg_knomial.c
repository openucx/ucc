/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "reduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include "../reduce_scatter/reduce_scatter.h"
#include "../gather/gather.h"
#include "../allgather/allgather.h"

/* SRG - scatter-reduce-gather knomial algorithm
   1. The algorithm performs collective reduce operation as a sequence of
      K-nomial Reduce-Scatter followed by K-nomial (with the same radix K)
      gather.
   2. In essence this is an extension of the Bi-nomial SRA algorithm algorithm
      proposed by Rabenseifner2004 (https://doi.org/10.1007/978-3-540-24685-5_1).
      The extension adds the support for arbitrary radix.
   3. The algorithm targets Large message sizes (ie. optimized for max bandwidth).
   4. If number of ranks in the team can not form a full radix subtree
      (this means the team size is not a power of the radix) then there will be
      "extra" ranks which don't participate in the main exchange loop. They
      will send the data to their "proxy" ranks in the beginning and then wait
      for the response with the final data.
   5. The knomial reduce-scatter and gather primitives can be used separately.
      However, if they are used together as part of SRG reduce one has to
      provide the same radix for both routines.
   6. After the completion of reduce-scatter phase the local result (at non EXTRA
      ranks) will be located in the dst buffer at an offset that can be computed by the
      routine from coll_patterns/sra_knomial.h: ucc_sra_kn_get_offset.
 */

#define GET_DT(_args, _trank)                                                  \
    (UCC_IS_ROOT(*(_args), _trank))                                            \
        ? (_args)->dst.info.datatype                                           \
        : (_args)->src.info.datatype

#define GET_MT(_args, _trank)                                                  \
    (UCC_IS_ROOT(*(_args), _trank))                                            \
        ? (_args)->dst.info.mem_type                                           \
        : (_args)->src.info.mem_type

#define GET_COUNT(_args, _trank)                                               \
    (UCC_IS_ROOT(*(_args), _trank))                                            \
        ? (_args)->dst.info.count                                              \
        : (_args)->src.info.count

/**
 * Get the buffers for the reduce-scatter and gather tasks from reduce args
 */
static void get_rs_ag_buffers(ucc_tl_ucp_schedule_t *rsg_schedule,
                              ucc_coll_args_t *reduce_args,
                              ucc_rank_t trank,
                              void **rs_rbuf, void **rs_sbuf,
                              void **g_rbuf, void **g_sbuf)
{
    if (UCC_IS_ROOT(*reduce_args, trank)) {
        if (UCC_IS_INPLACE(*reduce_args)) {
            *rs_sbuf = reduce_args->dst.info.buffer;
            *rs_rbuf = rsg_schedule->scratch_mc_header->addr;

            *g_sbuf = rsg_schedule->scratch_mc_header->addr;
            *g_rbuf = reduce_args->dst.info.buffer;
        } else {
            *rs_sbuf = reduce_args->src.info.buffer;
            *rs_rbuf = reduce_args->dst.info.buffer;

            // inplace gather case
            *g_sbuf = reduce_args->dst.info.buffer;
            *g_rbuf = reduce_args->dst.info.buffer;
        }
    } else {
        *rs_sbuf = reduce_args->src.info.buffer;
        *rs_rbuf = rsg_schedule->scratch_mc_header->addr;

        *g_sbuf = rsg_schedule->scratch_mc_header->addr;
        /* non-root gather has no receive buffer but we use rbuf to pass
         * scratch space needed for gather knomial algorithm */
        *g_rbuf = rsg_schedule->scratch_mc_header->addr;
    }
}

static ucc_status_t
ucc_tl_ucp_reduce_srg_knomial_frag_start(ucc_coll_task_t *task)
{
    return ucc_schedule_start(task);
}

static ucc_status_t
ucc_tl_ucp_reduce_srg_knomial_frag_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

static ucc_status_t
ucc_tl_ucp_reduce_srg_knomial_frag_setup(ucc_schedule_pipelined_t *schedule_p,
                                         ucc_schedule_t *frag, int frag_num)
{
    ucc_tl_ucp_schedule_t *rsg_schedule = ucc_derived_of(schedule_p,
                                                         ucc_tl_ucp_schedule_t);
    int                    n_frags      = schedule_p->super.n_tasks;
    ucc_coll_args_t       *args         = &schedule_p->super.super.bargs.args;
    ucc_rank_t             trank        = UCC_TL_TEAM_RANK(TASK_TEAM(&schedule_p->super));
    size_t                 dt_size      = ucc_dt_size(GET_DT(args, trank));
    size_t                 count        = GET_COUNT(args, trank);
    size_t                 frag_count;
    size_t                 offset;
    ucc_coll_args_t       *targs;
    void                  *rs_rbuf, *rs_sbuf, *g_rbuf, *g_sbuf;

    frag_count = ucc_buffer_block_count(count, n_frags, frag_num);
    offset     = ucc_buffer_block_offset(count, n_frags, frag_num);
    get_rs_ag_buffers(rsg_schedule, args, trank,
                      &rs_rbuf, &rs_sbuf, &g_rbuf, &g_sbuf);

    targs = &frag->tasks[0]->bargs.args; /* REDUCE_SCATTER */
    targs->src.info.buffer = PTR_OFFSET(rs_sbuf, offset * dt_size);
    if (UCC_IS_ROOT(*args, trank) && !UCC_IS_INPLACE(*args)) {
        targs->dst.info.buffer = PTR_OFFSET(rs_rbuf, offset * dt_size);
    }
    targs->src.info.count  = frag_count;
    targs->dst.info.count  = frag_count;

    targs = &frag->tasks[1]->bargs.args; /* GATHER */
    if (UCC_IS_ROOT(*args, trank) && !UCC_IS_INPLACE(*args)) {
        targs->src.info.buffer = PTR_OFFSET(g_sbuf, offset * dt_size);
    }
    targs->src.info.count  = frag_count;
    if (UCC_IS_ROOT(*args, trank)) {
        targs->dst.info.buffer = PTR_OFFSET(g_rbuf, offset * dt_size);
    }
    targs->dst.info.count  = frag_count;

    return UCC_OK;
}

static ucc_status_t
ucc_tl_ucp_reduce_srg_knomial_frag_init(ucc_base_coll_args_t *coll_args,
                                        ucc_schedule_pipelined_t *sp,
                                        ucc_base_team_t *team,
                                        ucc_schedule_t **frag_p)
{
    ucc_tl_ucp_team_t     *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t             trank   = UCC_TL_TEAM_RANK(tl_team);
    ucc_tl_ucp_schedule_t *rsg_schedule = ucc_derived_of(sp, ucc_tl_ucp_schedule_t);
    ucc_datatype_t         dt      = GET_DT(&coll_args->args, trank);
    size_t                 dt_size = ucc_dt_size(dt);
    ucc_memory_type_t      mt      = GET_MT(&coll_args->args, trank);
    size_t                 count   = GET_COUNT(&coll_args->args, trank);
    ucc_base_coll_args_t   args    = *coll_args;
    ucc_mrange_uint_t     *p       = &tl_team->cfg.reduce_srg_kn_radix;
    int                    n_frags = sp->super.n_tasks;
    ucc_kn_radix_t         radix, cfg_radix;
    ucc_schedule_t        *schedule;
    ucc_coll_task_t       *g_task, *rs_task;
    ucc_status_t           status;
    ptrdiff_t              scratch_offset;
    void                  *rs_rbuf, *rs_sbuf, *g_rbuf, *g_sbuf;


    scratch_offset = ucc_buffer_block_count(count, n_frags, 0);
    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                     (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    cfg_radix = ucc_tl_ucp_get_radix_from_range(tl_team, count * dt_size,
                                                mt, p, 4);
    radix     = ucc_knomial_pattern_get_min_radix(cfg_radix,
                                                  UCC_TL_TEAM_SIZE(tl_team),
                                                  count);
    get_rs_ag_buffers(rsg_schedule, &coll_args->args, trank,
                      &rs_rbuf, &rs_sbuf, &g_rbuf, &g_sbuf);
    /* 1st step of reduce: knomial reduce_scatter.
    Actual data pointer is set in the setup function */
    args.args.flags             &= ~UCC_COLL_ARGS_FLAG_IN_PLACE;
    args.args.src.info.buffer   = rs_sbuf;
    args.args.src.info.count    = count;
    args.args.src.info.datatype = dt;
    args.args.src.info.mem_type = mt;
    if (!UCC_IS_ROOT(coll_args->args, trank) || UCC_IS_INPLACE(coll_args->args)) {
        args.args.dst.info.buffer   = PTR_OFFSET(rs_rbuf,
            rsg_schedule->reduce_srg_kn.frag_offset * dt_size);
    }
    args.args.dst.info.count    = count;
    args.args.dst.info.datatype = dt;
    args.args.dst.info.mem_type = mt;

    UCC_CHECK_GOTO(ucc_tl_ucp_reduce_scatter_knomial_init_r(&args, team,
                                                            &rs_task, radix),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, rs_task), out, status);
    /* reduce scatter task starts when schedule is started */
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, rs_task,
                                          UCC_EVENT_SCHEDULE_STARTED),
                   out, status);

    /* 2nd step of reduce: knomial gather */
    if (!UCC_IS_ROOT(coll_args->args, trank) || UCC_IS_INPLACE(coll_args->args)) {
    args.args.src.info.buffer   = PTR_OFFSET(g_sbuf,
            rsg_schedule->reduce_srg_kn.frag_offset * dt_size);
    }
    args.args.src.info.count    = count;
    args.args.src.info.datatype = dt;
    args.args.src.info.mem_type = mt;
    if (UCC_IS_ROOT(coll_args->args, trank)) {
        args.args.dst.info.buffer   = PTR_OFFSET(g_rbuf,
            rsg_schedule->reduce_srg_kn.frag_offset * dt_size);
    }
    args.args.dst.info.count    = count;
    args.args.dst.info.datatype = dt;
    args.args.dst.info.mem_type = mt;
    if (UCC_IS_ROOT(coll_args->args, trank) && !UCC_IS_INPLACE(coll_args->args)) {
        args.args.mask |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    UCC_CHECK_GOTO(ucc_tl_ucp_gather_knomial_init_r(&args, team, &g_task, radix),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, g_task), out, status);
    /* gather task starts when reduce scatter task is completed */
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(rs_task, g_task, UCC_EVENT_COMPLETED),
                   out, status);
    rsg_schedule->reduce_srg_kn.frag_offset += scratch_offset;
    schedule->super.finalize = ucc_tl_ucp_reduce_srg_knomial_frag_finalize;
    schedule->super.post     = ucc_tl_ucp_reduce_srg_knomial_frag_start;
    *frag_p                  = schedule;
    return UCC_OK;
out:
    return status;
}

static ucc_status_t
ucc_tl_ucp_reduce_srg_knomial_finalize(ucc_coll_task_t *task)
{
    ucc_tl_ucp_schedule_t *schedule = ucc_derived_of(task,
                                                     ucc_tl_ucp_schedule_t);
    ucc_status_t status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(schedule, "ucp_reduce_srg_kn_done", 0);
    if (schedule->scratch_mc_header) {
        ucc_mc_free(schedule->scratch_mc_header);
    }
    status = ucc_schedule_pipelined_finalize(task);
    ucc_tl_ucp_put_schedule(&schedule->super.super);
    return status;
}

ucc_status_t ucc_tl_ucp_reduce_srg_knomial_start(ucc_coll_task_t *task)
{
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(task, "ucp_reduce_srg_kn_start", 0);
    return ucc_schedule_pipelined_post(task);
}

static void
ucc_tl_ucp_reduce_srg_knomial_get_pipeline_params(ucc_tl_ucp_team_t *team,
                                                  ucc_memory_type_t mtype,
                                                  ucc_pipeline_params_t *pp)
{
    ucc_tl_ucp_lib_config_t *cfg = &team->cfg;
    ucc_mc_attr_t mc_attr;

    if (!ucc_pipeline_params_is_auto(&cfg->reduce_srg_kn_pipeline)) {
        *pp = cfg->reduce_srg_kn_pipeline;
        return;
    }

    if (mtype == UCC_MEMORY_TYPE_CUDA) {
        mc_attr.field_mask = UCC_MC_ATTR_FIELD_FAST_ALLOC_SIZE;
        ucc_mc_get_attr(&mc_attr, UCC_MEMORY_TYPE_CUDA);
        pp->threshold = mc_attr.fast_alloc_size;
        pp->n_frags   = 2;
        pp->order     = UCC_PIPELINE_PARALLEL;
        pp->pdepth    = 2;
        pp->frag_size = mc_attr.fast_alloc_size / pp->pdepth;
    } else {
        pp->threshold = SIZE_MAX;
        pp->n_frags   = 0;
        pp->pdepth    = 1;
        pp->order     = UCC_PIPELINE_PARALLEL;
        pp->frag_size = 0;
    }
}

ucc_status_t ucc_tl_ucp_reduce_srg_knomial_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t *team,
                                                ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t     *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_coll_args_t       *args    = &coll_args->args;
    ucc_rank_t             trank   = UCC_TL_TEAM_RANK(tl_team);
    ucc_memory_type_t      mt      = GET_MT(args, trank);
    size_t                 count   = GET_COUNT(args, trank);
    size_t                 dt_size = ucc_dt_size(GET_DT(args, trank));
    int                    n_frags, pipeline_depth;
    ucc_tl_ucp_schedule_t *schedule;
    ucc_status_t           st;
    ucc_base_coll_args_t   bargs;
    size_t                 max_frag_count;
    ucc_pipeline_params_t  pipeline_params;

    st  = ucc_tl_ucp_get_schedule(tl_team, coll_args, &schedule);
    if (ucc_unlikely(UCC_OK != st)) {
        goto err_out;
    }

    schedule->scratch_mc_header = NULL;
    schedule->reduce_srg_kn.frag_offset = 0;

    bargs = *coll_args;
    max_frag_count = (bargs.mask & UCC_BASE_CARGS_MAX_FRAG_COUNT) ?
                      bargs.max_frag_count: count;
    ucc_tl_ucp_reduce_srg_knomial_get_pipeline_params(tl_team, mt,
                                                      &pipeline_params);
    ucc_pipeline_nfrags_pdepth(&pipeline_params, max_frag_count * dt_size,
                               &n_frags, &pipeline_depth);
    bargs.max_frag_count = ucc_buffer_block_count(max_frag_count, n_frags, 0);
    if (n_frags > 1) {
        bargs.mask           |= UCC_BASE_CARGS_MAX_FRAG_COUNT;
    }

    if (!UCC_IS_ROOT(*args, trank) || UCC_IS_INPLACE(*args)) {
        st = ucc_mc_alloc(&schedule->scratch_mc_header, bargs.max_frag_count  * dt_size * pipeline_depth, mt);
        if (ucc_unlikely(UCC_OK != st)) {
            tl_error(team->context->lib, "failed to alloc scratch memory");
            goto err_free_schedule;
        }
    }

    st = ucc_schedule_pipelined_init(&bargs, team,
                                     ucc_tl_ucp_reduce_srg_knomial_frag_init,
                                     ucc_tl_ucp_reduce_srg_knomial_frag_setup,
                                     pipeline_depth, n_frags,
                                     pipeline_params.order,
                                     &schedule->super);
    if (ucc_unlikely(UCC_OK != st)) {
        tl_error(team->context->lib, "failed to init pipelined schedule");
        goto err_free_scratch;
    }

    schedule->super.super.super.finalize = ucc_tl_ucp_reduce_srg_knomial_finalize;
    schedule->super.super.super.post     = ucc_tl_ucp_reduce_srg_knomial_start;

    *task_h = &schedule->super.super.super;
    return UCC_OK;

err_free_scratch:
    ucc_mc_free(schedule->scratch_mc_header);
err_free_schedule:
    ucc_tl_ucp_put_schedule(&schedule->super.super);
err_out:
    return st;
}
