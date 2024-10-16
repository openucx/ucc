/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
      (for radix=2 this means the team size is not power of 2) then there will be
      "extra" ranks which don't participate in the main exchange loop. They
      will send the data to their "proxy" ranks in the beginning and then wait
      for the response with the final data.
   5. The knomial reduce-scatter and gather primitives can be used separately.
      However, if they are used together as part of SRG reduce one has to
      provide the same radix for both routines.
   6. After the completion of reduce-scatter phase the local result (at non EXTRA
      ranks) will be located in dst buffer at offset the can be commputed by the
      routine from coll_patterns/sra_knomial.h: ucc_sra_kn_get_offset.
 */

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
    int              n_frags = schedule_p->super.n_tasks;
    ucc_coll_args_t *args    = &schedule_p->super.super.bargs.args;
    size_t           dt_size;
    size_t           count;
    size_t           frag_count;
    size_t           offset;
    ucc_coll_args_t *targs;
    ucc_rank_t rank;
    ucc_tl_ucp_team_t *team;

    team = TASK_TEAM(&schedule_p->super);
    rank = UCC_TL_TEAM_RANK(team);
    if (UCC_IS_ROOT(*args, rank)) {
        count = args->dst.info.count;
        dt_size = ucc_dt_size(args->dst.info.datatype);
    } else {
        count = args->src.info.count;
        dt_size = ucc_dt_size(args->src.info.datatype);
    }
    frag_count = ucc_buffer_block_count(count, n_frags, frag_num);
    offset     = ucc_buffer_block_offset(count, n_frags, frag_num);

    targs = &frag->tasks[0]->bargs.args; /* REDUCE_SCATTER */
    targs->src.info.buffer = PTR_OFFSET(targs->src.info.buffer, offset * dt_size);
    targs->src.info.count  = frag_count;
    targs->dst.info.buffer = PTR_OFFSET(targs->dst.info.buffer, offset * dt_size);
    targs->dst.info.count  = frag_count;

    targs = &frag->tasks[1]->bargs.args; /* GATHER */
    targs->src.info.buffer = PTR_OFFSET(targs->src.info.buffer, offset * dt_size);;
    targs->src.info.count  = 0;
    targs->dst.info.buffer = PTR_OFFSET(targs->dst.info.buffer, offset * dt_size);
    targs->dst.info.count  = frag_count;

    return UCC_OK;
}

static ucc_status_t
ucc_tl_ucp_reduce_srg_knomial_frag_init(ucc_base_coll_args_t *coll_args,
                                        ucc_schedule_pipelined_t *sp,
                                        ucc_base_team_t *team,
                                        ucc_schedule_t **frag_p)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t args     = *coll_args;
    ucc_mrange_uint_t   *p        = &tl_team->cfg.reduce_srg_kn_radix;
    ucc_rank_t           trank    = UCC_TL_TEAM_RANK(tl_team);
    ucc_schedule_t      *schedule;
    ucc_coll_task_t     *task, *rs_task;
    ucc_status_t         status;
    ucc_kn_radix_t       radix, cfg_radix;
    size_t               count;
    ucc_datatype_t       dt;
    void                 *rs_rbuf, *rs_sbuf;
    ucc_tl_ucp_schedule_t *rsg_schedule;
    ucc_memory_type_t mt;

    rsg_schedule = ucc_derived_of(sp, ucc_tl_ucp_schedule_t);
    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                     (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    if (UCC_IS_ROOT(coll_args->args, trank)) {
        dt      = coll_args->args.dst.info.datatype;
        mt      = coll_args->args.dst.info.mem_type;
        if (UCC_IS_INPLACE(coll_args->args)) {
            rs_rbuf = rsg_schedule->scratch_mc_header->addr;
            rs_sbuf = coll_args->args.dst.info.buffer;
        } else {
            rs_rbuf = coll_args->args.dst.info.buffer;
            rs_sbuf = coll_args->args.src.info.buffer;
        }
        count = coll_args->args.dst.info.count;
    } else {
        dt      = coll_args->args.src.info.datatype;
        mt      = coll_args->args.src.info.mem_type;
        rs_rbuf = rsg_schedule->scratch_mc_header->addr;
        rs_sbuf = coll_args->args.src.info.buffer;
        count   = coll_args->args.src.info.count;
    }

    if (coll_args->mask & UCC_BASE_CARGS_MAX_FRAG_COUNT) {
        count = coll_args->max_frag_count;
    }

    args.args.flags             &= ~UCC_COLL_ARGS_FLAG_IN_PLACE;
    args.args.dst.info.buffer   = rs_rbuf;
    args.args.dst.info.count    = count;
    args.args.dst.info.datatype = dt;
    args.args.dst.info.mem_type = mt;
    args.args.src.info.buffer   = rs_sbuf;
    args.args.src.info.count    = count;
    args.args.src.info.datatype = dt;
    args.args.src.info.mem_type = mt;

    cfg_radix = ucc_tl_ucp_get_radix_from_range(tl_team,
                                                count * ucc_dt_size(dt),
                                                mt, p, 4);
    radix     = ucc_knomial_pattern_get_min_radix(cfg_radix,
                                                  UCC_TL_TEAM_SIZE(tl_team),
                                                  count);

    /* 1st step of reduce: knomial reduce_scatter */
    UCC_CHECK_GOTO(ucc_tl_ucp_reduce_scatter_knomial_init_r(&args, team, &task,
                                                            radix),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, task), out, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, task,
                                          UCC_EVENT_SCHEDULE_STARTED),
                   out, status);
    rs_task = task;

    /* 2nd step of reduce: knomial gather */
    args.args.src.info.buffer = rs_rbuf;
    if (UCC_IS_ROOT(coll_args->args, trank)) {
        if (UCC_IS_INPLACE (coll_args->args)) {
            args.args.dst.info.buffer = rs_sbuf;
            args.args.src.info.buffer = rs_rbuf;
        } else {
            args.args.dst.info.buffer = rs_rbuf;
            args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
            args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;

        }
    } else {
        args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
        args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    }

    UCC_CHECK_GOTO(ucc_tl_ucp_gather_knomial_init_r(&args, team, &task, radix),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, task), out, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(rs_task, task, UCC_EVENT_COMPLETED),
                   out, status);
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
        pp->frag_size = mc_attr.fast_alloc_size;
        pp->order     = UCC_PIPELINE_PARALLEL;
        pp->pdepth    = 2;
    } else {
        pp->threshold = SIZE_MAX;
        pp->n_frags   = 0;
        pp->frag_size = 0;
        pp->pdepth    = 1;
        pp->order     = UCC_PIPELINE_PARALLEL;
    }
}

ucc_status_t ucc_tl_ucp_reduce_srg_knomial_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t *team,
                                                ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t     *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_coll_args_t       *args    = &coll_args->args;
    ucc_rank_t             trank   = UCC_TL_TEAM_RANK(tl_team);
    int                    n_frags, pipeline_depth;
    ucc_tl_ucp_schedule_t *schedule;
    ucc_status_t           st;
    ucc_base_coll_args_t   bargs;
    size_t                 max_frag_count, dt_size, count;
    ucc_pipeline_params_t  pipeline_params;
    ucc_datatype_t         dt;
    ucc_memory_type_t      mt;

    st  = ucc_tl_ucp_get_schedule(tl_team, coll_args, &schedule);
    if (ucc_unlikely(UCC_OK != st)) {
        goto err_out;
    }

    schedule->scratch_mc_header = NULL;
    if (UCC_IS_ROOT(*args, trank)) {
        count   = args->dst.info.count;
        dt      = args->dst.info.datatype;
        mt      = args->dst.info.mem_type;
        dt_size = ucc_dt_size(dt);
    } else  {
        count   = args->src.info.count;
        dt      = args->src.info.datatype;
        mt      = args->src.info.mem_type;
        dt_size = ucc_dt_size(dt);
    }

    if (!UCC_IS_ROOT(*args, trank) || UCC_IS_INPLACE(*args)) {
        st = ucc_mc_alloc(&schedule->scratch_mc_header, count * dt_size, mt);
        if (ucc_unlikely(UCC_OK != st)) {
            tl_error(team->context->lib, "failed to alloc scratch memory");
            goto err_free_schedule;
        }
    }

    bargs = *coll_args;
    max_frag_count = (bargs.mask & UCC_BASE_CARGS_MAX_FRAG_COUNT) ?
                      bargs.max_frag_count: count;
    ucc_tl_ucp_reduce_srg_knomial_get_pipeline_params(tl_team, mt,
                                                      &pipeline_params);
    ucc_pipeline_nfrags_pdepth(&pipeline_params, max_frag_count * dt_size,
                               &n_frags, &pipeline_depth);
    if (n_frags > 1) {
        bargs.mask           |= UCC_BASE_CARGS_MAX_FRAG_COUNT;
        bargs.max_frag_count = ucc_buffer_block_count(max_frag_count, n_frags, 0);
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
