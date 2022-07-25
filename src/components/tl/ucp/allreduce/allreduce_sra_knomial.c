/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "allreduce.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "../reduce_scatter/reduce_scatter.h"
#include "../allgather/allgather.h"

/* SRA - scatter-reduce-allgather knomial algorithm
   1. The algorithm performs collective allreduce operation as a sequence of
      K-nomial Reduce-Scatter followed by K-nomial (with the same radix K)
      allgather.
   2. In essence this is an extension of the Bi-nomial SRA algorithm algorithm
      proposed by Rabenseifner2004 (https://doi.org/10.1007/978-3-540-24685-5_1).
      The extension adds the support for arbitrary radix.
   3. The algorithm targets Large message sizes (ie. optimized for max bandwidth).
   4. If number of ranks in the team can not form a full radix subtree
      (for radix=2 this means the team size is not power of 2) then there will be
      "extra" ranks which don't participate in the main exchange loop. They
      will send the data to their "proxy" ranks in the beginning and then wait
      for the response with the final data.
   5. The knomial reduce-scatter and allgather primitives can be used separately.
      However, if they are used together as part of SRA allreduce one has to
      provide the same radix for both routines.
   6. If the allreduce is INPLACE or if a rank serves as a PROXY then the algorithm
      requires allocation of a scratch buffer of the size equal to input buffer.
   7. After the completion of reduce-scatter phase the local result (at non EXTRA
      ranks) will be located in dst buffer at offset the can be commputed by the
      routine from coll_patterns/sra_knomial.h: ucc_sra_kn_get_offset.
 */
static ucc_status_t
ucc_tl_ucp_allreduce_sra_knomial_frag_start(ucc_coll_task_t *task)
{
    return ucc_schedule_start(task);
}

static ucc_status_t
ucc_tl_ucp_allreduce_sra_knomial_frag_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t    status;

    status = ucc_schedule_finalize(task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

static ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_frag_setup(
    ucc_schedule_pipelined_t *schedule_p, ucc_schedule_t *frag, int frag_num)
{
    ucc_coll_args_t *args    = &schedule_p->super.super.bargs.args;
    ucc_datatype_t   dt      = args->dst.info.datatype;
    size_t           dt_size = ucc_dt_size(dt);
    ucc_coll_args_t *targs;
    int              n_frags    = schedule_p->super.n_tasks;
    size_t           frag_count = args->dst.info.count / n_frags;
    size_t           left       = args->dst.info.count % n_frags;
    size_t           offset     = frag_num * frag_count + left;

    if (frag_num < left) {
        frag_count++;
        offset -= left - frag_num;
    }

    targs = &frag->tasks[0]->bargs.args; //REDUCE_SCATTER
    targs->src.info.buffer =
        PTR_OFFSET(args->src.info.buffer, offset * dt_size);
    targs->dst.info.buffer =
        PTR_OFFSET(args->dst.info.buffer, offset * dt_size);
    targs->src.info.count = frag_count;
    targs->dst.info.count = frag_count;

    targs                  = &frag->tasks[1]->bargs.args; //ALLGATHER
    targs->src.info.buffer = NULL;
    targs->dst.info.buffer =
        PTR_OFFSET(args->dst.info.buffer, offset * dt_size);
    targs->src.info.count = 0;
    targs->dst.info.count = frag_count;

    return UCC_OK;
}

static ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_frag_init(
    ucc_base_coll_args_t     *coll_args,
    ucc_schedule_pipelined_t *sp, //NOLINT
    ucc_base_team_t *team, ucc_schedule_t **frag_p)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    size_t               count    = coll_args->args.dst.info.count;
    ucc_base_coll_args_t args     = *coll_args;
    ucc_schedule_t      *schedule =
        &ucc_tl_ucp_get_schedule(tl_team, coll_args)->super.super;
    ucc_coll_task_t     *task, *rs_task;
    ucc_status_t         status;
    ucc_kn_radix_t       radix, cfg_radix;

    cfg_radix = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.allreduce_sra_kn_radix;
    radix = ucc_knomial_pattern_get_min_radix(cfg_radix,
                                              UCC_TL_TEAM_SIZE(tl_team), count);

    /* 1st step of allreduce: knomial reduce_scatter */
    status = ucc_tl_ucp_reduce_scatter_knomial_init_r(&args, team, &task, radix);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failed to init reduce_scatter_knomial task");
        goto out;
    }
    ucc_schedule_add_task(schedule, task);
    ucc_task_subscribe_dep(&schedule->super, task, UCC_EVENT_SCHEDULE_STARTED);
    rs_task = task;
    /* 2nd step of allreduce: knomial allgather. 2nd task subscribes
     to completion event of reduce_scatter task. */
    args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
    args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    status = ucc_tl_ucp_allgather_knomial_init_r(&args, team, &task, radix);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failed to init allgather_knomial task");
        goto out;
    }
    ucc_schedule_add_task(schedule, task);
    ucc_task_subscribe_dep(rs_task, task, UCC_EVENT_COMPLETED);
    schedule->super.finalize = ucc_tl_ucp_allreduce_sra_knomial_frag_finalize;
    schedule->super.post     = ucc_tl_ucp_allreduce_sra_knomial_frag_start;
    *frag_p                  = schedule;
    return UCC_OK;
out:
    return status;
}

static inline void get_sra_n_frags(ucc_base_coll_args_t *coll_args,
                                   ucc_tl_ucp_team_t *team, int *n_frags,
                                   int *pipeline_depth)
{
    //TODO make selection mem_type - specific
    ucc_tl_ucp_lib_config_t *cfg     = &UCC_TL_UCP_TEAM_LIB(team)->cfg;
    size_t                   msgsize = coll_args->args.dst.info.count *
                     ucc_dt_size(coll_args->args.dst.info.datatype);
    int min_num_frags;

    *n_frags = 1;
    if (msgsize > cfg->allreduce_sra_kn_frag_thresh) {
        min_num_frags = ucc_div_round_up(msgsize,
                                             cfg->allreduce_sra_kn_frag_size);
        *n_frags = ucc_max(min_num_frags, cfg->allreduce_sra_kn_n_frags);
    }
    *pipeline_depth = ucc_min(*n_frags, cfg->allreduce_sra_kn_pipeline_depth);
}

static ucc_status_t
ucc_tl_ucp_allreduce_sra_knomial_finalize(ucc_coll_task_t *task)
{
    ucc_schedule_t *schedule = ucc_derived_of(task, ucc_schedule_t);
    ucc_status_t status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(schedule, "ucp_allreduce_sra_kn_done", 0);
    status = ucc_schedule_pipelined_finalize(task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t ucc_tl_ucp_allreduce_sra_knomial_start(ucc_coll_task_t *task)
{
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(task, "ucp_allreduce_sra_kn_start", 0);
    return ucc_schedule_pipelined_post(task);
}

ucc_status_t
ucc_tl_ucp_allreduce_sra_knomial_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t      *team,
                                      ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t        *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_lib_config_t  *cfg     = &UCC_TL_UCP_TEAM_LIB(tl_team)->cfg;
    int                       n_frags, pipeline_depth;
    ucc_schedule_pipelined_t *schedule_p =
        &ucc_tl_ucp_get_schedule(tl_team, NULL)->super;
    ucc_status_t status;

    if (!schedule_p) {
        tl_error(team->context->lib, "failed to allocate pipelined schedule");
        return UCC_ERR_NO_MEMORY;
    }
    get_sra_n_frags(coll_args, tl_team, &n_frags, &pipeline_depth);
    status = ucc_schedule_pipelined_init(
        coll_args, team, ucc_tl_ucp_allreduce_sra_knomial_frag_init,
        ucc_tl_ucp_allreduce_sra_knomial_frag_setup, pipeline_depth, n_frags,
        cfg->allreduce_sra_kn_seq, schedule_p);
    if (UCC_OK != status) {
        tl_error(team->context->lib, "failed to init pipelined schedule");
        ucc_tl_ucp_put_schedule(&schedule_p->super);
        return status;
    }
    schedule_p->super.super.finalize =
        ucc_tl_ucp_allreduce_sra_knomial_finalize;
    schedule_p->super.super.triggered_post = ucc_triggered_post;
    schedule_p->super.super.post = ucc_tl_ucp_allreduce_sra_knomial_start;
    *task_h                                = &schedule_p->super.super;
    return UCC_OK;
}
