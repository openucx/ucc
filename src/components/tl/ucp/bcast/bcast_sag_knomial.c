/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "bcast.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "coll_patterns/sra_knomial.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include "../scatter/scatter.h"
#include "../allgather/allgather.h"

/* SAG - scatter-allgather knomial algorithm
   1. The algorithm performs collective bcast operation for large messages
      as a sequence of K-nomial Scatter followed by K-nomial
      (with the same radix K) allgather.
   2. In essence this is an extension of the Bi-nomial SRA algorithm algorithm
      proposed by Rabenseifner2004 (https://doi.org/10.1007/978-3-540-24685-5_1).
      The extension adds the support for arbitrary radix.
   3. The algorithm targets Large message sizes (ie. optimized for max bandwidth).
   4. If number of ranks in the team can not form a full radix subtree
      (for radix=2 this means the team size is not power of 2) then there will be
      "extra" ranks which don't participate in the main exchange loop. They
      will wait to recieve the final data from their "proxy" ranks at the end of
      exchange loop of all other ranks.
   5. The knomial scatter and allgather primitives can be used separately.
      However, if they are used together as part of SAG bcast one has to
      provide the same radix for both routines.
   6. After the completion of scatter phase the local result (at non EXTRA
      ranks) will be located in dst buffer at offset the can be commputed by the
      routine from coll_patterns/sra_knomial.h: ucc_sra_kn_get_offset.
 */
ucc_status_t ucc_tl_ucp_bcast_sag_knomial_start(ucc_coll_task_t *coll_task)
{
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_bcast_sag_kn_start", 0);
    return ucc_schedule_start(coll_task);
}

ucc_status_t
ucc_tl_ucp_bcast_sag_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_status_t    status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(schedule, "ucp_bcast_sag_kn_done", 0);
    status = ucc_schedule_finalize(coll_task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t
ucc_tl_ucp_bcast_sag_knomial_init(ucc_base_coll_args_t *coll_args,
                                      ucc_base_team_t      *team,
                                      ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    size_t               count    = coll_args->args.src.info.count;
    ucc_base_coll_args_t args     = *coll_args;
    ucc_schedule_t      *schedule;
    ucc_coll_task_t     *task, *rs_task;
    ucc_status_t         status;
    ucc_kn_radix_t       radix, cfg_radix;

    if (UCC_COLL_ARGS_ACTIVE_SET(&coll_args->args)) {
        /* ActiveSets currently are only supported with KN alg */
        return ucc_tl_ucp_bcast_knomial_init(coll_args, team, task_h);
    }

    cfg_radix = UCC_TL_UCP_TEAM_LIB(tl_team)->cfg.bcast_sag_kn_radix;
    radix = ucc_knomial_pattern_get_min_radix(cfg_radix,
                                              UCC_TL_TEAM_SIZE(tl_team), count);
    schedule = &ucc_tl_ucp_get_schedule(tl_team, coll_args)->super.super;
    if (ucc_unlikely(!schedule)) {
        return UCC_ERR_NO_MEMORY;
    }
    /* 1st step of bcast: knomial scatter */
    args.args.dst.info.buffer   = args.args.src.info.buffer;
    args.args.dst.info.mem_type = args.args.src.info.mem_type;
    args.args.dst.info.datatype = args.args.src.info.datatype;
    args.args.dst.info.count    = args.args.src.info.count;
    status = ucc_tl_ucp_scatter_knomial_init_r(&args, team, &task, radix);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failed to init scatter_knomial task");
        goto out;
    }
    ucc_schedule_add_task(schedule, task);
    ucc_event_manager_subscribe(&schedule->super.em, UCC_EVENT_SCHEDULE_STARTED,
                                task, ucc_task_start_handler);
    rs_task = task;

    /* 2nd step of bcast: knomial allgather. 2nd task subscribes
     to completion event of scatter task. */
    args.args.mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
    args.args.flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
    status = ucc_tl_ucp_allgather_knomial_init_r(&args, team, &task, radix);
    if (UCC_OK != status) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failed to init allgather_knomial task");
        goto out;
    }

    ucc_schedule_add_task(schedule, task);
    ucc_event_manager_subscribe(&rs_task->em, UCC_EVENT_COMPLETED, task,
                                ucc_task_start_handler);

    schedule->super.post           = ucc_tl_ucp_bcast_sag_knomial_start;
    schedule->super.progress       = NULL;
    schedule->super.finalize       = ucc_tl_ucp_bcast_sag_knomial_finalize;
    schedule->super.triggered_post = ucc_triggered_post;
    *task_h                        = &schedule->super;
    return UCC_OK;
out:
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}
