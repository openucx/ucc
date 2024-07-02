#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include "../gather/gather.h"
#include "../bcast/bcast.h"

ucc_status_t ucc_tl_ucp_allgather_gb_knomial_start(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t  *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_coll_args_t *args     = &schedule->super.bargs.args;
    ucc_coll_task_t *gather_task, *bcast_task;

    gather_task = schedule->tasks[0];
    gather_task->bargs.args.src.info.buffer = args->src.info.buffer;
    gather_task->bargs.args.dst.info.buffer = args->dst.info.buffer;
    gather_task->bargs.args.src.info.count  = args->src.info.count;
    gather_task->bargs.args.dst.info.count  = args->dst.info.count;

    bcast_task = schedule->tasks[1];
    bcast_task->bargs.args.src.info.buffer = args->dst.info.buffer;
    bcast_task->bargs.args.src.info.count  = args->dst.info.count;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_gb_knomial_start", 0);
    return ucc_schedule_start(coll_task);
}

ucc_status_t ucc_tl_ucp_allgather_gb_knomial_finalize(ucc_coll_task_t *coll_task)
{
    ucc_schedule_t *schedule = ucc_derived_of(coll_task, ucc_schedule_t);
    ucc_status_t    status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(schedule, "ucp_allgather_gb_knomial_done", 0);
    status = ucc_schedule_finalize(coll_task);
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}

ucc_status_t ucc_tl_ucp_allgather_gb_knomial_init(ucc_base_coll_args_t *coll_args,
                                                  ucc_base_team_t *team,
                                                  ucc_coll_task_t **task_h)
{
    ucc_tl_ucp_team_t   *tl_team  = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_base_coll_args_t args     = *coll_args;
    ucc_schedule_t      *schedule;
    ucc_coll_task_t     *gather_task, *bcast_task;
    ucc_status_t         status;

    
    status = ucc_tl_ucp_get_schedule(tl_team, coll_args,
                                     (ucc_tl_ucp_schedule_t **)&schedule);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }

    
    UCC_CHECK_GOTO(ucc_tl_ucp_gather_knomial_init(&args, team, &gather_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, gather_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_event_manager_subscribe(&schedule->super,
                                               UCC_EVENT_SCHEDULE_STARTED,
                                               gather_task,
                                               ucc_task_start_handler),
                   out, status);

    UCC_CHECK_GOTO(ucc_tl_ucp_bcast_sag_knomial_init(&args, team, &bcast_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, bcast_task),
                   out, status);
    UCC_CHECK_GOTO(ucc_event_manager_subscribe(gather_task, UCC_EVENT_COMPLETED,
                                               bcast_task,
                                               ucc_task_start_handler),
                   out, status);

    schedule->super.post = ucc_tl_ucp_allgather_gb_knomial_start;
    schedule->super.progress = NULL;
    schedule->super.finalize =  ucc_tl_ucp_allgather_gb_knomial_finalize;
    *task_h = &schedule->super;

    return UCC_OK;

out:
    ucc_tl_ucp_put_schedule(schedule);
    return status;
}