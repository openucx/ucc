/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "alltoallv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "tl_ucp_sendrecv.h"
#include "schedule/ucc_schedule.h"
#include "tl_ucp_coll.h"

ucc_status_t ucc_tl_ucp_alltoallv_onesided_data_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t     *task     = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t     *team     = TASK_TEAM(task);
    ucc_tl_ucp_schedule_t *sched    =
        ucc_derived_of(task->super.schedule, ucc_tl_ucp_schedule_t);
    ptrdiff_t          src      = (ptrdiff_t)TASK_ARGS(task).src.info_v.buffer;
    ptrdiff_t          dest     = (ptrdiff_t)TASK_ARGS(task).dst.info_v.buffer;
    ucc_rank_t         grank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize    = UCC_TL_TEAM_SIZE(team);
    long              *pSync    = TASK_ARGS(task).global_work_buffer;
    ucc_aint_t        *s_disp   = TASK_ARGS(task).src.info_v.displacements;
    ucc_count_t       *s_counts = TASK_ARGS(task).src.info_v.counts;
    size_t             sdt_size = ucc_dt_size(TASK_ARGS(task).src.info_v.datatype);
    size_t             rdt_size = ucc_dt_size(TASK_ARGS(task).dst.info_v.datatype);
    ucc_mem_map_mem_h  src_memh = TASK_ARGS(task).src_memh.local_memh;
    ucc_mem_map_mem_h *dst_memh = TASK_ARGS(task).dst_memh.global_memh;
    ucc_rank_t         peer;
    size_t             sd_disp, dd_disp, data_size;
    ucc_aint_t        *recv_displs;
    ucc_aint_t         recv_displ;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    /* Get pointer to received displacements from scratch buffer */
    recv_displs = (ucc_aint_t *)sched->scratch_mc_header->addr;

    /* perform a put to each member peer using the received displacement
     * information from the alltoall exchange phase. */
    for (peer = (grank + 1) % gsize; task->onesided.put_posted < gsize;
         peer = (peer + 1) % gsize) {
        /* Use source displacement and count (what we send to peer) */
        sd_disp = ucc_coll_args_get_displacement(&TASK_ARGS(task), s_disp, peer) * sdt_size;
        data_size = ucc_coll_args_get_count(&TASK_ARGS(task), s_counts, peer) * sdt_size;

        /* Use received displacement from peer (where peer told us to put data) */
        recv_displ = ucc_coll_args_get_displacement(&TASK_ARGS(task), recv_displs, peer);
        dd_disp = recv_displ * rdt_size;

        UCPCHECK_GOTO(ucc_tl_ucp_put_nb(PTR_OFFSET(src, sd_disp),
                                        PTR_OFFSET(dest, dd_disp),
                                        data_size, peer, src_memh,
                                        dst_memh, team, task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_atomic_inc(pSync, peer,
                                            dst_memh, team),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_ep_flush(peer, team, task), task, out);
    }
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return task->super.status;
}

void ucc_tl_ucp_alltoallv_onesided_data_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    long              *pSync = TASK_ARGS(task).global_work_buffer;

    if (ucc_tl_ucp_test_onesided(task, gsize) == UCC_INPROGRESS) {
        return;
    }

    tl_debug(UCC_TL_TEAM_LIB(team),
             "onesided data transfer completed successfully");
    pSync[0]           = 0;
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoallv_onesided_sched_start(ucc_coll_task_t *task)
{
    tl_debug(UCC_TASK_LIB(task),
             "starting onesided alltoallv schedule (will run displacement exchange then data transfer)");
    return ucc_schedule_start(task);
}

ucc_status_t ucc_tl_ucp_alltoallv_onesided_sched_finalize(ucc_coll_task_t *task)
{
    ucc_tl_ucp_schedule_t *sched = ucc_derived_of(task, ucc_tl_ucp_schedule_t);
    ucc_status_t           status;

    status = ucc_schedule_finalize(task);
    if (sched->scratch_mc_header) {
        ucc_mc_free(sched->scratch_mc_header);
    }
    ucc_tl_ucp_put_schedule(&sched->super.super);
    return status;
}

ucc_status_t ucc_tl_ucp_alltoallv_onesided_data_finalize(ucc_coll_task_t *coll_task)
{
    ucc_status_t status;

    status = ucc_tl_ucp_coll_finalize(coll_task);
    if (ucc_unlikely(UCC_OK != status)) {
        tl_error(UCC_TASK_LIB(coll_task), "failed to finalize collective");
    }
    return status;
}

ucc_status_t ucc_tl_ucp_alltoallv_onesided_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t     *tl_team   = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_rank_t             gsize     = UCC_TL_TEAM_SIZE(tl_team);
    ucc_tl_ucp_schedule_t *sched     = NULL;
    ucc_schedule_t        *schedule  = NULL;
    ucc_coll_task_t       *a2a_task  = NULL;
    ucc_tl_ucp_task_t     *data_task = NULL;
    ucc_datatype_t         dt;
    ucc_base_coll_args_t   bargs;
    ucc_status_t           status;
    size_t                 scratch_size;
    int                    displ_size;

    ALLTOALLV_TASK_CHECK(coll_args->args, tl_team);
    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "global work buffer not provided nor associated with team");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    if (coll_args->args.mask & UCC_COLL_ARGS_FIELD_FLAGS) {
        if (!(coll_args->args.flags & UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "non memory mapped buffers are not supported");
            status = UCC_ERR_NOT_SUPPORTED;
            goto out;
        }
    }
    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_SRC_MEMH)) {
        coll_args->args.src_memh.global_memh = NULL;
    }
    if (!(coll_args->args.mask & UCC_COLL_ARGS_FIELD_MEM_MAP_DST_MEMH)) {
        coll_args->args.dst_memh.global_memh = NULL;
    }

    /* Determine size based on displacement types */
    if (UCC_COLL_ARGS_DISPL64(&coll_args->args)) {
        displ_size = sizeof(uint64_t);
        dt = UCC_DT_INT64;
    } else {
        displ_size = sizeof(uint32_t);
        dt = UCC_DT_INT32;
    }

    /* Allocate schedule with scratch memory for received displacements */
    status = ucc_tl_ucp_get_schedule(tl_team, coll_args, &sched);
    if (ucc_unlikely(status != UCC_OK)) {
        goto out;
    }
    schedule = &sched->super.super;
    ucc_schedule_init(schedule, coll_args, team);
    schedule->super.post     = ucc_tl_ucp_alltoallv_onesided_sched_start;
    schedule->super.progress = NULL;
    schedule->super.finalize = ucc_tl_ucp_alltoallv_onesided_sched_finalize;

    /* Allocate scratch buffer for received displacements */
    scratch_size = gsize * displ_size;
    status = ucc_mc_alloc(&sched->scratch_mc_header, scratch_size,
                          UCC_MEMORY_TYPE_HOST);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failed to allocate scratch buffer for displ exchange");
        goto free_schedule;
    }

    /* exchange displacements with alltoall */
    memset(&bargs, 0, sizeof(bargs));
    bargs.args.coll_type         = UCC_COLL_TYPE_ALLTOALL;
    bargs.args.mask              = 0;
    bargs.args.flags             = 0;
    bargs.args.src.info.buffer   = (void *)coll_args->args.dst.info_v.displacements;
    bargs.args.src.info.count    = gsize;
    bargs.args.src.info.datatype = dt;
    bargs.args.src.info.mem_type = UCC_MEMORY_TYPE_HOST;
    bargs.args.dst.info.buffer   = (void *)sched->scratch_mc_header->addr;
    bargs.args.dst.info.count    = gsize;
    bargs.args.dst.info.datatype = dt;
    bargs.args.dst.info.mem_type = UCC_MEMORY_TYPE_HOST;

    bargs.team = team->params.team;
    status     = ucc_tl_ucp_coll_init(&bargs, team, &a2a_task);
    if (ucc_unlikely(status != UCC_OK)) {
        tl_error(UCC_TL_TEAM_LIB(tl_team),
                 "failed to init displs exchange alltoall");
        goto free_schedule;
    }

    tl_debug(UCC_TL_TEAM_LIB(tl_team),
             "initialized displacement exchange alltoall as Task 0");

    data_task = ucc_tl_ucp_init_task(coll_args, team);
    if (!data_task) {
        status = UCC_ERR_NO_MEMORY;
        goto free_tasks;
    }
    data_task->super.post     = ucc_tl_ucp_alltoallv_onesided_data_start;
    data_task->super.progress = ucc_tl_ucp_alltoallv_onesided_data_progress;
    data_task->super.finalize = ucc_tl_ucp_alltoallv_onesided_data_finalize;

    /* Add tasks to schedule and set dependencies */
    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, a2a_task),
                   free_tasks, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(&schedule->super, a2a_task,
                                          UCC_EVENT_SCHEDULE_STARTED),
                   free_tasks, status);

    UCC_CHECK_GOTO(ucc_schedule_add_task(schedule, &data_task->super),
                   free_tasks, status);
    UCC_CHECK_GOTO(ucc_task_subscribe_dep(a2a_task, &data_task->super,
                                          UCC_EVENT_COMPLETED),
                   free_tasks, status);

    *task_h = &schedule->super;
    return UCC_OK;

free_tasks:
    if (data_task) {
        ucc_tl_ucp_put_task(data_task);
    }
    if (a2a_task && a2a_task->finalize) {
        a2a_task->finalize(a2a_task);
    }
free_schedule:
    if (sched->scratch_mc_header) {
        ucc_mc_free(sched->scratch_mc_header);
    }
    ucc_tl_ucp_put_schedule(&sched->super.super);
out:
    return status;
}
