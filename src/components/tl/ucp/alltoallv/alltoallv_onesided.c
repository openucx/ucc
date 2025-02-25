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

ucc_status_t ucc_tl_ucp_alltoallv_onesided_start(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ptrdiff_t          src      = (ptrdiff_t)TASK_ARGS(task).src.info_v.buffer;
    ptrdiff_t          dest     = (ptrdiff_t)TASK_ARGS(task).dst.info_v.buffer;
    ucc_rank_t         grank    = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         gsize    = UCC_TL_TEAM_SIZE(team);
    long              *pSync    = TASK_ARGS(task).global_work_buffer;
    ucc_aint_t        *s_disp   = TASK_ARGS(task).src.info_v.displacements;
    ucc_aint_t        *d_disp   = TASK_ARGS(task).dst.info_v.displacements;
    size_t             sdt_size = ucc_dt_size(TASK_ARGS(task).src.info_v.datatype);
    size_t             rdt_size = ucc_dt_size(TASK_ARGS(task).dst.info_v.datatype);
    ucc_mem_map_mem_h  src_memh = TASK_ARGS(task).src_memh.local_memh;
    ucc_mem_map_mem_h *dst_memh = TASK_ARGS(task).dst_memh.global_memh;
    ucc_rank_t         peer;
    size_t             sd_disp, dd_disp, data_size;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    /* perform a put to each member peer using the peer's index in the
     * destination displacement. */
    for (peer = (grank + 1) % gsize; task->onesided.put_posted < gsize;
         peer = (peer + 1) % gsize) {
        sd_disp =
            ucc_coll_args_get_displacement(&TASK_ARGS(task), s_disp, peer) *
            sdt_size;
        dd_disp =
            ucc_coll_args_get_displacement(&TASK_ARGS(task), d_disp, peer) *
            rdt_size;
        data_size =
            ucc_coll_args_get_count(
                &TASK_ARGS(task), TASK_ARGS(task).src.info_v.counts, peer) *
            sdt_size;

        UCPCHECK_GOTO(ucc_tl_ucp_put_nb(PTR_OFFSET(src, sd_disp),
                                        PTR_OFFSET(dest, dd_disp),
                                        data_size, peer, src_memh,
                                        dst_memh, team, task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_atomic_inc(pSync, peer,
                                            dst_memh, team),
                      task, out);
    }
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
out:
    return task->super.status;
}

void ucc_tl_ucp_alltoallv_onesided_progress(ucc_coll_task_t *ctask)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(ctask, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         gsize = UCC_TL_TEAM_SIZE(team);
    long              *pSync = TASK_ARGS(task).global_work_buffer;

    if (ucc_tl_ucp_test_onesided(task, gsize) == UCC_INPROGRESS) {
        return;
    }

    pSync[0]           = 0;
    task->super.status = UCC_OK;
}

ucc_status_t ucc_tl_ucp_alltoallv_onesided_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task;
    ucc_status_t       status;

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

    task                 = ucc_tl_ucp_init_task(coll_args, team);
    *task_h              = &task->super;
    task->super.post     = ucc_tl_ucp_alltoallv_onesided_start;
    task->super.progress = ucc_tl_ucp_alltoallv_onesided_progress;
    status               = UCC_OK;
out:
    return status;
}
