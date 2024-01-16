/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgather.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"
#include <stdio.h>

ucc_status_t ucc_tl_ucp_allgather_bruck_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t      *team,
                                             ucc_coll_task_t     **task_h)
{
    ucc_status_t       status = UCC_OK;
    ucc_tl_ucp_task_t *task;
    ucc_tl_ucp_team_t *ucp_team;

    task     = ucc_tl_ucp_init_task(coll_args, team);
    ucp_team = TASK_TEAM(task);

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    printf("ucc_tl_ucp_allgather_bruck_init\n");
    if (UCC_TL_TEAM_SIZE(ucp_team) % 2) {
        tl_debug(UCC_TASK_LIB(task),
                 "odd team size is not supported, switching to ring");
        status = ucc_tl_ucp_allgather_ring_init_common(task);
    } else {
        task->super.post     = ucc_tl_ucp_allgather_bruck_start;
        task->super.progress = ucc_tl_ucp_allgather_bruck_progress;
    }

out:
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }

    *task_h = &task->super;
    return status;
}

/* Inspired by implementation: https://github.com/open-mpi/ompi/blob/main/ompi/mca/coll/base/coll_base_allgather.c */
void ucc_tl_ucp_allgather_bruck_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    ucc_rank_t         trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize     = UCC_TL_TEAM_SIZE(team);
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    size_t             count     = TASK_ARGS(task).dst.info.count;
    size_t             data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_rank_t         recvfrom, sendto;
    ucc_status_t       status;
    size_t             blockcount, distance;
    void              *tmprecv, *tmpsend;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    /* On each step doubles distance */
    distance = 1 << task->tagged.send_posted;
    printf("bruck\n");
    tmpsend = rbuf;
    while (distance < (tsize)) {

        recvfrom = (trank + distance) % tsize;
        sendto   = (trank + tsize - distance) % tsize;

        tmprecv = PTR_OFFSET(tmpsend, distance * data_size);

        if (distance <= tsize >> 1) {
            blockcount = distance;
        } else {
            /* send-recv all reminder*/
            blockcount = tsize - distance;
        }

        /* Sendreceive */
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(tmpsend, blockcount * data_size, rmem,
                                         sendto, team, task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(tmprecv, blockcount * data_size, rmem,
                                         recvfrom, team, task),
                      task, out);

        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
    }

    /* post processing step */
    if (trank != 0) {
        ucc_mc_buffer_header_t *scratch_header;
        size_t                  scratch_size = (tsize - trank) * data_size;
        /* allocate scratch buffer */
        status =
            ucc_mc_alloc(&scratch_header, scratch_size, UCC_MEMORY_TYPE_HOST);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TASK_LIB(task), "failed to allocate scratch buffer");
            ucc_tl_ucp_coll_finalize(&task->super);
            return;
        }

        status = ucc_mc_memcpy(scratch_header->addr, rbuf, scratch_size,
                               UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_HOST);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to copy data to scratch buffer");
            ucc_tl_ucp_coll_finalize(&task->super);
            return;
        }

        status = ucc_mc_memcpy(rbuf, PTR_OFFSET(rbuf, scratch_size),
                               trank * data_size, UCC_MEMORY_TYPE_HOST,
                               UCC_MEMORY_TYPE_HOST);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to move data inside rbuff buffer");
            ucc_tl_ucp_coll_finalize(&task->super);
            return;
        }

        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, trank * data_size),
                               scratch_header->addr, scratch_size,
                               UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_HOST);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to copy data from scratch to rbuff buffer");
            ucc_tl_ucp_coll_finalize(&task->super);
            return;
        }

        /* deallocate scratch buffer */
        status = ucc_mc_free(scratch_header);
        if (ucc_unlikely(status != UCC_OK)) {
            tl_error(UCC_TASK_LIB(task),
                     "failed to free scratch buffer memory");
            ucc_tl_ucp_coll_finalize(&task->super);
            return;
        }
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;

out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_bruck_done", 0);
}

ucc_status_t ucc_tl_ucp_allgather_bruck_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    size_t             count     = TASK_ARGS(task).dst.info.count;
    void              *sbuf      = TASK_ARGS(task).src.info.buffer;
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  smem      = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    ucc_rank_t         trank     = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize     = UCC_TL_TEAM_SIZE(team);
    size_t             data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_bruck_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    /* initial step: copy data on non root ranks to the beginning of buffer */
    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        status = ucc_mc_memcpy(rbuf, PTR_OFFSET(sbuf, data_size * trank),
                               data_size, rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    } else if (trank != 0) {
        status = ucc_mc_memcpy(rbuf, PTR_OFFSET(rbuf, data_size * trank),
                               data_size, rmem, rmem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}
