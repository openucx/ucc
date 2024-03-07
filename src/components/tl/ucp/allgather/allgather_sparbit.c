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

ucc_status_t ucc_tl_ucp_allgather_sparbit_init(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t      *team,
                                             ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_team_t *tl_team      = ucc_derived_of(team, ucc_tl_ucp_team_t);
    ucc_tl_ucp_task_t *task         = ucc_tl_ucp_init_task(coll_args, team);
    ucc_status_t       status       = UCC_OK;
    ucc_rank_t         trank        = UCC_TL_TEAM_RANK(tl_team);
    ucc_rank_t         tsize        = UCC_TL_TEAM_SIZE(tl_team);
    ucc_memory_type_t  rmem         = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t     dt           = TASK_ARGS(task).dst.info.datatype;
    size_t             count        = TASK_ARGS(task).dst.info.count;
    size_t             data_size    = (count / tsize) * ucc_dt_size(dt);
    size_t             scratch_size = (tsize - trank) * data_size;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        status = UCC_ERR_NOT_SUPPORTED;
        goto out;
    }
    tl_trace(UCC_TASK_LIB(task), "ucc_tl_ucp_allgather_sparbit_init");

    task->super.post     = ucc_tl_ucp_allgather_sparbit_start;
    task->super.progress = ucc_tl_ucp_allgather_sparbit_progress;

out:
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }

    *task_h = &task->super;
    return status;
}

/* Inspired by implementation: https://github.com/open-mpi/ompi/blob/main/ompi/mca/coll/base/coll_base_allgather.c */
void ucc_tl_ucp_allgather_sparbit_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t      *task = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t      *team = TASK_TEAM(task);
    ucc_rank_t              trank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t              tsize = UCC_TL_TEAM_SIZE(team);
    void                   *rbuf  = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t       rmem  = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t          dt    = TASK_ARGS(task).dst.info.datatype;
    size_t                  count = TASK_ARGS(task).dst.info.count;
    ucc_mc_buffer_header_t *scratch_header =
        task->allgather_sparbit.scratch_header;
    size_t       scratch_size = task->allgather_sparbit.scratch_size;
    size_t       data_size    = (count / tsize) * ucc_dt_size(dt);
    ucc_rank_t   recvfrom, sendto;
    ucc_status_t status;
    size_t       blockcount, distance;
    void        *tmprecv, *tmpsend;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    tsize_log = ceil(log(tsize)/log(2));
    distance = 1 << (tsize_log - 1);

    last_ignore = __builtin_ctz(tsize);
    ignore_steps = (~((uint32_t) tsize >> last_ignore) | 1) << last_ignore;

    data_expected = 1;

    tmpsend  = rbuf;
    while (i < tsize_log) {

        recvfrom = (trank + tsize - distance) % tsize;
        sendto   = (trank + distance) % tsize;
        exclusion = (distance & ignore_steps) == distance;

        for (transfer_count = 0; transfer_count < data_expected - exclusion; transfer_count++) {
            tmprecv = PTR_OFFSET(rbuf, (trank - (2 * transfer_count + 1) * distance + tsize) % tsize * data_size);
            tmpsend = PTR_OFFSET(rbuf, (trank - 2 * transfer_count * distance + tsize) % tsize * data_size)
        /* Sendreceive */
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(tmpsend, blockcount * data_size, rmem,
                                         sendto, team, task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(tmprecv, blockcount * data_size, rmem,
                                         recvfrom, team, task),
                      task, out);
        }

        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }

        distance >>= 1;
        data_expected = (data_expected << 1) - exclusion;
        exclusion = 0;
    }

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));

    task->super.status = UCC_OK;

out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_sparbit_done", 0);
}

ucc_status_t ucc_tl_ucp_allgather_sparbit_start(ucc_coll_task_t *coll_task)
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

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_sparbit_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    /* initial step: copy data on non root ranks to the beginning of buffer */
    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        // not inplace
        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, data_size * trank), sbuf, data_size, rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}
