/**
 * Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static ucc_rank_t get_recv_from_rank(ucc_rank_t rank, ucc_rank_t size, int i)
{
    const int  i_parity = i % 2;
    int        offset_at_step[2];
    ucc_rank_t recv_data_from;

    if (rank % 2) {
        recv_data_from    = (rank - 1 + size) % size;
        offset_at_step[0] = (-2);
        offset_at_step[1] = (+2);
    } else {
        recv_data_from    = rank;
        offset_at_step[0] = (+2);
        offset_at_step[1] = (-2);
    }

    return (recv_data_from + offset_at_step[i_parity] * ucc_div_round_up(i, 2) + size) % size;
}

ucc_status_t ucc_tl_ucp_allgather_neighbor_start(ucc_coll_task_t *coll_task);

ucc_status_t ucc_tl_ucp_allgather_neighbor_init(ucc_base_coll_args_t *coll_args,
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

    if (UCC_TL_TEAM_SIZE(ucp_team) % 2) {
        tl_debug(UCC_TASK_LIB(task),
                 "odd team size is not supported, switching to ring");
        status = ucc_tl_ucp_allgather_ring_init_common(task);
    } else {
        task->super.post     = ucc_tl_ucp_allgather_neighbor_start;
        task->super.progress = ucc_tl_ucp_allgather_neighbor_progress;
    }

out:
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }

    *task_h = &task->super;
    return status;
}

/* Original implementation: https://github.com/open-mpi/ompi/blob/main/ompi/mca/coll/base/coll_base_allgather.c */
void ucc_tl_ucp_allgather_neighbor_progress(ucc_coll_task_t *coll_task)
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
    ucc_rank_t         neighbors[2], i;
    int                i_parity, even_rank;
    void              *tmprecv, *tmpsend;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    even_rank = !(trank % 2);
    if (even_rank) {
        neighbors[0] = (trank + 1) % tsize;
        neighbors[1] = (trank - 1 + tsize) % tsize;
    } else {
        neighbors[0] = (trank - 1 + tsize) % tsize;
        neighbors[1] = (trank + 1) % tsize;
    }

    while (task->tagged.send_posted < (tsize / 2)) {
        i        = task->tagged.send_posted;
        i_parity = i % 2;

        tmprecv =
            PTR_OFFSET(rbuf, get_recv_from_rank(trank, tsize, i) * data_size);
        tmpsend = PTR_OFFSET(rbuf, get_recv_from_rank(trank, tsize, i - 1) *
                                       data_size);

        /* Sendreceive */
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb(tmpsend, 2 * data_size, rmem,
                                         neighbors[i_parity], team, task),
                      task, out);
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(tmprecv, 2 * data_size, rmem,
                                         neighbors[i_parity], team, task),
                      task, out);

        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;

out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_neighbor_done",
                                     0);
}

ucc_status_t ucc_tl_ucp_allgather_neighbor_start(ucc_coll_task_t *coll_task)
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
    ucc_rank_t         neighbor;
    void              *tmprecv, *tmpsend;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_neighbor_start",
                                     0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, data_size * trank), sbuf,
                               data_size, rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    if (trank % 2) {
        neighbor = (trank - 1 + tsize) % tsize;
    } else {
        neighbor = (trank + 1) % tsize;
    }

    tmprecv = PTR_OFFSET(rbuf, neighbor * data_size);
    tmpsend = PTR_OFFSET(rbuf, trank * data_size);

    /* Sendreceive */
    UCPCHECK_GOTO(
        ucc_tl_ucp_send_nb(tmpsend, data_size, rmem, neighbor, team, task),
        task, out);
    UCPCHECK_GOTO(
        ucc_tl_ucp_recv_nb(tmprecv, data_size, rmem, neighbor, team, task),
        task, out);
out:
    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}
