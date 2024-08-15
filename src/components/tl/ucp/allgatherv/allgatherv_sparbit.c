/**
 * Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#include "config.h"
#include "tl_ucp.h"
#include "allgatherv.h"
#include "core/ucc_progress_queue.h"
#include "tl_ucp_sendrecv.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "components/mc/ucc_mc.h"

static void ucc_tl_ucp_allgather_sparbit_progress(ucc_coll_task_t *task);

static ucc_status_t ucc_tl_ucp_allgather_sparbit_start(ucc_coll_task_t *task);

ucc_status_t ucc_tl_ucp_allgatherv_sparbit_init(ucc_base_coll_args_t *coll_args,
                                                ucc_base_team_t      *team,
                                                ucc_coll_task_t     **task_h)
{
    ucc_tl_ucp_task_t *task = ucc_tl_ucp_init_task(coll_args, team);

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        ucc_tl_ucp_put_task(task);
        return UCC_ERR_NOT_SUPPORTED;
    }
    tl_trace(UCC_TASK_LIB(task), "ucc_tl_ucp_allgather_sparbit_init");

    ucc_info("sparbitV");

    task->super.post     = ucc_tl_ucp_allgather_sparbit_start;
    task->super.progress = ucc_tl_ucp_allgather_sparbit_progress;

    *task_h = &task->super;

    return UCC_OK;
}

/* Inspired by implementation: https://github.com/open-mpi/ompi/blob/main/ompi/mca/coll/base/coll_base_allgather.c */
void ucc_tl_ucp_allgather_sparbit_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_rank_t         trank = UCC_TL_TEAM_RANK(team);
    ucc_rank_t         tsize = UCC_TL_TEAM_SIZE(team);
    void              *rbuf  = args->dst.info_v.buffer;
    ucc_memory_type_t  rmem  = args->dst.info_v.mem_type;
    uint32_t i        = task->allgather_sparbit.i; // restore iteration number
    size_t   rdt_size = ucc_dt_size(args->dst.info_v.datatype);

    uint32_t   tsize_log = ucc_ilog2_ceil(tsize);
    ucc_rank_t recvfrom, sendto, distance;
    uint32_t   last_ignore, ignore_steps, data_expected, transfer_count;
    uint32_t   exclusion;
    void      *tmprecv, *tmpsend;

    // here we can't make any progress while transfers from previous step are running, emulation of wait all in async manner
    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    last_ignore  = __builtin_ctz(tsize); //  count trailing zeros
    ignore_steps = (~((uint32_t)tsize >> last_ignore) | 1) << last_ignore;

    while (i < tsize_log) {
        data_expected = task->allgather_sparbit.data_expected;

        distance = 1 << (tsize_log - 1);
        distance >>=
            i; // restore distance in case of continuation depending on step

        recvfrom  = (trank + tsize - distance) % tsize;
        sendto    = (trank + distance) % tsize;
        exclusion = (distance & ignore_steps) == distance;

        for (transfer_count = 0; transfer_count < data_expected - exclusion;
             transfer_count++) {
            int send_disp_idx =
                (trank - 2 * transfer_count * distance + tsize) % tsize;
            int recv_disp_idx =
                (trank - (2 * transfer_count + 1) * distance + tsize) % tsize;

            size_t data_displ_send =
                ucc_coll_args_get_displacement(
                    args, args->dst.info_v.displacements, send_disp_idx) *
                rdt_size;
            size_t data_displ_recv =
                ucc_coll_args_get_displacement(
                    args, args->dst.info_v.displacements, recv_disp_idx) *
                rdt_size;

            size_t data_size_send =
                ucc_coll_args_get_count(args, args->dst.info_v.counts,
                                        send_disp_idx) *
                rdt_size;
            size_t data_size_recv =
                ucc_coll_args_get_count(args, args->dst.info_v.counts,
                                        recv_disp_idx) *
                rdt_size;

            /* Sendreceive */
            tmpsend = PTR_OFFSET(rbuf, data_displ_send);
            tmprecv = PTR_OFFSET(rbuf, data_displ_recv);
            UCPCHECK_GOTO(ucc_tl_ucp_send_nb(tmpsend, data_size_send, rmem,
                                             sendto, team, task),
                          task, out);
            UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(tmprecv, data_size_recv, rmem,
                                             recvfrom, team, task),
                          task, out);
        }

        task->allgather_sparbit.data_expected =
            (data_expected << 1) - exclusion;
        task->allgather_sparbit.i++;
        // check if we could make one more step right now or we should yeld task
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
        i = task->allgather_sparbit.i;
    }

    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;

out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_sparbit_done",
                                     0);
}

ucc_status_t ucc_tl_ucp_allgather_sparbit_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    void              *sbuf  = args->src.info.buffer;
    void              *rbuf  = args->dst.info_v.buffer;
    ucc_memory_type_t  smem  = args->src.info.mem_type;
    ucc_memory_type_t  rmem  = args->dst.info_v.mem_type;
    ucc_rank_t         trank = UCC_TL_TEAM_RANK(team);
    size_t             data_size, data_displ, rdt_size;
    ucc_status_t       status;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgatherv_sparbit_start",
                                     0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);
    task->allgather_sparbit.i             = 0; // setup iteration
    task->allgather_sparbit.data_expected = 1;

    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        rdt_size   = ucc_dt_size(args->dst.info_v.datatype);
        data_displ = ucc_coll_args_get_displacement(
                         args, args->dst.info_v.displacements, trank) *
                     rdt_size;
        data_size =
            ucc_coll_args_get_count(args, args->dst.info_v.counts, trank) *
            rdt_size;

        status = ucc_mc_memcpy(PTR_OFFSET(rbuf, data_displ), sbuf, data_size,
                               rmem, smem);
        if (ucc_unlikely(UCC_OK != status)) {
            return status;
        }
    }

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
}
