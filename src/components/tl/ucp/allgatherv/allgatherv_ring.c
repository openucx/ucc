/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "tl_ucp.h"
#include "allgatherv.h"
#include "core/ucc_progress_queue.h"
#include "utils/ucc_math.h"
#include "utils/ucc_coll_utils.h"
#include "tl_ucp_sendrecv.h"
#include <stdio.h>

void ucc_tl_ucp_allgatherv_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task     = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_coll_args_t   *args     = &TASK_ARGS(task);
    ucc_tl_ucp_team_t *team     = TASK_TEAM(task);
    ucc_rank_t         trank    = task->subset.myrank;
    ucc_rank_t         tsize    = (ucc_rank_t)task->subset.map.ep_num;
    ptrdiff_t          rbuf     = (ptrdiff_t)args->dst.info_v.buffer;
    ucc_memory_type_t  rmem     = args->dst.info_v.mem_type;
    size_t             rdt_size = ucc_dt_size(args->dst.info_v.datatype);
    ucc_rank_t         send_idx, recv_idx, sendto, recvfrom;
    size_t             data_size, data_displ;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }

    sendto   = ucc_ep_map_eval(task->subset.map, (trank + 1) % tsize);
    recvfrom = ucc_ep_map_eval(task->subset.map, (trank - 1 + tsize) % tsize);

    while (task->tagged.send_posted < tsize) {
        send_idx = ucc_ep_map_eval(
            task->subset.map,
            (trank - task->tagged.send_posted + 1 + tsize) % tsize);
        data_displ = ucc_coll_args_get_displacement(
                         args, args->dst.info_v.displacements, send_idx) *
                     rdt_size;
        data_size =

            ucc_coll_args_get_count(args, args->dst.info_v.counts, send_idx) *
            rdt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_send_nb((void *)(rbuf + data_displ), data_size,
                                         rmem, sendto, team, task),
                      task, out);
        recv_idx =
            ucc_ep_map_eval(task->subset.map,
                            (trank - task->tagged.recv_posted + tsize) % tsize);
        data_displ = ucc_coll_args_get_displacement(
                         args, args->dst.info_v.displacements, recv_idx) *
                     rdt_size;
        data_size =
            ucc_coll_args_get_count(args, args->dst.info_v.counts, recv_idx) *
            rdt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb((void *)(rbuf + data_displ), data_size,
                                         rmem, recvfrom, team, task),
                      task, out);
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
    }
    // DEBUG
    int _i, _total_counts;
    _total_counts=0;
    for (_i=0; _i < tsize; _i++){
        _total_counts += ucc_coll_args_get_count(args, args->dst.info_v.counts, _i);
    }
    printf("[%d] end-allgatherv rbuf: [", trank);
    for (_i=0; _i < _total_counts; _i++){
        printf("%u, ", ((uint32_t *)rbuf)[_i]);
    }
    printf("]\n");
    //--
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    task->super.status = UCC_OK;
out:
    return;
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task  = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team  = TASK_TEAM(task);
    ucc_coll_args_t   *args  = &TASK_ARGS(task);
    void              *sbuf  = args->src.info.buffer;
    void              *rbuf  = args->dst.info_v.buffer;
    ucc_memory_type_t  smem  = args->src.info.mem_type;
    ucc_memory_type_t  rmem  = args->dst.info_v.mem_type;
    ucc_rank_t         grank = UCC_TL_TEAM_RANK(team);
    size_t             data_size, data_displ, rdt_size;

    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    if (!UCC_IS_INPLACE(*args)) {
        /* TODO replace local sendrecv with memcpy? */
        rdt_size   = ucc_dt_size(args->dst.info_v.datatype);
        data_displ = ucc_coll_args_get_displacement(
                         args, args->dst.info_v.displacements, grank) *
                     rdt_size;
        data_size =
            ucc_coll_args_get_count(args, args->dst.info_v.counts, grank) *
            rdt_size;
        UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(PTR_OFFSET(rbuf, data_displ),
                                         data_size, rmem, grank, team, task),
                      task, error);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(sbuf, data_size, smem, grank, team, task), task,
            error);
    } else {
        /* to simplify progress fucnction and make it identical for
           in-place and non in-place */
        task->tagged.send_posted = task->tagged.send_completed = 1;
        task->tagged.recv_posted = task->tagged.recv_completed = 1;
    }

    // DEBUG
    // int _i;
    // printf("[%d] start-allgatherv sbuf: [", grank);
    // for (_i=0; _i < 4; _i++){
    //     printf("%u, ", ((uint32_t *)sbuf)[_i]);
    // }
    // printf("]\n");
    // -----

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);
error:
    return task->super.status;
}

ucc_status_t ucc_tl_ucp_allgatherv_ring_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_sbgp_t        *sbgp;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (team->cfg.use_reordering) {
        sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
        task->subset.myrank = sbgp->group_rank;
        task->subset.map    = sbgp->map;
    }

    task->super.post     = ucc_tl_ucp_allgatherv_ring_start;
    task->super.progress = ucc_tl_ucp_allgatherv_ring_progress;

    return UCC_OK;
}
