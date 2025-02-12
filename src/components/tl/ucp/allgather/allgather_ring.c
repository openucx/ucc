/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

static ucc_rank_t ucc_tl_ucp_allgather_ring_get_send_block(ucc_subset_t *subset,
                                                           ucc_rank_t trank,
                                                           ucc_rank_t tsize,
                                                           int step)
{
    return ucc_ep_map_eval(subset->map, (trank - step + tsize) % tsize);
}

static ucc_rank_t ucc_tl_ucp_allgather_ring_get_recv_block(ucc_subset_t *subset,
                                                           ucc_rank_t trank,
                                                           ucc_rank_t tsize,
                                                           int step)
{
    return ucc_ep_map_eval(subset->map, (trank - step - 1 + tsize) % tsize);
}

void ucc_tl_ucp_allgather_ring_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task       = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team       = TASK_TEAM(task);
    ucc_rank_t         trank      = task->subset.myrank;
    ucc_rank_t         tsize      = (ucc_rank_t)task->subset.map.ep_num;
    void              *rbuf       = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  rmem       = TASK_ARGS(task).dst.info.mem_type;
    size_t             count      = TASK_ARGS(task).dst.info.count;
    ucc_datatype_t     dt         = TASK_ARGS(task).dst.info.datatype;
    size_t             data_size  = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t       status     = UCC_OK;
    ucc_rank_t         sendto, recvfrom, sblock, rblock;
    int                step;
    void              *buf;

    if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
        return;
    }
    sendto   = ucc_ep_map_eval(task->subset.map, (trank + 1) % tsize);
    recvfrom = ucc_ep_map_eval(task->subset.map, (trank - 1 + tsize) % tsize);

    while (task->tagged.send_posted < tsize - 1) {
        step = task->tagged.send_posted;
        sblock = task->allgather_ring.get_send_block(&task->subset, trank,
                                                     tsize, step);
        rblock = task->allgather_ring.get_recv_block(&task->subset, trank,
                                                     tsize, step);
        buf = PTR_OFFSET(rbuf, sblock * data_size);
        UCPCHECK_GOTO(
            ucc_tl_ucp_send_nb(buf, data_size, rmem, sendto, team, task),
            task, out);
        buf = PTR_OFFSET(rbuf, rblock * data_size);
        UCPCHECK_GOTO(
            ucc_tl_ucp_recv_nb(buf, data_size, rmem, recvfrom, team, task),
            task, out);
        if (UCC_INPROGRESS == ucc_tl_ucp_test(task)) {
            return;
        }
    }
    ucc_assert(UCC_TL_UCP_TASK_P2P_COMPLETE(task));
    if (task->allgather_ring.etask) {
        status = ucc_ee_executor_task_test(task->allgather_ring.etask);
        if (status == UCC_INPROGRESS) {
            return;
        }
        ucc_ee_executor_task_finalize(task->allgather_ring.etask);
    }
    task->super.status = status;
out:
    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_ring_done", 0);
}

ucc_status_t ucc_tl_ucp_allgather_ring_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_ucp_task_t *task      = ucc_derived_of(coll_task, ucc_tl_ucp_task_t);
    ucc_tl_ucp_team_t *team      = TASK_TEAM(task);
    size_t             count     = TASK_ARGS(task).dst.info.count;
    void              *sbuf      = TASK_ARGS(task).src.info.buffer;
    void              *rbuf      = TASK_ARGS(task).dst.info.buffer;
    ucc_memory_type_t  smem      = TASK_ARGS(task).src.info.mem_type;
    ucc_memory_type_t  rmem      = TASK_ARGS(task).dst.info.mem_type;
    ucc_datatype_t     dt        = TASK_ARGS(task).dst.info.datatype;
    ucc_rank_t         trank     = task->subset.myrank;
    ucc_rank_t         tsize     = (ucc_rank_t)task->subset.map.ep_num;
    size_t             data_size = (count / tsize) * ucc_dt_size(dt);
    ucc_status_t       status;
    ucc_rank_t         sendto, recvfrom, sblock, rblock;
    ucc_ee_executor_t *exec;
    ucc_ee_executor_task_args_t eargs;
    void *buf;

    UCC_TL_UCP_PROFILE_REQUEST_EVENT(coll_task, "ucp_allgather_ring_start", 0);
    ucc_tl_ucp_task_reset(task, UCC_INPROGRESS);

    sendto   = ucc_ep_map_eval(task->subset.map, (trank + 1) % tsize);
    recvfrom = ucc_ep_map_eval(task->subset.map, (trank - 1 + tsize) % tsize);
    sblock = task->allgather_ring.get_send_block(&task->subset, trank, tsize, 0);
    rblock = task->allgather_ring.get_recv_block(&task->subset, trank, tsize, 0);
    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        status = ucc_coll_task_get_executor(&task->super, &exec);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }

        eargs.task_type = UCC_EE_EXECUTOR_TASK_COPY;
        eargs.copy.src  = sbuf;
        eargs.copy.dst  = PTR_OFFSET(rbuf, data_size * sblock);
        eargs.copy.len  = data_size;

        status = ucc_ee_executor_task_post(exec, &eargs,
                                           &task->allgather_ring.etask);
        if (ucc_unlikely(status != UCC_OK)) {
            return status;
        }
        buf = sbuf;
    } else {
        task->allgather_ring.etask = NULL;
        buf = PTR_OFFSET(rbuf, data_size * sblock);
    }

    UCPCHECK_GOTO(ucc_tl_ucp_send_nb(buf, data_size, smem, sendto, team, task),
                  task, out);
    UCPCHECK_GOTO(ucc_tl_ucp_recv_nb(PTR_OFFSET(rbuf, rblock * data_size),
                                     data_size, rmem, recvfrom, team, task),
                  task, out);

    return ucc_progress_queue_enqueue(UCC_TL_CORE_CTX(team)->pq, &task->super);

out:
    return status;
}

ucc_status_t ucc_tl_ucp_allgather_ring_init_common(ucc_tl_ucp_task_t *task)
{
    ucc_tl_ucp_team_t *team = TASK_TEAM(task);
    ucc_sbgp_t *sbgp;

    if (!ucc_coll_args_is_predefined_dt(&TASK_ARGS(task), UCC_RANK_INVALID)) {
        tl_error(UCC_TASK_LIB(task), "user defined datatype is not supported");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (!(task->flags & UCC_TL_UCP_TASK_FLAG_SUBSET)) {
        if (team->cfg.use_reordering) {
            sbgp = ucc_topo_get_sbgp(team->topo, UCC_SBGP_FULL_HOST_ORDERED);
            task->subset.myrank = sbgp->group_rank;
            task->subset.map    = sbgp->map;
        }
    }

    task->allgather_ring.get_send_block = ucc_tl_ucp_allgather_ring_get_send_block;
    task->allgather_ring.get_recv_block = ucc_tl_ucp_allgather_ring_get_recv_block;
    task->super.post                    = ucc_tl_ucp_allgather_ring_start;
    task->super.progress                = ucc_tl_ucp_allgather_ring_progress;
    if (!UCC_IS_INPLACE(TASK_ARGS(task))) {
        task->super.flags |= UCC_COLL_TASK_FLAG_EXECUTOR;
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_ucp_allgather_ring_init(ucc_base_coll_args_t *coll_args,
                                            ucc_base_team_t *     team,
                                            ucc_coll_task_t **    task_h)
{
    ucc_tl_ucp_task_t *task;
    ucc_status_t status;

    task = ucc_tl_ucp_init_task(coll_args, team);
    status = ucc_tl_ucp_allgather_ring_init_common(task);
    if (status != UCC_OK) {
        ucc_tl_ucp_put_task(task);
        return status;
    }
    *task_h = &task->super;
    return UCC_OK;
}
