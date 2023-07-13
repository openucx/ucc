/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "alltoall/alltoall.h"
#include "alltoall/alltoall_mkeys.h"
#include "alltoall/alltoall_inline.h"

#include "components/mc/ucc_mc.h"
#include "core/ucc_team.h"
#include "tl_mlx5_wqe.h"
#include "tl_mlx5_ib.h"

static ucc_status_t
ucc_tl_mlx5_poll_free_op_slot_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task      = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team      = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_alltoall_t *a2a       = team->a2a;
    int                     seq_index = task->alltoall.seq_index;

    if (a2a->op_busy[seq_index] && !task->alltoall.started) {
        tl_debug(
            UCC_TL_TEAM_LIB(team),
            "Operation num %d must wait for previous outstanding to complete",
            task->alltoall.seq_num);
    }

    coll_task->status       = UCC_INPROGRESS;
    coll_task->super.status = UCC_INPROGRESS;
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

void ucc_tl_mlx5_poll_free_op_slot_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task      = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team      = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_alltoall_t *a2a       = team->a2a;
    int                     seq_index = task->alltoall.seq_index;

    if (a2a->op_busy[seq_index] && !task->alltoall.started) {
        coll_task->status = UCC_INPROGRESS;
        return;
    } //wait for slot to be open
    a2a->op_busy[seq_index] = 1;
    task->alltoall.started  = 1;
    coll_task->status       = UCC_OK;
    tl_debug(UCC_TL_TEAM_LIB(team), "Operation num %d started",
             task->alltoall.seq_num);
}

static ucc_status_t ucc_tl_mlx5_poll_cq(struct ibv_cq *cq, ucc_base_lib_t *lib)
{
    int           i, completions_num;
    struct ibv_wc wcs[MIN_POLL_WC];

    completions_num = ibv_poll_cq(cq, MIN_POLL_WC, wcs);
    if (completions_num < 0) {
        tl_error(lib, "ibv_poll_cq() failed, errno %d", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    for (i = 0; i < completions_num; i++) {
        if (wcs[i].status != IBV_WC_SUCCESS) {
            tl_error(lib, "bad work completion status %s, wr_id %zu",
                     ibv_wc_status_str(wcs[i].status), wcs[i].wr_id);
            return UCC_ERR_NO_MESSAGE;
        }

        ucc_assert(wcs[i].opcode == IBV_WC_DRIVER2);
        if (wcs[i].wr_id == 0) {
            /* signalled transpose */
            continue;
        } else if (wcs[i].wr_id & 0x1) {
            ucc_tl_mlx5_schedule_t *task =
                (ucc_tl_mlx5_schedule_t *)(uintptr_t)(wcs[i].wr_id &
                                                      (~(uint64_t)0x1));
            task->alltoall.wait_wc = 1;
        } else {
            ucc_tl_mlx5_dm_chunk_t *dm = (ucc_tl_mlx5_dm_chunk_t *)wcs[i].wr_id;
            dm->task->alltoall.blocks_completed++;
            ucc_mpool_put(dm);
        }
    }
    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_node_fanin(ucc_tl_mlx5_team_t     *team,
                                           ucc_tl_mlx5_schedule_t *task)
{
    ucc_tl_mlx5_alltoall_t      *a2a       = team->a2a;
    int                          seq_index = task->alltoall.seq_index;
    int                          i;
    ucc_tl_mlx5_alltoall_ctrl_t *ctrl_v;

    if (a2a->node.sbgp->group_rank != a2a->node.asr_rank) {
        ucc_tl_mlx5_get_my_ctrl(a2a, seq_index)->seq_num =
            task->alltoall.seq_num;
    } else {
        for (i = 0; i < a2a->node.sbgp->group_size; i++) {
            if (i == a2a->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = ucc_tl_mlx5_get_ctrl(a2a, seq_index, i);
            if (ctrl_v->seq_num != task->alltoall.seq_num) {
                return UCC_INPROGRESS;
            }
        }
        for (i = 0; i < a2a->node.sbgp->group_size; i++) {
            if (i == a2a->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = ucc_tl_mlx5_get_ctrl(a2a, seq_index, i);
            ucc_tl_mlx5_get_my_ctrl(a2a, seq_index)->mkey_cache_flag |=
                ctrl_v->mkey_cache_flag;
        }
    }
    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanin_done", 0);
    return UCC_OK;
}

/* Each rank registers sbuf and rbuf and place the registration data
   in the shared memory location. Next, all rank in node nitify the
   ASR the registration data is ready using SHM Fanin */
static ucc_status_t ucc_tl_mlx5_reg_fanin_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t      *task            = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t          *team            = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_context_t       *ctx             = UCC_TL_MLX5_TEAM_CTX(team);
    ucc_tl_mlx5_alltoall_t      *a2a             = team->a2a;
    int                          reg_change_flag = 0;
    int                          seq_index       = task->alltoall.seq_index;
    int                          flag;
    ucc_tl_mlx5_rcache_region_t *send_ptr;
    ucc_tl_mlx5_rcache_region_t *recv_ptr;

    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanin_start", 0);
    tl_debug(UCC_TASK_LIB(task), "register memory buffers");
    coll_task->status       = UCC_INPROGRESS;
    coll_task->super.status = UCC_INPROGRESS;

    errno = 0;
    if (UCC_OK !=
        ucc_rcache_get(ctx->rcache, (void *)SCHEDULE_ARGS(task).src.info.buffer,
                       task->alltoall.msg_size * UCC_TL_TEAM_SIZE(team),
                       &reg_change_flag, (ucc_rcache_region_t **)&send_ptr)) {
        tl_error(UCC_TASK_LIB(task),
                 "Failed to register send_bf memory (errno=%d)", errno);
        return UCC_ERR_NO_RESOURCE;
    }
    task->alltoall.send_rcache_region_p = send_ptr;

    /* NOTE: we don't support alternating block_size for the same msg size - TODO
       Will need to add the possibility of block_size change into consideration
       when initializing the mkey_cache_flag */

    flag =
        (task->alltoall.msg_size == a2a->previous_msg_size[seq_index])
            ? 0
            : (UCC_MLX5_NEED_SEND_MKEY_UPDATE | UCC_MLX5_NEED_RECV_MKEY_UPDATE);

    if (reg_change_flag || (task->alltoall.send_rcache_region_p->reg.mr->addr !=
                            a2a->previous_send_address[seq_index])) {
        flag |= UCC_MLX5_NEED_SEND_MKEY_UPDATE;
    }
    reg_change_flag = 0;
    if (UCC_OK !=
        ucc_rcache_get(ctx->rcache, (void *)SCHEDULE_ARGS(task).dst.info.buffer,
                       task->alltoall.msg_size * UCC_TL_TEAM_SIZE(team),
                       &reg_change_flag, (ucc_rcache_region_t **)&recv_ptr)) {
        tl_error(UCC_TASK_LIB(task), "Failed to register receive_bf memory");
        ucc_rcache_region_put(ctx->rcache,
                              &task->alltoall.send_rcache_region_p->super);
        return UCC_ERR_NO_RESOURCE;
    }
    task->alltoall.recv_rcache_region_p = recv_ptr;
    if (reg_change_flag || (task->alltoall.recv_rcache_region_p->reg.mr->addr !=
                            a2a->previous_recv_address[seq_index])) {
        flag |= UCC_MLX5_NEED_RECV_MKEY_UPDATE;
    }

    tl_debug(UCC_TL_MLX5_TEAM_LIB(team), "fanin start");
    /* start task if completion event received */
    /* Start fanin */
    ucc_tl_mlx5_get_my_ctrl(a2a, seq_index)->mkey_cache_flag = flag;
    ucc_tl_mlx5_update_mkeys_entries(a2a, task, flag);

    if (UCC_OK == ucc_tl_mlx5_node_fanin(team, task)) {
        tl_debug(UCC_TL_MLX5_TEAM_LIB(team), "fanin complete");
        coll_task->status = UCC_OK;
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanin_done", 0);
        return ucc_task_complete(coll_task);
    }

    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

void ucc_tl_mlx5_reg_fanin_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team = SCHEDULE_TEAM(task);

    ucc_assert(team->a2a->node.sbgp->group_rank == team->a2a->node.asr_rank);
    if (UCC_OK == ucc_tl_mlx5_node_fanin(team, task)) {
        tl_debug(UCC_TL_MLX5_TEAM_LIB(team), "fanin complete");
        coll_task->status = UCC_OK;
    }
}

static ucc_status_t ucc_tl_mlx5_node_fanout(ucc_tl_mlx5_team_t     *team,
                                            ucc_tl_mlx5_schedule_t *task)
{
    ucc_tl_mlx5_alltoall_t      *a2a = team->a2a;
    ucc_tl_mlx5_alltoall_ctrl_t *ctrl_v;
    /* First phase of fanout: asr signals it completed local ops
       and other ranks wait for asr */
    if (a2a->node.sbgp->group_rank == a2a->node.asr_rank) {
        /* no need to check counter - we wait on data in device */
        ucc_tl_mlx5_get_my_ctrl(a2a, task->alltoall.seq_index)->seq_num =
            task->alltoall.seq_num;
    } else {
        ctrl_v = ucc_tl_mlx5_get_ctrl(a2a, task->alltoall.seq_index,
                                      a2a->node.asr_rank);
        if (ctrl_v->seq_num != task->alltoall.seq_num) {
            return UCC_INPROGRESS;
        }
    }
    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_fanout_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team = SCHEDULE_TEAM(task);

    coll_task->status       = UCC_INPROGRESS;
    coll_task->super.status = UCC_INPROGRESS;

    tl_debug(UCC_TASK_LIB(task), "fanout start");
    /* start task if completion event received */
    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanout_start", 0);
    /* Start fanout */
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

static void ucc_tl_mlx5_fanout_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_alltoall_t *a2a  = team->a2a;
    ucc_status_t            status;

    if (a2a->node.sbgp->group_rank == a2a->node.asr_rank) {
        status = ucc_tl_mlx5_poll_cq(a2a->net.umr_cq, UCC_TASK_LIB(task));
        if (UCC_OK != status) {
            coll_task->status = status;
            return;
        }
        if (!task->alltoall.wait_wc) {
            coll_task->status = UCC_INPROGRESS;
            return;
        }
    }

    if (UCC_OK == ucc_tl_mlx5_node_fanout(team, task)) {
        /*Cleanup alg resources - all done */
        tl_debug(UCC_TASK_LIB(task), "Algorithm completion");
        a2a->op_busy[task->alltoall.seq_index] = 0;
        coll_task->status                      = UCC_OK;
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanout_done", 0);
    }
}

static ucc_status_t ucc_tl_mlx5_asr_barrier_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task  = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team  = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_alltoall_t *a2a   = team->a2a;
    tl_mlx5_barrier_t      *local = tl_mlx5_barrier_local_addr(task);
    ucc_status_t            status;
    int                     i;

    coll_task->status       = UCC_INPROGRESS;
    coll_task->super.status = UCC_INPROGRESS;

    task->alltoall.started = 0;
    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_barrier_start", 0);
    // despite while statement in poll_umr_cq, non blocking because have independent cq,
    // will be finished in a finite time
    ucc_tl_mlx5_populate_send_recv_mkeys(team, task);

    //Reset atomic notification counter to 0
#if ATOMIC_IN_MEMIC
    tl_mlx5_atomic_t zero = 0;
    if (0 !=
        ibv_memcpy_to_dm(a2a->net.atomic.counters,
                         task->alltoall.seq_index * sizeof(tl_mlx5_atomic_t),
                         &zero, sizeof(tl_mlx5_atomic_t))) {
        tl_error(UCC_TASK_LIB(task), "failed to reset atomic in memic");
        return UCC_ERR_NO_MESSAGE;
    }
#else
    a2a->net.atomic.counters[task->alltoall.seq_index] = 0;
#endif
    if (UCC_TL_MLX5_TEAM_LIB(team)->cfg.asr_barrier) {
        tl_debug(UCC_TASK_LIB(task), "asr barrier start");
        status = ucc_service_allreduce(
            UCC_TL_CORE_TEAM(team), &task->alltoall.barrier_scratch[0],
            &task->alltoall.barrier_scratch[1], UCC_DT_INT32, 1, UCC_OP_SUM,
            ucc_sbgp_to_subset(a2a->net.sbgp), &task->alltoall.barrier_req);
        if (status < 0) {
            tl_error(UCC_TASK_LIB(task), "failed to start asr barrier");
        }
        for (i = 0; i < a2a->net.net_size; i++) {
            task->alltoall.op->blocks_sent[i] = 0;
            a2a->net.barrier
                .flags[(a2a->net.net_size + 1) * task->alltoall.seq_index + i] =
                task->alltoall.seq_num;
        }
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    } else {
        *local = task->alltoall.seq_num;
        for (i = 0; i < a2a->net.net_size; i++) {
            task->alltoall.op->blocks_sent[i] = 0;
            if (i == a2a->net.sbgp->group_rank) {
                tl_mlx5_barrier_flag_set(task, i);
                continue;
            }

            send_start(team, i);
            status = send_block_data(
                a2a, i, (uintptr_t)local, sizeof(tl_mlx5_barrier_t),
                a2a->net.barrier.mr->lkey,
                tl_mlx5_barrier_my_remote_addr(task, i),
                tl_mlx5_barrier_remote_rkey(task, i), 0, NULL);
            if (UCC_OK == status) {
                status = send_done(team, i);
            }
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task), "failed  sending barrier notice");
                return status;
            }
        }
        coll_task->status = UCC_OK;
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_barreir_done",
                                          0);
        return ucc_task_complete(coll_task);
    }
    return UCC_OK;
}

static void ucc_tl_mlx5_asr_barrier_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task = TASK_SCHEDULE(coll_task);
    ucc_status_t            status;

    status = ucc_collective_test(&task->alltoall.barrier_req->task->super);
    if (status < 0) {
        tl_error(UCC_TASK_LIB(task), "failure during asr barrier");
    } else if (UCC_OK == status) {
        tl_debug(UCC_TASK_LIB(task), "asr barrier done");
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_barreir_done",
                                          0);
        ucc_service_coll_finalize(task->alltoall.barrier_req);
        coll_task->status = UCC_OK;
    }
}

// add polling mechanism for blocks in order to maintain const qp tx rx
static ucc_status_t ucc_tl_mlx5_send_blocks_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task           = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team           = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_alltoall_t *a2a            = team->a2a;
    int                     node_size      = a2a->node.sbgp->group_size;
    int                     net_size       = a2a->net.sbgp->group_size;
    int                     op_msgsize     = node_size * a2a->max_msg_size
                                                     * UCC_TL_TEAM_SIZE(team)
                                                     * a2a->max_num_of_columns;
    int                     node_msgsize   = SQUARED(node_size)
                                                     * task->alltoall.msg_size;
    int                     block_size     = task->alltoall.block_size;
    int                     col_msgsize    = task->alltoall.msg_size
                                                      * block_size * node_size;
    int                     block_msgsize  = SQUARED(block_size)
                                                     * task->alltoall.msg_size;
    int                     dm_host        = UCC_TL_MLX5_TEAM_LIB(team)
                                                                 ->cfg.dm_host;
    ucc_status_t            status         = UCC_OK;
    ucc_base_lib_t         *lib            = UCC_TASK_LIB(task);
    int                     seq_index      = task->alltoall.seq_index;
    int                     i, j, k, rank, dest_rank, cyc_rank;
    uint64_t                src_addr, remote_addr;
    ucc_tl_mlx5_dm_chunk_t *dm;
    uintptr_t               dm_addr;

    coll_task->status       = UCC_INPROGRESS;
    coll_task->super.status = UCC_INPROGRESS;

    tl_debug(lib, "send blocks start");
    rank = a2a->net.rank_map[a2a->net.sbgp->group_rank];
    if (!task->alltoall.send_blocks_enqueued) {
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task,
                                          "mlx5_alltoall_block_send_start", 0);
    }

    for (i = 0; i < net_size; i++) {
        cyc_rank  = (i + a2a->net.sbgp->group_rank) % net_size;
        dest_rank = a2a->net.rank_map[cyc_rank];
        if (task->alltoall.op->blocks_sent[cyc_rank] ||
            tl_mlx5_barrier_flag(task, cyc_rank) != task->alltoall.seq_num) {
            continue;
        }

        //send all blocks from curr node to some ARR
        for (j = 0; j < (node_size / block_size); j++) {
            for (k = 0; k < (node_size / block_size); k++) {
                src_addr = (uintptr_t)(node_msgsize * dest_rank +
                                       col_msgsize * j + block_msgsize * k);
                remote_addr =
                    (uintptr_t)(op_msgsize * seq_index + node_msgsize * rank +
                                block_msgsize * j + col_msgsize * k);

                send_start(team, cyc_rank);
                if (cyc_rank == a2a->net.sbgp->group_rank) {
                    status = ucc_tl_mlx5_post_transpose(
                        tl_mlx5_get_qp(a2a, cyc_rank),
                        a2a->node.ops[seq_index].send_mkeys[0]->lkey,
                        a2a->net.rkeys[cyc_rank], src_addr, remote_addr,
                        task->alltoall.msg_size, block_size, block_size,
                        (j == 0 && k == 0) ? IBV_SEND_SIGNALED : 0);
                    if (UCC_OK != status) {
                        return status;
                    }
                } else {
                    dm = ucc_mpool_get(&team->dm_pool);
                    while (!dm) {
                        status = send_done(team, cyc_rank);
                        if (UCC_OK != status) {
                            return status;
                        }

                        status = ucc_tl_mlx5_poll_cq(a2a->net.cq, lib);
                        if (UCC_OK != status) {
                            return status;
                        }
                        dm = ucc_mpool_get(&team->dm_pool);
                        send_start(team, cyc_rank);
                    }
                    if (dm_host) {
                        dm_addr =
                            (uintptr_t)PTR_OFFSET(team->dm_ptr, dm->offset);
                    } else {
                        dm_addr = dm->offset; // dm reg mr 0 based
                    }
                    dm->task = task;

                    status = ucc_tl_mlx5_post_transpose(
                        tl_mlx5_get_qp(a2a, cyc_rank),
                        a2a->node.ops[seq_index].send_mkeys[0]->lkey,
                        team->dm_mr->rkey, src_addr, dm_addr,
                        task->alltoall.msg_size, block_size, block_size, 0);
                    if (UCC_OK != status) {
                        return status;
                    }
                    status = send_block_data(
                        a2a, cyc_rank, dm_addr, block_msgsize,
                        team->dm_mr->lkey, remote_addr,
                        a2a->net.rkeys[cyc_rank], IBV_SEND_SIGNALED, dm);
                    if (status != UCC_OK) {
                        tl_error(lib, "failed sending block [%d,%d,%d]", i, j,
                                 k);
                        return status;
                    }
                }
                status = send_done(team, cyc_rank);
                if (status != UCC_OK) {
                    return status;
                }
            }
        }
        send_start(team, cyc_rank);
        status = send_atomic(a2a, cyc_rank, tl_mlx5_atomic_addr(task, cyc_rank),
                             tl_mlx5_atomic_rkey(task, cyc_rank));

        if (UCC_OK == status) {
            status = send_done(team, cyc_rank);
        }
        task->alltoall.op->blocks_sent[cyc_rank] = 1;
        task->alltoall.started++;
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "Failed sending atomic to node [%d]",
                     cyc_rank);
            return status;
        }
    }
    if (!task->alltoall.send_blocks_enqueued) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
        task->alltoall.send_blocks_enqueued = 1;
    }

    if (task->alltoall.started == a2a->net.net_size) {
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_block_send_done",
                                          0);
        status = ucc_tl_mlx5_post_wait_on_data(
            a2a->net.umr_qp, a2a->net.net_size, a2a->net.atomic.mr->lkey,
            (uintptr_t)
#if ATOMIC_IN_MEMIC
                PTR_OFFSET(0,
#else
                PTR_OFFSET(a2a->net.atomic.counters,
#endif
                           seq_index * sizeof(tl_mlx5_atomic_t)),
            task);
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(
            task, "mlx5_alltoall_block_post_wait_on_data_done", 0);
    }
    return status;
}

static ucc_status_t
ucc_tl_mlx5_send_blocks_leftovers_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task      = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team      = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_alltoall_t *a2a       = team->a2a;
    int                     node_size = a2a->node.sbgp->group_size;
    int                     net_size  = a2a->net.sbgp->group_size;
    int                     seq_index = task->alltoall.seq_index;
    size_t                  msg_size  = task->alltoall.msg_size;
    int op_msgsize = node_size * a2a->max_msg_size * UCC_TL_TEAM_SIZE(team) *
                     a2a->max_num_of_columns;
    int mkey_msgsize  = node_size * a2a->max_msg_size * UCC_TL_TEAM_SIZE(team);
    int block_size    = task->alltoall.block_size;
    int col_msgsize   = msg_size * block_size * node_size;
    int block_msgsize = SQUARED(block_size) * msg_size;
    int block_size_leftovers_side = node_size % block_size;
    int col_msgsize_leftovers =
        msg_size * block_size_leftovers_side * node_size;
    int block_msgsize_leftovers =
        block_size_leftovers_side * block_size * msg_size;
    int          corner_msgsize = SQUARED(block_size_leftovers_side) * msg_size;
    int          dm_host        = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host;
    ucc_status_t status         = UCC_OK;
    ucc_base_lib_t *lib         = UCC_TASK_LIB(coll_task);
    int             nbc         = task->alltoall.num_of_blocks_columns;

    int i, j, k, dest_rank, rank, cyc_rank, current_block_msgsize, bs_x, bs_y;
    uint64_t                src_addr, remote_addr;
    ucc_tl_mlx5_dm_chunk_t *dm;
    uintptr_t               dm_addr;

    coll_task->status       = UCC_INPROGRESS;
    coll_task->super.status = UCC_INPROGRESS;

    tl_debug(UCC_TASK_LIB(task), "send blocks start");
    rank = a2a->net.rank_map[a2a->net.sbgp->group_rank];

    for (i = 0; i < net_size; i++) {
        cyc_rank  = (i + a2a->net.sbgp->group_rank) % net_size;
        dest_rank = a2a->net.rank_map[cyc_rank];
        if (task->alltoall.op->blocks_sent[cyc_rank] ||
            tl_mlx5_barrier_flag(task, cyc_rank) != task->alltoall.seq_num) {
            continue;
        }
        //send all blocks from curr node to some ARR
        for (j = 0; j < nbc; j++) {
            for (k = 0; k < nbc; k++) {
                if (j != (nbc - 1)) {
                    src_addr = (uintptr_t)(col_msgsize * dest_rank +
                                           block_msgsize * k);
                } else {
                    src_addr = (uintptr_t)(col_msgsize_leftovers * dest_rank +
                                           block_msgsize_leftovers * k);
                }
                if (k != (nbc - 1)) {
                    remote_addr = (uintptr_t)(
                        op_msgsize * seq_index + col_msgsize * rank +
                        block_msgsize * j + mkey_msgsize * k);
                    current_block_msgsize = (j != (nbc - 1))
                                                ? block_msgsize
                                                : block_msgsize_leftovers;
                } else {
                    remote_addr = (uintptr_t)(
                        op_msgsize * seq_index + col_msgsize_leftovers * rank +
                        block_msgsize_leftovers * j + mkey_msgsize * k);
                    current_block_msgsize = (j != (nbc - 1))
                                                ? block_msgsize_leftovers
                                                : corner_msgsize;
                }
                bs_x = k < nbc - 1 ? block_size : block_size_leftovers_side;
                bs_y = j < nbc - 1 ? block_size : block_size_leftovers_side;

                send_start(team, cyc_rank);

                //todo : start/end for RC ?
                if (bs_x == 1 || bs_y == 1) {
                    status = send_block_data(
                        a2a, cyc_rank, src_addr, current_block_msgsize,
                        a2a->node.ops[seq_index].send_mkeys[j]->lkey,
                        remote_addr, a2a->net.rkeys[cyc_rank], 0, NULL);
                } else {
                    dm = ucc_mpool_get(&team->dm_pool);
                    while (!dm) {
                        status = send_done(team, cyc_rank);
                        if (UCC_OK != status) {
                            return status;
                        }

                        status = ucc_tl_mlx5_poll_cq(a2a->net.cq, lib);
                        if (UCC_OK != status) {
                            return status;
                        }
                        dm = ucc_mpool_get(&team->dm_pool);
                        send_start(team, cyc_rank);
                    }
                    if (dm_host) {
                        dm_addr =
                            (uintptr_t)PTR_OFFSET(team->dm_ptr, dm->offset);
                    } else {
                        dm_addr = dm->offset; // dm reg mr 0 based
                    }
                    dm->task = task;

                    status = ucc_tl_mlx5_post_transpose(
                        tl_mlx5_get_qp(a2a, cyc_rank),
                        a2a->node.ops[seq_index].send_mkeys[j]->lkey,
                        team->dm_mr->rkey, src_addr, dm_addr, msg_size, bs_x,
                        bs_y, 0);
                    if (UCC_OK != status) {
                        return status;
                    }

                    status = send_block_data(
                        a2a, cyc_rank, dm_addr, current_block_msgsize,
                        team->dm_mr->lkey, remote_addr,
                        a2a->net.rkeys[cyc_rank], IBV_SEND_SIGNALED, dm);
                }
                if (status != UCC_OK) {
                    tl_error(UCC_TASK_LIB(task),
                             "Failed sending block [%d,%d,%d]", i, j, k);
                    return status;
                }
                status = send_done(team, cyc_rank);
                if (UCC_OK != status) {
                    return status;
                }
            }
        }
        send_start(team, cyc_rank);
        status = send_atomic(a2a, cyc_rank, tl_mlx5_atomic_addr(task, cyc_rank),
                             tl_mlx5_atomic_rkey(task, cyc_rank));
        if (UCC_OK == status) {
            status = send_done(team, cyc_rank);
        }
        task->alltoall.op->blocks_sent[cyc_rank] = 1;
        task->alltoall.started++;
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task), "Failed sending atomic to node [%d]",
                     cyc_rank);
            return status;
        }
    }
    if (!task->alltoall.send_blocks_enqueued) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
        task->alltoall.send_blocks_enqueued = 1;
    }

    if (task->alltoall.started == a2a->net.net_size) {
        status = ucc_tl_mlx5_post_wait_on_data(
            a2a->net.umr_qp, a2a->net.net_size, a2a->net.atomic.mr->lkey,
            (uintptr_t)
#if ATOMIC_IN_MEMIC
                PTR_OFFSET(0,
#else
                PTR_OFFSET(a2a->net.atomic.counters,
#endif
                           seq_index * sizeof(tl_mlx5_atomic_t)),
            task);
    }

    return status;
}

static void ucc_tl_mlx5_send_blocks_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t     *team = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_alltoall_t *a2a  = team->a2a;
    ucc_status_t            status;

    if (task->alltoall.started != a2a->net.net_size) {
        coll_task->post(coll_task);
        return;
    }
    status = ucc_tl_mlx5_poll_cq(a2a->net.cq, UCC_TASK_LIB(coll_task));
    if (UCC_OK != status) {
        coll_task->status = status;
        return;
    }

    if (task->alltoall.blocks_sent == task->alltoall.blocks_completed) {
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(
            task, "mlx5_alltoall_all_blocks_completed", 0);
        coll_task->status = UCC_OK;
    }
}

ucc_status_t ucc_tl_mlx5_alltoall_start(ucc_coll_task_t *coll_task)
{
    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(coll_task, "mlx5_alltoall_start", 0);
    return ucc_schedule_start(coll_task);
}

ucc_status_t ucc_tl_mlx5_alltoall_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task =
        ucc_derived_of(coll_task, ucc_tl_mlx5_schedule_t);
    ucc_tl_mlx5_team_t     *team      = SCHEDULE_TEAM(task);
    ucc_tl_mlx5_alltoall_t *a2a       = team->a2a;
    ucc_tl_mlx5_context_t  *ctx       = SCHEDULE_CTX(task);
    int                     seq_index = task->alltoall.seq_index;
    ucc_status_t            status;

    a2a->previous_msg_size[seq_index] = task->alltoall.msg_size;
    a2a->previous_send_address[seq_index] =
        task->alltoall.send_rcache_region_p->reg.mr->addr;
    a2a->previous_recv_address[seq_index] =
        task->alltoall.recv_rcache_region_p->reg.mr->addr;
    ucc_rcache_region_put(ctx->rcache,
                          &task->alltoall.send_rcache_region_p->super);
    ucc_rcache_region_put(ctx->rcache,
                          &task->alltoall.recv_rcache_region_p->super);
    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(coll_task, "mlx5_alltoall_done", 0);
    status = ucc_schedule_finalize(coll_task);
    ucc_tl_mlx5_put_schedule(task);
    return status;
}

static inline int power2(int value)
{
    int p = 2;

    while (p < value) {
        p *= 2;
    }
    return p;
}

static inline int block_size_fits(size_t msgsize, int block_size)
{
    return block_size * ucc_max(power2(block_size) * msgsize, MAX_MSG_SIZE) <=
           MAX_TRANSPOSE_SIZE;
}

static inline int get_block_size(ucc_tl_mlx5_schedule_t *task)
{
    ucc_tl_mlx5_team_t *team              = SCHEDULE_TEAM(task);
    int                 ppn               = team->a2a->node.sbgp->group_size;
    size_t              effective_msgsize = power2(ucc_max(
                                                task->alltoall.msg_size, 8));
    int                 block_size;

    block_size = ppn;
    while (!block_size_fits(effective_msgsize, block_size)) {
        block_size--;
    }

    return ucc_max(1, block_size);
}

UCC_TL_MLX5_PROFILE_FUNC(ucc_status_t, ucc_tl_mlx5_alltoall_init,
                         (coll_args, team, task_h),
                         ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                         ucc_coll_task_t **task_h)
{
    ucc_tl_mlx5_team_t     *tl_team   = ucc_derived_of(team,
                                                           ucc_tl_mlx5_team_t);
    ucc_tl_mlx5_alltoall_t *a2a       = tl_team->a2a;
    int                     is_asr    = (a2a->node.sbgp->group_rank
                                                        == a2a->node.asr_rank);
    int                     n_tasks   = is_asr ? 5 : 3;
    int                     curr_task = 0;
    ucc_schedule_t         *schedule;
    ucc_tl_mlx5_schedule_t *task;
    size_t                  msg_size;
    int                     block_size, i;
    ucc_coll_task_t        *tasks[5];
    ucc_status_t            status;

    if (UCC_IS_INPLACE(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    msg_size = (coll_args->args.src.info.count *
                ucc_dt_size(coll_args->args.src.info.datatype)) /
               UCC_TL_TEAM_SIZE(tl_team);
    if (!msg_size) {
        tl_trace(UCC_TL_TEAM_LIB(tl_team),
                 "msg size too short, reduces to nullop");
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (msg_size > a2a->max_msg_size) {
        tl_trace(UCC_TL_TEAM_LIB(tl_team), "msg size too long");
        return UCC_ERR_NOT_SUPPORTED;
    }

    status = ucc_tl_mlx5_get_schedule(tl_team, coll_args, &task);
    if (ucc_unlikely(UCC_OK != status)) {
        return status;
    }
    schedule = &task->super;

    for (i = 0; i < n_tasks; i++) {
        tasks[i] = &ucc_tl_mlx5_init_task(coll_args, team, schedule)->super;
    }

    task->alltoall.send_blocks_enqueued = 0;
    task->alltoall.started              = 0;
    task->alltoall.wait_wc              = 0;
    task->alltoall.blocks_sent          = 0;
    task->alltoall.blocks_completed     = 0;
    task->alltoall.seq_num              = a2a->sequence_number;
    task->alltoall.seq_index            = SEQ_INDEX(a2a->sequence_number);
    task->alltoall.op                   = &a2a->node.ops[
                                                     task->alltoall.seq_index];
    task->alltoall.msg_size             = msg_size;

    tl_trace(UCC_TL_TEAM_LIB(tl_team), "Seq num is %d", task->alltoall.seq_num);
    a2a->sequence_number += 1;

    block_size = a2a->requested_block_size ? a2a->requested_block_size
                                           : get_block_size(task);

    //todo following section correct assuming homogenous PPN across all nodes
    task->alltoall.num_of_blocks_columns =
        (a2a->node.sbgp->group_size % block_size)
            ? ucc_div_round_up(a2a->node.sbgp->group_size, block_size)
            : 0;
    task->alltoall.block_size = block_size;

    // TODO remove for connectX-7 - this is mkey_entry->stride (count+skip) limitation - only 16 bits
    if (task->alltoall
            .num_of_blocks_columns) { // for other case we will never reach limit - we use bytes skip only for the "leftovers" mode, which means when
                                      // num_of_blocks_columns != 0
        size_t limit =
            (1ULL
             << 16); // TODO We need to query this from device (or device type) and not user hardcoded values
        size_t bytes_count, bytes_count_last, bytes_skip, bytes_skip_last;
        int    ppn = a2a->node.sbgp->group_size;
        int    bs  = block_size;

        bytes_count_last = (ppn % bs) * msg_size;
        bytes_skip_last  = (ppn - (ppn % bs)) * msg_size;
        bytes_count      = bs * msg_size;
        bytes_skip       = (ppn - bs) * msg_size;
        if ((bytes_count + bytes_skip >= limit) ||
            (bytes_count_last + bytes_skip_last >= limit)) {
            tl_debug(UCC_TL_TEAM_LIB(tl_team), "unsupported operation");
            status = UCC_ERR_NOT_SUPPORTED;
            goto put_schedule;
        }
    }

    ucc_schedule_add_task(schedule, tasks[0]);
    ucc_event_manager_subscribe(&schedule->super, UCC_EVENT_SCHEDULE_STARTED,
                                tasks[0], ucc_task_start_handler);
    for (i = 0; i < (n_tasks - 1); i++) {
        ucc_schedule_add_task(schedule, tasks[i + 1]);
        ucc_event_manager_subscribe(tasks[i], UCC_EVENT_COMPLETED, tasks[i + 1],
                                    ucc_task_start_handler);
    }

    tasks[curr_task]->post     = ucc_tl_mlx5_poll_free_op_slot_start;
    tasks[curr_task]->progress = ucc_tl_mlx5_poll_free_op_slot_progress;
    curr_task++;

    tasks[curr_task]->post     = ucc_tl_mlx5_reg_fanin_start;
    tasks[curr_task]->progress = ucc_tl_mlx5_reg_fanin_progress;
    curr_task++;

    if (is_asr) {
        tasks[curr_task]->post     = ucc_tl_mlx5_asr_barrier_start;
        tasks[curr_task]->progress = ucc_tl_mlx5_asr_barrier_progress;
        curr_task++;
        if (task->alltoall.num_of_blocks_columns) {
            tasks[curr_task]->post = ucc_tl_mlx5_send_blocks_leftovers_start;
        } else {
            tasks[curr_task]->post = ucc_tl_mlx5_send_blocks_start;
        }
        tasks[curr_task]->progress = ucc_tl_mlx5_send_blocks_progress;
        curr_task++;
    }
    tasks[curr_task]->post     = ucc_tl_mlx5_fanout_start;
    tasks[curr_task]->progress = ucc_tl_mlx5_fanout_progress;

    schedule->super.post           = ucc_tl_mlx5_alltoall_start;
    schedule->super.progress       = NULL;
    schedule->super.finalize       = ucc_tl_mlx5_alltoall_finalize;
    schedule->super.triggered_post = NULL;

    *task_h = &schedule->super;
    return status;

put_schedule:
    ucc_tl_mlx5_put_schedule(task);
    return status;
}
