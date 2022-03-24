/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_coll.h"
#include "tl_mlx5_mkeys.h"
#include "components/mc/ucc_mc.h"
#include "core/ucc_team.h"
#include "tl_mlx5_inline.h"
#include "tl_mlx5_ib.h"

static ucc_status_t ucc_tl_mlx5_poll_cq(ucc_tl_mlx5_team_t *team,
                                        struct ibv_cq *cq)
{
    int                     i, completions_num;

    completions_num = ibv_poll_cq(cq,
                                  ucc_min(team->net.sbgp->group_size, MIN_POLL_WC),
                                  team->work_completion);
    if (completions_num < 0) {
        tl_error(UCC_TL_TEAM_LIB(team),
                 "ibv_poll_cq() failed, errno %d", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    for (i = 0; i < completions_num; i++) {
        /* printf("got completion wr_id %zu, opcode %d\n", */
        /*        team->work_completion[i].wr_id, */
        /*        team->work_completion[i].opcode); */
        if (team->work_completion[i].status != IBV_WC_SUCCESS) {
            tl_error(UCC_TL_TEAM_LIB(team),
                     "bad work completion status %s, wr_id %zu",
                     ibv_wc_status_str(team->work_completion[i].status),
                     team->work_completion[i].wr_id);
            return UCC_ERR_NO_MESSAGE;
        }

        ucc_assert(team->work_completion[i].opcode == IBV_WC_DRIVER2);
        if (team->work_completion[i].wr_id == 0) {
            /* signalled transpose */
            continue;
        } else if (team->work_completion[i].wr_id & 0x1) {
            ucc_tl_mlx5_schedule_t *task = (ucc_tl_mlx5_schedule_t *)
                (uintptr_t)(team->work_completion[i].wr_id & (~(uint64_t)0x1));
            /* printf("wait on data completion, task %p\n", task); */
            task->wait_wc = 1;
        } else {
            ucc_tl_mlx5_dm_chunk_t *dm = (ucc_tl_mlx5_dm_chunk_t *)
                team->work_completion[i].wr_id;
            dm->task->blocks_completed++;
            /* printf("returning dm %p to pool\n", (void*)team->work_completion[i].wr_id); */
            ucc_mpool_put(dm);
        }
    }
    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_node_fanin(ucc_tl_mlx5_team_t *team,
                                           ucc_tl_mlx5_schedule_t *task)
{
    int                 i;
    ucc_tl_mlx5_ctrl_t *ctrl_v;

    if (team->op_busy[task->seq_index] && !task->started) {
        return UCC_INPROGRESS;
    } //wait for slot to be open
    team->op_busy[task->seq_index] = 1;
    task->started                  = 1;

    if (team->node.sbgp->group_rank != team->node.asr_rank) {
        ucc_tl_mlx5_get_my_ctrl(team, task->seq_index)->seq_num =
            task->seq_num;
    } else {
        for (i = 0; i < team->node.sbgp->group_size; i++) {
            if (i == team->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = ucc_tl_mlx5_get_ctrl(team, task->seq_index, i);
            if (ctrl_v->seq_num != task->seq_num) {
                return UCC_INPROGRESS;
            }
        }
        for (i = 0; i < team->node.sbgp->group_size; i++) {
            if (i == team->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = ucc_tl_mlx5_get_ctrl(team, task->seq_index, i);
            ucc_tl_mlx5_get_my_ctrl(team, task->seq_index)
                ->mkey_cache_flag |= ctrl_v->mkey_cache_flag;
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
    ucc_tl_mlx5_schedule_t *task     = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t *    team     = TASK_TEAM(task);
    ucc_tl_mlx5_context_t *ctx = UCC_TL_MLX5_TEAM_CTX(team);
    int                     reg_change_flag               = 0;
    int                     flag;
    ucc_rcache_region_t *   send_ptr;
    ucc_rcache_region_t *   recv_ptr;

    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanin_start", 0);
    tl_debug(UCC_TASK_LIB(task),"register memory buffers");
    coll_task->super.status = UCC_INPROGRESS;

    if (UCC_OK != ucc_rcache_get_arg(ctx->rcache,
                                 (void *)TASK_ARGS(task).src.info.buffer,
                                 task->msg_size * UCC_TL_TEAM_SIZE(team),
                                 &reg_change_flag, &send_ptr)) {
        tl_error(UCC_TASK_LIB(task),
                 "Failed to register send_bf memory (errno=%d)", errno);
        return UCC_ERR_NO_RESOURCE;
    }
    task->send_rcache_region_p = ucc_tl_mlx5_get_rcache_reg_data(send_ptr);

    /* NOTE: we does not support alternating block_size for the same msg size - TODO
       Will need to add the possibility of block_size change into consideration
       when initializing the mkey_cache_flag */

    flag = (task->msg_size == team->previous_msg_size[task->seq_index])
            ? 0
            : (UCC_MLX5_NEED_RECV_MKEY_UPDATE | UCC_MLX5_NEED_RECV_MKEY_UPDATE);

    if (reg_change_flag || (task->send_rcache_region_p->mr->addr !=
                            team->previous_send_address[task->seq_index])) {
        flag |= UCC_MLX5_NEED_SEND_MKEY_UPDATE;
    }
    reg_change_flag = 0;
    if (UCC_OK != ucc_rcache_get_arg(ctx->rcache,
                                 (void *)TASK_ARGS(task).dst.info.buffer,
                                 task->msg_size * UCC_TL_TEAM_SIZE(team),
                                 &reg_change_flag, &recv_ptr)) {
        tl_error(UCC_TASK_LIB(task), "Failed to register receive_bf memory");
        ucc_rcache_region_put(ctx->rcache,
                              task->send_rcache_region_p->region);
        return UCC_ERR_NO_RESOURCE;
    }
    task->recv_rcache_region_p = ucc_tl_mlx5_get_rcache_reg_data(recv_ptr);
    if (reg_change_flag || (task->recv_rcache_region_p->mr->addr !=
                            team->previous_recv_address[task->seq_index])) {
        flag |= UCC_MLX5_NEED_RECV_MKEY_UPDATE;
    }

    tl_debug(UCC_TL_MLX5_TEAM_LIB(team), "fanin start");
    /* start task if completion event received */
    /* Start fanin */
    ucc_tl_mlx5_get_my_ctrl(team, task->seq_index)->mkey_cache_flag = flag;
    ucc_tl_mlx5_update_mkeys_entries(
        &team->node, task, flag,
        UCC_TL_MLX5_TEAM_LIB(team)); // no option for failure status
    if (UCC_OK == ucc_tl_mlx5_node_fanin(team, task)) {
        tl_debug(UCC_TL_MLX5_TEAM_LIB(team), "fanin complete");
        coll_task->super.status = UCC_OK;
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanin_done", 0);
        return ucc_task_complete(coll_task);
    } else {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_reg_fanin_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t *    team    = TASK_TEAM(task);
    ucc_assert(team->node.sbgp->group_rank == team->node.asr_rank);
    if (UCC_OK == ucc_tl_mlx5_node_fanin(team, task)) {
        tl_debug(UCC_TL_MLX5_TEAM_LIB(team), "fanin complete");
        coll_task->super.status = UCC_OK;
    }
    return coll_task->super.status;
}

static ucc_status_t ucc_tl_mlx5_node_fanout(ucc_tl_mlx5_team_t *team,
                                            ucc_tl_mlx5_schedule_t *task)
{
    ucc_tl_mlx5_ctrl_t *ctrl_v;
    /* tl_mlx5_atomic_t   atomic_counter; */

    /* First phase of fanout: asr signals it completed local ops
       and other ranks wait for asr */
    if (team->node.sbgp->group_rank == team->node.asr_rank) {
#if 0
        /* ASR waits for atomic replies from other ASRs */
        atomic_counter = team->net.atomic.counters[task->seq_index];
        ucc_assert(atomic_counter <= team->net.net_size);

        if (atomic_counter != team->net.net_size) {
            return UCC_INPROGRESS;
        }
#endif /* no need to check counter - we wait on data in device */
        ucc_tl_mlx5_get_my_ctrl(team, task->seq_index)->seq_num =
            task->seq_num;
    } else {
        ctrl_v =
            ucc_tl_mlx5_get_ctrl(team, task->seq_index, team->node.asr_rank);
        if (ctrl_v->seq_num != task->seq_num) {
            return UCC_INPROGRESS;
        }
    }
    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_fanout_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t *team    = TASK_TEAM(task);

    coll_task->super.status              = UCC_INPROGRESS;
    tl_debug(UCC_TASK_LIB(task),"fanout start");
    /* start task if completion event received */
    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanout_start", 0);
    /* Start fanout */
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

static ucc_status_t ucc_tl_mlx5_fanout_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t *team    = TASK_TEAM(task);
    ucc_status_t status;

    if (team->node.sbgp->group_rank == team->node.asr_rank) {
        status = ucc_tl_mlx5_poll_cq(team, team->net.umr_cq);
        if (UCC_OK != status) {
            coll_task->super.status = status;
            return status;
        }
        if (!task->wait_wc) {
            return UCC_INPROGRESS;
        }
    }

    if (UCC_OK == ucc_tl_mlx5_node_fanout(team, task)) {
        /*Cleanup alg resources - all done */
        tl_debug(UCC_TASK_LIB(task),"Algorithm completion");
        team->op_busy[task->seq_index] = 0;
        coll_task->super.status                = UCC_OK;
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_fanout_done", 0);
    }
    return coll_task->super.status;
}

static ucc_status_t ucc_tl_mlx5_asr_barrier_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t *team    = TASK_TEAM(task);
    ucc_status_t status;
    int i;
    coll_task->super.status              = UCC_INPROGRESS;

    task->started = 0;
    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_barreir_start", 0);
    // despite while statement in poll_umr_cq, non blocking because have independent cq,
    // will be finished in a finite time
    ucc_tl_mlx5_populate_send_recv_mkeys(team, task);

    //Reset atomic notification counter to 0
#if ATOMIC_IN_MEMIC
    tl_mlx5_atomic_t zero = 0;
    if (0 != ibv_memcpy_to_dm(team->net.atomic.counters, task->seq_index * sizeof(tl_mlx5_atomic_t),
                              &zero, sizeof(tl_mlx5_atomic_t))) {
        tl_error(UCC_TASK_LIB(task), "failed to reset atomic in memic");
        return UCC_ERR_NO_MESSAGE;
    }
#else
    team->net.atomic.counters[task->seq_index] = 0;
#endif
    if (UCC_TL_MLX5_TEAM_LIB(team)->cfg.asr_barrier) {
        tl_debug(UCC_TASK_LIB(task),"asr barrier start");
        status = ucc_service_allreduce(UCC_TL_CORE_TEAM(team), &task->barrier_scratch[0],
                                       &task->barrier_scratch[1], UCC_DT_INT32, 1, UCC_OP_SUM,
                                       ucc_sbgp_to_subset(team->net.sbgp), &task->barrier_req);
        if (status < 0) {
            tl_error(UCC_TASK_LIB(task), "failed to start asr barrier");
        }
        for(i=0; i<team->net.net_size;i++) {
            task->op->blocks_sent[i] = 0;
            team->net.barrier.flags[(team->net.net_size + 1) * task->seq_index + i] =
                task->seq_num;
        }
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    } else {
        tl_mlx5_barrier_t *local = tl_mlx5_barrier_local_addr(task);
        *local = task->seq_num;
        for(i=0; i<team->net.net_size;i++) {
            task->op->blocks_sent[i] = 0;
            if (i == team->net.sbgp->group_rank) {
                tl_mlx5_barrier_flag_set(task, i);
                continue;
            }

            send_start(team, i);
            status = send_block_data(team, i, (uintptr_t)local,
                                     sizeof(tl_mlx5_barrier_t), team->net.barrier.mr->lkey,
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
        coll_task->super.status = UCC_OK;
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_barreir_done", 0);
        return ucc_task_complete(coll_task);
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_mlx5_asr_barrier_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_status_t status;

    status = ucc_collective_test(&task->barrier_req->task->super);
    if (status < 0) {
        tl_error(UCC_TASK_LIB(task), "failure during asr barrier");
    } else if (UCC_OK == status) {
        tl_debug(UCC_TASK_LIB(task),"asr barrier done");
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_barreir_done", 0);
        ucc_service_coll_finalize(task->barrier_req);
        coll_task->super.status = UCC_OK;
    }
    return status;
}

// add polling mechanism for blocks in order to maintain const qp tx rx
static ucc_status_t ucc_tl_mlx5_send_blocks_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t *team    = TASK_TEAM(task);
    int node_size                   = team->node.sbgp->group_size;
    int net_size                    = team->net.sbgp->group_size;
    int op_msgsize = node_size * team->max_msg_size * UCC_TL_TEAM_SIZE(team) *
                     team->max_num_of_columns;
    int          node_msgsize  = SQUARED(node_size) * task->msg_size;
    int          block_size    = task->block_size;
    int          col_msgsize   = task->msg_size * block_size * node_size;
    int          block_msgsize = SQUARED(block_size) * task->msg_size;
    int    dm_host             = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host;
    ucc_status_t status = UCC_OK;
    int          i, j, k, rank, dest_rank, cyc_rank;
    uint64_t     src_addr, remote_addr;
    ucc_tl_mlx5_dm_chunk_t *dm;
    uintptr_t dm_addr;

    coll_task->super.status              = UCC_INPROGRESS;

    tl_debug(UCC_TASK_LIB(task), "send blocks start");
    rank = team->net.rank_map[team->net.sbgp->group_rank];
    if (!task->send_blocks_enqueued) {
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_block_send_start", 0);
    }

    for (i = 0; i < net_size; i++) {
        cyc_rank  = (i + team->net.sbgp->group_rank) % net_size;
        dest_rank = team->net.rank_map[cyc_rank];
        if (task->op->blocks_sent[cyc_rank] ||
            tl_mlx5_barrier_flag(task, cyc_rank) != task->seq_num) {
            continue;
        }

        //send all blocks from curr node to some ARR
        for (j = 0; j < (node_size / block_size); j++) {
            for (k = 0; k < (node_size / block_size); k++) {
                src_addr    = (uintptr_t)(node_msgsize * dest_rank +
                                       col_msgsize * j + block_msgsize * k);
                remote_addr = (uintptr_t)(op_msgsize * task->seq_index +
                                          node_msgsize * rank +
                                          block_msgsize * j + col_msgsize * k);

                send_start(team, cyc_rank);
                if (cyc_rank == team->net.sbgp->group_rank) {
                    status = ucc_tl_mlx5_post_transpose(tl_mlx5_get_qp(team, cyc_rank),
                                                        team->node.ops[task->seq_index].send_mkeys[0]->lkey,
                                                        team->net.rkeys[cyc_rank], src_addr, remote_addr, task->msg_size,
                                                        block_size, block_size, (j ==0 && k == 0) ? IBV_SEND_SIGNALED : 0);
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

                        status = ucc_tl_mlx5_poll_cq(team, team->net.cq);
                        if (UCC_OK != status) {
                            return status;
                        }
                        dm = ucc_mpool_get(&team->dm_pool);
                        send_start(team, cyc_rank);
                    }
                    if (dm_host) {
                        dm_addr = (uintptr_t)PTR_OFFSET(team->dm_ptr, dm->offset);
                    } else {
                        dm_addr = dm->offset; // dm reg mr 0 based
                    }
                    dm->task = task;

                    status = ucc_tl_mlx5_post_transpose(tl_mlx5_get_qp(team, cyc_rank),
                                                        team->node.ops[task->seq_index].send_mkeys[0]->lkey,
                                                        team->dm_mr->rkey, src_addr, dm_addr, task->msg_size,
                                                        block_size, block_size, 0);
                    if (UCC_OK != status) {
                        return status;
                    }
                    status = send_block_data(team, cyc_rank, dm_addr,
                                             block_msgsize, team->dm_mr->lkey,
                                             remote_addr, team->net.rkeys[cyc_rank],
                                             IBV_SEND_SIGNALED, dm);
                    if (status != UCC_OK) {
                        tl_error(UCC_TASK_LIB(task),
                                 "Failed sending block [%d,%d,%d]", i, j, k);
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
        status = send_atomic(team, cyc_rank, tl_mlx5_atomic_addr(task, cyc_rank),
                             tl_mlx5_atomic_rkey(task, cyc_rank));

        if (UCC_OK == status) {
            status = send_done(team, cyc_rank);
        }
        task->op->blocks_sent[cyc_rank]  = 1;
        task->started++;
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task),
                     "Failed sending atomic to node [%d]", cyc_rank);
            return status;
        }
    }
    if (!task->send_blocks_enqueued) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
        task->send_blocks_enqueued = 1;
    }

    if (task->started == team->net.net_size) {
        status = ucc_tl_mlx5_post_wait_on_data(team->net.umr_qp, team->net.net_size,
                                               team->net.atomic.mr->lkey, (uintptr_t)
#if ATOMIC_IN_MEMIC
                                      PTR_OFFSET(0,
#else
                                      PTR_OFFSET(team->net.atomic.counters,
#endif
                                            task->seq_index * sizeof(tl_mlx5_atomic_t)),
                                       task);
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_block_send_done", 0);
    }
    return status;
}

static ucc_status_t
ucc_tl_mlx5_send_blocks_leftovers_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t *    team    = TASK_TEAM(task);
    int node_size                   = team->node.sbgp->group_size;
    int net_size                    = team->net.sbgp->group_size;
    int op_msgsize = node_size * team->max_msg_size * UCC_TL_TEAM_SIZE(team) *
                     team->max_num_of_columns;
    int mkey_msgsize  = node_size * team->max_msg_size * UCC_TL_TEAM_SIZE(team);
    int block_size    = task->block_size;
    int col_msgsize   = task->msg_size * block_size * node_size;
    int block_msgsize = SQUARED(block_size) * task->msg_size;
    int block_size_leftovers_side = node_size % task->block_size;
    int col_msgsize_leftovers =
        task->msg_size * block_size_leftovers_side * node_size;
    int block_msgsize_leftovers =
        block_size_leftovers_side * block_size * task->msg_size;
    int corner_msgsize = SQUARED(block_size_leftovers_side) * task->msg_size;
    int    dm_host             = UCC_TL_MLX5_TEAM_LIB(team)->cfg.dm_host;
    ucc_status_t status = UCC_OK;
    int i, j, k, dest_rank, rank, cyc_rank, current_block_msgsize, bs_x, bs_y;
    uint64_t     src_addr, remote_addr;
    ucc_tl_mlx5_dm_chunk_t *dm;
    uintptr_t dm_addr;

    coll_task->super.status              = UCC_INPROGRESS;
    tl_debug(UCC_TASK_LIB(task), "send blocks start");
    rank = team->net.rank_map[team->net.sbgp->group_rank];

    for (i = 0; i < net_size; i++) {
        cyc_rank  = (i + team->net.sbgp->group_rank) % net_size;
        dest_rank = team->net.rank_map[cyc_rank];
        if (task->op->blocks_sent[cyc_rank] ||
            tl_mlx5_barrier_flag(task, cyc_rank) != task->seq_num) {
            continue;
        }
        //send all blocks from curr node to some ARR
        for (j = 0; j < task->num_of_blocks_columns; j++) {
            for (k = 0; k < task->num_of_blocks_columns; k++) {
                if (j != (task->num_of_blocks_columns - 1)) {
                    src_addr = (uintptr_t)(col_msgsize * dest_rank +
                                           block_msgsize * k);
                } else {
                    src_addr = (uintptr_t)(col_msgsize_leftovers * dest_rank +
                                           block_msgsize_leftovers * k);
                }
                if (k != (task->num_of_blocks_columns - 1)) {
                    remote_addr = (uintptr_t)(
                        op_msgsize * task->seq_index + col_msgsize * rank +
                        block_msgsize * j + mkey_msgsize * k);
                    current_block_msgsize =
                        (j != (task->num_of_blocks_columns - 1))
                            ? block_msgsize
                            : block_msgsize_leftovers;
                } else {
                    remote_addr = (uintptr_t)(op_msgsize * task->seq_index +
                                              col_msgsize_leftovers * rank +
                                              block_msgsize_leftovers * j +
                                              mkey_msgsize * k);
                    current_block_msgsize =
                        (j != (task->num_of_blocks_columns - 1))
                            ? block_msgsize_leftovers
                            : corner_msgsize;
                }
                bs_x = k < task->num_of_blocks_columns - 1 ? block_size : block_size_leftovers_side;
                bs_y = j < task->num_of_blocks_columns - 1 ? block_size : block_size_leftovers_side;

                send_start(team, cyc_rank);

                //todo : start/end for RC ?
                if (bs_x == 1 || bs_y == 1) {
                    status = send_block_data(team, cyc_rank, src_addr, current_block_msgsize,
                                             team->node.ops[task->seq_index].send_mkeys[j]->lkey,
                                             remote_addr, team->net.rkeys[cyc_rank], 0, NULL);
                } else {
                    dm = ucc_mpool_get(&team->dm_pool);
                    while (!dm) {
                        status = send_done(team, cyc_rank);
                        if (UCC_OK != status) {
                            return status;
                        }

                        status = ucc_tl_mlx5_poll_cq(team, team->net.cq);
                        if (UCC_OK != status) {
                            return status;
                        }
                        dm = ucc_mpool_get(&team->dm_pool);
                        send_start(team, cyc_rank);
                    }
                    if (dm_host) {
                        dm_addr = (uintptr_t)PTR_OFFSET(team->dm_ptr, dm->offset);
                    } else {
                        dm_addr = dm->offset; // dm reg mr 0 based
                    }
                    dm->task = task;

                    status = ucc_tl_mlx5_post_transpose(tl_mlx5_get_qp(team, cyc_rank),
                                                        team->node.ops[task->seq_index].send_mkeys[j]->lkey,
                                                        team->dm_mr->rkey, src_addr, dm_addr, task->msg_size,
                                                        bs_x, bs_y, 0);
                    if (UCC_OK != status) {
                        return status;
                    }

                    status = send_block_data(team, cyc_rank, dm_addr, current_block_msgsize,
                                             team->dm_mr->lkey,
                                             remote_addr, team->net.rkeys[cyc_rank],
                                             IBV_SEND_SIGNALED, dm);
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
        status = send_atomic(team, cyc_rank, tl_mlx5_atomic_addr(task, cyc_rank),
                             tl_mlx5_atomic_rkey(task, cyc_rank));
        if (UCC_OK == status) {
            status = send_done(team, cyc_rank);
        }
        task->op->blocks_sent[cyc_rank]  = 1;
        task->started++;
        if (status != UCC_OK) {
            tl_error(UCC_TASK_LIB(task),
                     "Failed sending atomic to node [%d]", cyc_rank);
            return status;
        }
    }
    if (!task->send_blocks_enqueued) {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
        task->send_blocks_enqueued = 1;
    }

    if (task->started == team->net.net_size) {
        status = ucc_tl_mlx5_post_wait_on_data(team->net.umr_qp, team->net.net_size,
                                               team->net.atomic.mr->lkey, (uintptr_t)
#if ATOMIC_IN_MEMIC
                                      PTR_OFFSET(0,
#else
                                     PTR_OFFSET(team->net.atomic.counters,
#endif
                                            task->seq_index * sizeof(tl_mlx5_atomic_t)),
                                      task);
    }

    return status;
}

ucc_status_t ucc_tl_mlx5_send_blocks_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mlx5_team_t *    team    = TASK_TEAM(task);
    ucc_status_t            status;


    if (task->started != team->net.net_size) {
        return coll_task->post(coll_task);
    }
    status = ucc_tl_mlx5_poll_cq(team, team->net.cq);
    if (UCC_OK != status) {
        return status;
    }

    if (task->blocks_sent == task->blocks_completed) {
        UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_all_blocks_completed", 0);
        coll_task->super.status                       = UCC_OK;
    }
    return coll_task->super.status;
}

ucc_status_t ucc_tl_mlx5_task_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_task_t *task = ucc_derived_of(coll_task, ucc_tl_mlx5_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_mlx5_put_task(task);
    return UCC_OK;
}

static inline ucc_tl_mlx5_task_t *
ucc_tl_mlx5_init_task(ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                      ucc_schedule_t *schedule)
{
    ucc_tl_mlx5_task_t *task    = ucc_tl_mlx5_get_task(coll_args, team);

    task->super.schedule = schedule;
    task->super.finalize = ucc_tl_mlx5_task_finalize;
    return task;
}

ucc_status_t ucc_tl_mlx5_alltoall_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = ucc_derived_of(coll_task,
                                                     ucc_tl_mlx5_schedule_t);

    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(task, "mlx5_alltoall_start", 0);
    return ucc_schedule_start(&task->super);
}

ucc_status_t ucc_tl_mlx5_alltoall_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_mlx5_schedule_t *task    = ucc_derived_of(coll_task,
                                                     ucc_tl_mlx5_schedule_t);
    ucc_tl_mlx5_team_t *    team    = TASK_TEAM(task);
    ucc_tl_mlx5_context_t *ctx = TASK_CTX(task);
    ucc_status_t        status;

    team->previous_msg_size[task->seq_index] = task->msg_size;
    team->previous_send_address[task->seq_index] =
        task->send_rcache_region_p->mr->addr;
    team->previous_recv_address[task->seq_index] =
        task->recv_rcache_region_p->mr->addr;
    ucc_rcache_region_put(ctx->rcache, task->send_rcache_region_p->region);
    ucc_rcache_region_put(ctx->rcache, task->recv_rcache_region_p->region);
    UCC_TL_MLX5_PROFILE_REQUEST_EVENT(coll_task, "mlx5_alltoall_done", 0);
    status = ucc_schedule_finalize(coll_task);
    ucc_tl_mlx5_put_schedule(task);
    return status;
}

static inline int power2(int value) {
    int p = 2;

    while (p < value) {
        p *= 2;
    }
    return p;
}

static inline int block_size_fits(size_t msgsize, int block_size)
{
    size_t t1    = power2(ucc_max(msgsize, 8));
    size_t tsize = block_size * ucc_max(power2(block_size) * t1, MAX_MSG_SIZE);

    return tsize <= MAX_TRANSPOSE_SIZE;
}

static inline int get_block_size(ucc_tl_mlx5_schedule_t *task)
{
    ucc_tl_mlx5_team_t *team = TASK_TEAM(task);
    int                 ppn  = team->node.sbgp->group_size;
    int                 block_size;

    block_size = ppn;
    while (!block_size_fits(task->msg_size, block_size)) {
        block_size--;
    }
    return block_size;
}

UCC_TL_MLX5_PROFILE_FUNC(ucc_status_t, ucc_tl_mlx5_alltoall_init,
                         (coll_args, team, task_h),
                         ucc_base_coll_args_t *coll_args,
                         ucc_base_team_t *     team,
                         ucc_coll_task_t **    task_h)
{
    ucc_tl_mlx5_team_t *    tl_team = ucc_derived_of(team, ucc_tl_mlx5_team_t);
    int is_asr = (tl_team->node.sbgp->group_rank == tl_team->node.asr_rank);
    ucc_status_t status = UCC_OK;
    int i, n_tasks = is_asr ? 4 : 2, curr_task = 0;
    ucc_schedule_t *        schedule;
    ucc_tl_mlx5_schedule_t *task;
    size_t           msg_size;
    int              block_size;
    ucc_coll_task_t *tasks[4];

    if (UCC_IS_INPLACE(coll_args->args)) {
        return UCC_ERR_NOT_SUPPORTED;
    }

    msg_size = coll_args->args.src.info.count / UCC_TL_TEAM_SIZE(tl_team) *
        ucc_dt_size(coll_args->args.src.info.datatype);

    if (msg_size > tl_team->max_msg_size) {
        tl_debug(UCC_TL_TEAM_LIB(tl_team), "msg size too long");
        return UCC_ERR_NOT_SUPPORTED;
    }

    task = ucc_tl_mlx5_get_schedule(tl_team, coll_args);
    schedule    = &task->super;

    for (i = 0; i < n_tasks; i++) {
        tasks[i] = &ucc_tl_mlx5_init_task(coll_args, team, schedule)->super;
    }

    task->send_blocks_enqueued = 0;
    task->started              = 0;
    task->wait_wc              = 0;
    task->blocks_sent          = 0;
    task->blocks_completed     = 0;
    task->seq_num   = tl_team->sequence_number;
    task->seq_index = SEQ_INDEX(tl_team->sequence_number);
    task->op        = &tl_team->node.ops[task->seq_index];
    task->msg_size  = msg_size;

    tl_debug(UCC_TL_TEAM_LIB(tl_team), "Seq num is %d", task->seq_num);
    tl_team->sequence_number += 1;

    block_size = tl_team->requested_block_size ? tl_team->requested_block_size
        : get_block_size(task);

    //todo following section correct assuming homogenous PPN across all nodes
    task->num_of_blocks_columns =
        (tl_team->node.sbgp->group_size % block_size)
            ? ucc_div_round_up(tl_team->node.sbgp->group_size, block_size)
            : 0;
    if (((tl_team->net.sbgp->group_rank == 0) && is_asr) &&
        (task->msg_size != tl_team->previous_msg_size[task->seq_index])) {
        tl_info(UCC_TL_TEAM_LIB(tl_team), "Block size is %d msg_size is %zu",
                block_size, task->msg_size);
    }

    task->block_size        = block_size;

    // TODO remove for connectX-7 - this is mkey_entry->stride (count+skip) limitation - only 16 bits
    if (task->num_of_blocks_columns) { // for other case we will never reach limit - we use bytes skip only for the "leftovers" mode, which means when
                                      // num_of_blocks_columns != 0
        size_t limit =
            (1ULL
             << 16); // TODO We need to query this from device (or device type) and not user hardcoded values
        size_t bytes_count, bytes_count_last, bytes_skip, bytes_skip_last;
        int    ppn = tl_team->node.sbgp->group_size;
        int    bs  = task->block_size;

        bytes_count_last = (ppn % bs) * task->msg_size;
        bytes_skip_last  = (ppn - (ppn % bs)) * task->msg_size;
        bytes_count      = bs * task->msg_size;
        bytes_skip       = (ppn - bs) * task->msg_size;
        if ((bytes_count + bytes_skip >= limit) ||
            (bytes_count_last + bytes_skip_last >= limit)) {
            tl_error(UCC_TL_TEAM_LIB(tl_team),
                     "Currently can't support this operation in connectX-6");
            status = UCC_ERR_NO_MESSAGE;
            goto put_schedule;
        }
    }

    ucc_schedule_add_task(schedule, tasks[0]);
    ucc_event_manager_subscribe(&schedule->super.em, UCC_EVENT_SCHEDULE_STARTED,
                                tasks[0], ucc_task_start_handler);
    for (i = 0; i < (n_tasks - 1); i++) {
        ucc_schedule_add_task(schedule, tasks[i + 1]);
        ucc_event_manager_subscribe(&tasks[i]->em, UCC_EVENT_COMPLETED,
                                    tasks[i + 1], ucc_task_start_handler);
    }

    tasks[curr_task]->post     = ucc_tl_mlx5_reg_fanin_start;
    tasks[curr_task]->progress = ucc_tl_mlx5_reg_fanin_progress;
    curr_task++;

    if (is_asr) {
        tasks[curr_task]->post     = ucc_tl_mlx5_asr_barrier_start;
        tasks[curr_task]->progress = ucc_tl_mlx5_asr_barrier_progress;
        curr_task++;
        if (task->num_of_blocks_columns) {
            tasks[curr_task]->post =
                ucc_tl_mlx5_send_blocks_leftovers_start;
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

    *task_h                        = &schedule->super;
    return status;

put_schedule:
    ucc_tl_mlx5_put_schedule(task);
    return status;
}


ucc_status_t ucc_tl_mlx5_team_get_scores(ucc_base_team_t    *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_mlx5_team_t *team  = ucc_derived_of(tl_team, ucc_tl_mlx5_team_t);
    ucc_base_context_t *ctx   = UCC_TL_TEAM_CTX(team);
    ucc_base_lib_t     *lib   = UCC_TL_TEAM_LIB(team);
    ucc_coll_score_t   *score;
    ucc_status_t        status;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score_t");
        return status;
    }


    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_HOST, 0,
        MAX_MSG_SIZE * UCC_TL_TEAM_SIZE(team),
        UCC_TL_MLX5_DEFAULT_SCORE, ucc_tl_mlx5_alltoall_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        return status;
    }


    if (strlen(ctx->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            ctx->score_str, score, UCC_TL_TEAM_SIZE(team),
            NULL, tl_team, UCC_TL_MLX5_DEFAULT_SCORE, NULL);

        /* If INVALID_PARAM - User provided incorrect input - try to proceed */
        if ((status < 0) && (status != UCC_ERR_INVALID_PARAM) &&
            (status != UCC_ERR_NOT_SUPPORTED)) {
            goto err;
        }
    }
    *score_p = score;
    return UCC_OK;
err:
    ucc_coll_score_free(score);
    *score_p = NULL;
    return status;
}

ucc_status_t ucc_tl_mlx5_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     team,
                                   ucc_coll_task_t **    task)
{
    return UCC_OK;
}
