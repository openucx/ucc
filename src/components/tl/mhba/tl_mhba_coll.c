/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "tl_mhba_coll.h"
#include "tl_mhba_mkeys.h"
#include "core/ucc_mc.h"
#include "core/ucc_team.h"


static ucc_status_t ucc_tl_mhba_node_fanin(ucc_tl_mhba_team_t *team,
                                           ucc_tl_mhba_schedule_t *task)
{
    int                 i;
    ucc_tl_mhba_ctrl_t *ctrl_v;

    if (team->op_busy[task->seq_index] && !task->started) {
        return UCC_INPROGRESS;
    } //wait for slot to be open
    team->op_busy[task->seq_index] = 1;
    task->started                  = 1;

    if (team->node.sbgp->group_rank != team->node.asr_rank) {
        ucc_tl_mhba_get_my_ctrl(team, task->seq_index)->seq_num =
            task->seq_num;
    } else {
        for (i = 0; i < team->node.sbgp->group_size; i++) {
            if (i == team->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = ucc_tl_mhba_get_ctrl(team, task->seq_index, i);
            if (ctrl_v->seq_num != task->seq_num) {
                return UCC_INPROGRESS;
            }
        }
        for (i = 0; i < team->node.sbgp->group_size; i++) {
            if (i == team->node.sbgp->group_rank) {
                continue;
            }
            ctrl_v = ucc_tl_mhba_get_ctrl(team, task->seq_index, i);
            ucc_tl_mhba_get_my_ctrl(team, task->seq_index)
                ->mkey_cache_flag |= ctrl_v->mkey_cache_flag;
        }
    }
    return UCC_OK;
}

/* Each rank registers sbuf and rbuf and place the registration data
   in the shared memory location. Next, all rank in node nitify the
   ASR the registration data is ready using SHM Fanin */
static ucc_status_t ucc_tl_mhba_reg_fanin_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task     = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *    team     = TASK_TEAM(task);
    ucc_tl_mhba_context_t *ctx = UCC_TL_MHBA_TEAM_CTX(team);
    int                     reg_change_flag               = 0;
    ucc_rcache_region_t *   send_ptr;
    ucc_rcache_region_t *   recv_ptr;

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
    task->send_rcache_region_p = ucc_tl_mhba_get_rcache_reg_data(send_ptr);

    /* NOTE: we does not support alternating block_size for the same msg size - TODO
       Will need to add the possibility of block_size change into consideration
       when initializing the mkey_cache_flag */

    ucc_tl_mhba_get_my_ctrl(team, task->seq_index)->mkey_cache_flag =
        (task->msg_size == team->previous_msg_size[task->seq_index])
            ? 0
            : (UCC_MHBA_NEED_RECV_MKEY_UPDATE | UCC_MHBA_NEED_RECV_MKEY_UPDATE);

    if (reg_change_flag || (task->send_rcache_region_p->mr->addr !=
                            team->previous_send_address[task->seq_index])) {
        ucc_tl_mhba_get_my_ctrl(team, task->seq_index)->mkey_cache_flag |=
            UCC_MHBA_NEED_SEND_MKEY_UPDATE;
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
    task->recv_rcache_region_p = ucc_tl_mhba_get_rcache_reg_data(recv_ptr);
    if (reg_change_flag || (task->recv_rcache_region_p->mr->addr !=
                            team->previous_recv_address[task->seq_index])) {
        ucc_tl_mhba_get_my_ctrl(team, task->seq_index)->mkey_cache_flag |=
            UCC_MHBA_NEED_RECV_MKEY_UPDATE;
    }

    tl_debug(UCC_TL_MHBA_TEAM_LIB(team), "fanin start");
    /* start task if completion event received */
    /* Start fanin */
    ucc_tl_mhba_update_mkeys_entries(
        &team->node, task,
        UCC_TL_MHBA_TEAM_LIB(team)); // no option for failure status
    if (UCC_OK == ucc_tl_mhba_node_fanin(team, task)) {
        tl_debug(UCC_TL_MHBA_TEAM_LIB(team), "fanin complete");
        coll_task->super.status = UCC_OK;
        return ucc_task_complete(coll_task);
    } else {
        ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    }
    return UCC_OK;
}

ucc_status_t ucc_tl_mhba_reg_fanin_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *    team    = TASK_TEAM(task);
    ucc_assert(team->node.sbgp->group_rank == team->node.asr_rank);
    if (UCC_OK == ucc_tl_mhba_node_fanin(team, task)) {
        tl_debug(UCC_TL_MHBA_TEAM_LIB(team), "fanin complete");
        coll_task->super.status = UCC_OK;
    }
    return coll_task->super.status;
}

static ucc_status_t ucc_tl_mhba_node_fanout(ucc_tl_mhba_team_t *team,
                                            ucc_tl_mhba_schedule_t *task)
{
    ucc_tl_mhba_ctrl_t *ctrl_v;
    int                 atomic_counter;

    /* First phase of fanout: asr signals it completed local ops
       and other ranks wait for asr */
    if (team->node.sbgp->group_rank == team->node.asr_rank) {
        ucc_tl_mhba_get_my_ctrl(team, task->seq_index)->seq_num =
            task->seq_num;
    } else {
        ctrl_v =
            ucc_tl_mhba_get_ctrl(team, task->seq_index, team->node.asr_rank);
        if (ctrl_v->seq_num != task->seq_num) {
            return UCC_INPROGRESS;
        }
    }
    /*Second phase of fanout: wait for remote atomic counters -
      ie wait for the remote data */
    atomic_counter = task->op->net_ctrl->atomic_counter;
    ucc_assert(atomic_counter <= team->net.net_size);

    if (atomic_counter != team->net.net_size) {
        return UCC_INPROGRESS;
    }
    return UCC_OK;
}

static ucc_status_t ucc_tl_mhba_fanout_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *team    = TASK_TEAM(task);

    coll_task->super.status              = UCC_INPROGRESS;
    tl_debug(UCC_TASK_LIB(task),"fanout start");
    /* start task if completion event received */

    /* Start fanout */
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

static ucc_status_t ucc_tl_mhba_fanout_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *team    = TASK_TEAM(task);
    if (UCC_OK == ucc_tl_mhba_node_fanout(team, task)) {
        /*Cleanup alg resources - all done */
        tl_debug(UCC_TASK_LIB(task),"Algorithm completion");
        team->op_busy[task->seq_index] = 0;
        coll_task->super.status                = UCC_OK;
    }
    return coll_task->super.status;
}

static ucc_status_t ucc_tl_mhba_asr_barrier_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *team    = TASK_TEAM(task);
    ucc_status_t status;

    coll_task->super.status              = UCC_INPROGRESS;


    // despite while statement in poll_umr_cq, non blocking because have independent cq,
    // will be finished in a finite time
    ucc_tl_mhba_populate_send_recv_mkeys(team, task);

    //Reset atomic notification counter to 0
    task->op->net_ctrl->atomic_counter = 0;

    tl_debug(UCC_TASK_LIB(task),"asr barrier start");
    status = ucc_service_allreduce(UCC_TL_CORE_TEAM(team), &task->barrier_scratch[0],
                                   &task->barrier_scratch[1], UCC_DT_INT32, 1, UCC_OP_SUM,
                                   ucc_sbgp_to_subset(team->net.sbgp), &task->barrier_req);
    if (status < 0) {
        tl_error(UCC_TASK_LIB(task), "failed to start asr barrier");
    }
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

ucc_status_t ucc_tl_mhba_asr_barrier_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_status_t status;

    status = ucc_collective_test(&task->barrier_req->task->super);
    if (status < 0) {
        tl_error(UCC_TASK_LIB(task), "failure during asr barrier");
    } else if (UCC_OK == status) {
        tl_debug(UCC_TASK_LIB(task),"asr barrier done");
        ucc_service_coll_finalize(task->barrier_req);
        coll_task->super.status = UCC_OK;
    }
    return status;
}

static inline void send_block_data_dc(uint64_t src_addr, uint32_t msg_size,
                                      uint32_t lkey, uint64_t remote_addr,
                                      uint32_t rkey, int send_flags,
                                      struct dci *dci_struct,
                                      uint32_t dct_number, struct ibv_ah *ah)
{
    dci_struct->dc_qpex->wr_id    = 1;
    dci_struct->dc_qpex->wr_flags = send_flags;
    ibv_wr_rdma_write(dci_struct->dc_qpex, rkey, remote_addr);
    mlx5dv_wr_set_dc_addr(dci_struct->dc_mqpex, ah, dct_number, DC_KEY);
    ibv_wr_set_sge(dci_struct->dc_qpex, lkey, src_addr, msg_size);
}

static inline ucc_status_t
send_block_data_rc(struct ibv_qp *qp, uint64_t src_addr, uint32_t msg_size,
                   uint32_t lkey, uint64_t remote_addr, uint32_t rkey,
                   int send_flags, int with_imm)
{
    struct ibv_send_wr *bad_wr;
    struct ibv_sge      list = {
        .addr   = src_addr,
        .length = msg_size,
        .lkey   = lkey,
    };

    struct ibv_send_wr wr = {
        .wr_id      = 12345,
        .sg_list    = &list,
        .num_sge    = 1,
        .opcode     = with_imm ? IBV_WR_RDMA_WRITE_WITH_IMM : IBV_WR_RDMA_WRITE,
        .send_flags = send_flags,
        .wr.rdma.remote_addr = remote_addr,
        .wr.rdma.rkey        = rkey,
    };

    if (ibv_post_send(qp, &wr, &bad_wr)) {
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static inline void
send_atomic_dc(uint64_t remote_addr, uint32_t rkey, struct dci *dci_struct,
               uint32_t dct_number, struct ibv_ah *ah, ucc_tl_mhba_team_t *team,
               ucc_tl_mhba_schedule_t *task)
{
    dci_struct->dc_qpex->wr_id    = task->seq_num;
    dci_struct->dc_qpex->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_atomic_fetch_add(dci_struct->dc_qpex, rkey, remote_addr, 1ULL);
    mlx5dv_wr_set_dc_addr(dci_struct->dc_mqpex, ah, dct_number, DC_KEY);
    ibv_wr_set_sge(dci_struct->dc_qpex, team->dummy_bf_mr->lkey,
                   (uint64_t)team->dummy_bf_mr->addr,
                   team->dummy_bf_mr->length);
}

static inline ucc_status_t send_atomic_rc(struct ibv_qp *qp,
                                          uint64_t remote_addr, uint32_t rkey,
                                          ucc_tl_mhba_team_t *    team,
                                          ucc_tl_mhba_schedule_t *task)
{
    struct ibv_send_wr *bad_wr;
    struct ibv_sge      list = {
        .addr   = (uint64_t)team->dummy_bf_mr->addr,
        .length = team->dummy_bf_mr->length,
        .lkey   = team->dummy_bf_mr->lkey,
    };

    struct ibv_send_wr wr = {
        .wr_id                 = task->seq_num,
        .sg_list               = &list,
        .num_sge               = 1,
        .opcode                = IBV_WR_ATOMIC_FETCH_AND_ADD,
        .send_flags            = IBV_SEND_SIGNALED,
        .wr.atomic.remote_addr = remote_addr,
        .wr.atomic.rkey        = rkey,
        .wr.atomic.compare_add = 1ULL,
    };

    if (ibv_post_send(qp, &wr, &bad_wr)) {
        tl_error(UCC_TL_MHBA_TEAM_LIB(team),"failed to post atomic send");
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static inline void tranpose_non_square_mat(void *addr, int transposed_rows_len,
                                           int transposed_columns_len,
                                           int unit_size,
										   ucc_base_lib_t *lib)
{
    void *tmp =
        ucc_malloc(transposed_rows_len * transposed_columns_len * unit_size);
    if (!tmp) {
        tl_error(lib, "malloc failed");
    }
    int i, j;
    for (i = 0; i < transposed_columns_len; i++) {
        for (j = 0; j < transposed_rows_len; j++) {
            memcpy(tmp + (unit_size * (i * transposed_rows_len + j)),
                   addr + (unit_size * ((j * transposed_columns_len) + i)),
                   unit_size);
        }
    }
    memcpy(addr, tmp, unit_size * transposed_rows_len * transposed_columns_len);
    ucc_free(tmp);
}

static inline void transpose_square_mat(void *addr, int side_len, int unit_size,
                                        void *temp_buffer)
{
    int   i, j;
    char  tmp_preallocated[TMP_TRANSPOSE_PREALLOC];
    void *tmp =
        unit_size <= TMP_TRANSPOSE_PREALLOC ? tmp_preallocated : temp_buffer;
    for (i = 0; i < side_len - 1; i++) {
        for (j = i + 1; j < side_len; j++) {
            memcpy(tmp, addr + (i * unit_size * side_len) + (j * unit_size),
                   unit_size);
            memcpy(addr + (i * unit_size * side_len) + (j * unit_size),
                   addr + (j * unit_size * side_len) + (i * unit_size),
                   unit_size);
            memcpy(addr + j * unit_size * side_len + i * unit_size, tmp,
                   unit_size);
        }
    }
}

static inline ucc_status_t prepost_dummy_recv(struct ibv_qp *qp, int num,ucc_base_lib_t *lib)
{
    struct ibv_recv_wr  wr;
    struct ibv_recv_wr *bad_wr;
    int                 i;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id   = 0;
    wr.num_sge = 0;
    for (i = 0; i < num; i++) {
        if (ibv_post_recv(qp, &wr, &bad_wr)) {
            tl_error(lib,"failed to prepost %d receives", num);
            return UCC_ERR_NO_MESSAGE;
        }
    }
    return UCC_OK;
}

// add polling mechanism for blocks in order to maintain const qp tx rx
static ucc_status_t
ucc_tl_mhba_send_blocks_start_with_transpose(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *    team    = TASK_TEAM(task);
    int node_size                   = team->node.sbgp->group_size;
    int net_size                    = team->net.sbgp->group_size;
    int op_msgsize = node_size * team->max_msg_size * UCC_TL_TEAM_SIZE(team) *
                     team->max_num_of_columns;
    int           node_msgsize  = SQUARED(node_size) * task->msg_size;
    int           block_size    = task->block_size;
    int           col_msgsize   = task->msg_size * block_size * node_size;
    int           block_msgsize = SQUARED(block_size) * task->msg_size;
    int           i, j, k, dest_rank, rank, n_compl, ret, cyc_rank, current_dci=0;
    uint64_t      src_addr, remote_addr;
    struct ibv_wc transpose_completion[1];
    ucc_status_t  status;

    coll_task->super.status              = UCC_INPROGRESS;

    tl_debug(UCC_TASK_LIB(task), "send blocks start");
    rank = team->net.rank_map[team->net.sbgp->group_rank];

    for (i = 0; i < net_size; i++) {
        cyc_rank  = (i + team->net.sbgp->group_rank) % net_size;
        dest_rank = team->net.rank_map[cyc_rank];
        //send all blocks from curr node to some ARR
        if (team->is_dc) {
            current_dci = cyc_rank % team->num_dci_qps;
        }
        for (j = 0; j < (node_size / block_size); j++) {
            for (k = 0; k < (node_size / block_size); k++) {
                src_addr    = (uintptr_t)(node_msgsize * dest_rank +
                                       col_msgsize * j + block_msgsize * k);
                remote_addr = (uintptr_t)(op_msgsize * task->seq_index +
                                          node_msgsize * rank +
                                          block_msgsize * j + col_msgsize * k);
                prepost_dummy_recv(team->node.ns_umr_qp.qp, 1,UCC_TASK_LIB(task));
                // SW Transpose
                status = send_block_data_rc(
                    team->node.ns_umr_qp.qp, src_addr, block_msgsize,
                    team->node.ops[task->seq_index].send_mkeys[0]->lkey,
                    (uintptr_t)task->transpose_buf_mr->addr,
                    task->transpose_buf_mr->rkey, IBV_SEND_SIGNALED, 1);
                if (status != UCC_OK) {
                    tl_error(
                        UCC_TASK_LIB(task),
                        "Failed sending block to transpose buffer[%d,%d,%d]", i,
                        j, k);
                    return status;
                }
                n_compl = 0;
                while (n_compl != 2) {
                    ret =
                        ibv_poll_cq(team->node.umr_cq, 1, transpose_completion);
                    if (ret > 0) {
                        if (transpose_completion[0].status != IBV_WC_SUCCESS) {
                            tl_error(UCC_TASK_LIB(task),
                                     "local copy for transpose CQ returned "
                                     "completion with status %s (%d)",
                                     ibv_wc_status_str(
                                         transpose_completion[0].status),
                                     transpose_completion[0].status);
                            return UCC_ERR_NO_MESSAGE;
                        }
                        n_compl++;
                    }
                }
                transpose_square_mat(task->transpose_buf_mr->addr,
                                     block_size, task->msg_size,
                                     task->tmp_transpose_buf);
                if (team->is_dc) {
                    ibv_wr_start(team->net.dcis[current_dci].dc_qpex);
                    send_block_data_dc(
                        (uintptr_t)task->transpose_buf_mr->addr,
                        block_msgsize, task->transpose_buf_mr->lkey,
                        remote_addr, team->net.rkeys[cyc_rank],
                        IBV_SEND_SIGNALED, &team->net.dcis[current_dci],
                        team->net.remote_dctns[cyc_rank],
                        team->net.ahs[cyc_rank]);
                    if (ibv_wr_complete(team->net.dcis[current_dci].dc_qpex)) {
                        tl_error(UCC_TASK_LIB(task),
                                 "can't post request [%d,%d,%d]", i, j, k);
                        return UCC_ERR_NO_MESSAGE;
                    }
                } else {
                    status = send_block_data_rc(
                        team->net.rc_qps[cyc_rank],
                        (uintptr_t)task->transpose_buf_mr->addr,
                        block_msgsize, task->transpose_buf_mr->lkey,
                        remote_addr, team->net.rkeys[cyc_rank],
                        IBV_SEND_SIGNALED, 0);
                    if (status != UCC_OK) {
                        tl_error(UCC_TASK_LIB(task),
                                 "Failed sending block [%d,%d,%d]", i, j, k);
                        return status;
                    }
                }
                while (!ibv_poll_cq(team->net.cq, 1, transpose_completion)) {
                }
            }
        }
    }

    for (i = 0; i < net_size; i++) {
        if (team->is_dc) {
            current_dci = i % team->num_dci_qps;
            ibv_wr_start(team->net.dcis[current_dci].dc_qpex);
            send_atomic_dc(
                (uintptr_t)team->net.remote_ctrl[i].addr +
                (task->seq_index * OP_SEGMENT_SIZE(team)),
                team->net.remote_ctrl[i].rkey, &team->net.dcis[current_dci],
                team->net.remote_dctns[i], team->net.ahs[i], team, task);
            if (ibv_wr_complete(team->net.dcis[current_dci].dc_qpex)) {
                tl_error(UCC_TASK_LIB(task), "can't post atomic request [%d]",
                         i);
                return UCC_ERR_NO_MESSAGE;
            }
        } else {
            status =
                send_atomic_rc(team->net.rc_qps[i],
                               (uintptr_t)team->net.remote_ctrl[i].addr +
                                   (task->seq_index * OP_SEGMENT_SIZE(team)),
                               team->net.remote_ctrl[i].rkey, team, task);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task),
                         "Failed sending atomic to node [%d]", i);
                return status;
            }
        }
    }
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

static ucc_status_t
ucc_tl_mhba_send_blocks_leftovers_start_with_transpose(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *team    = TASK_TEAM(task);
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
    int i, j, k, dest_rank, rank, n_compl, ret, cyc_rank, current_block_msgsize,
        current_dci=0;
    uint64_t      src_addr, remote_addr;
    struct ibv_wc transpose_completion[1];
    ucc_status_t  status;

    coll_task->super.status              = UCC_INPROGRESS;

    tl_debug(UCC_TASK_LIB(task), "send blocks start");
    rank = team->net.rank_map[team->net.sbgp->group_rank];

    for (i = 0; i < net_size; i++) {
        cyc_rank  = (i + team->net.sbgp->group_rank) % net_size;
        dest_rank = team->net.rank_map[cyc_rank];
        if (team->is_dc) {
            current_dci = cyc_rank % team->num_dci_qps;
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

                prepost_dummy_recv(team->node.ns_umr_qp.qp, 1,UCC_TASK_LIB(task));
                // SW Transpose
                status = send_block_data_rc(
                    team->node.ns_umr_qp.qp, src_addr, current_block_msgsize,
                    team->node.ops[task->seq_index].send_mkeys[j]->lkey,
                    (uintptr_t)task->transpose_buf_mr->addr,
                    task->transpose_buf_mr->rkey, IBV_SEND_SIGNALED, 1);
                if (status != UCC_OK) {
                    tl_error(
                        UCC_TASK_LIB(task),
                        "Failed sending block to transpose buffer[%d,%d,%d]", i,
                        j, k);
                    return status;
                }
                n_compl = 0;
                while (n_compl != 2) {
                    ret =
                        ibv_poll_cq(team->node.umr_cq, 1, transpose_completion);
                    if (ret > 0) {
                        if (transpose_completion[0].status != IBV_WC_SUCCESS) {
                            tl_error(UCC_TASK_LIB(task),
                                     "local copy for transpose CQ returned "
                                     "completion with status %s (%d)",
                                     ibv_wc_status_str(
                                         transpose_completion[0].status),
                                     transpose_completion[0].status);
                            return UCC_ERR_NO_MESSAGE;
                        }
                        n_compl++;
                    }
                }

                if (k != (task->num_of_blocks_columns - 1)) {
                    if (j != (task->num_of_blocks_columns - 1)) {
                        transpose_square_mat(task->transpose_buf_mr->addr,
                                             block_size, task->msg_size,
                                             task->tmp_transpose_buf);
                    } else {
                        tranpose_non_square_mat(
                            task->transpose_buf_mr->addr, block_size,
                            block_size_leftovers_side, task->msg_size, UCC_TASK_LIB(task));
                    }
                } else {
                    if (j != (task->num_of_blocks_columns - 1)) {
                        tranpose_non_square_mat(task->transpose_buf_mr->addr,
                                                block_size_leftovers_side,
                                                block_size, task->msg_size,UCC_TASK_LIB(task));
                    } else {
                        transpose_square_mat(task->transpose_buf_mr->addr,
                                             block_size_leftovers_side,
                                             task->msg_size,
                                             task->tmp_transpose_buf);
                    }
                }

                if (team->is_dc) {
                    ibv_wr_start(team->net.dcis[current_dci].dc_qpex);
                    send_block_data_dc(
                        (uintptr_t)task->transpose_buf_mr->addr,
                        current_block_msgsize, task->transpose_buf_mr->lkey,
                        remote_addr, team->net.rkeys[cyc_rank],
                        IBV_SEND_SIGNALED, &team->net.dcis[current_dci],
                        team->net.remote_dctns[cyc_rank],
                        team->net.ahs[cyc_rank]);
                    if (ibv_wr_complete(team->net.dcis[current_dci].dc_qpex)) {
                        tl_error(UCC_TASK_LIB(task),
                                 "can't post request [%d,%d,%d]", i, j, k);
                        return UCC_ERR_NO_MESSAGE;
                    }
                } else {
                    status = send_block_data_rc(
                        team->net.rc_qps[cyc_rank],
                        (uintptr_t)task->transpose_buf_mr->addr,
                        current_block_msgsize, task->transpose_buf_mr->lkey,
                        remote_addr, team->net.rkeys[cyc_rank],
                        IBV_SEND_SIGNALED, 0);
                    if (status != UCC_OK) {
                        tl_error(UCC_TASK_LIB(task),
                                 "Failed sending block [%d,%d,%d]", i, j, k);
                        return status;
                    }
                }
                while (!ibv_poll_cq(team->net.cq, 1, transpose_completion)) {
                }
            }
        }
    }

    for (i = 0; i < net_size; i++) {
        if (team->is_dc) {
            current_dci = i % team->num_dci_qps;
            ibv_wr_start(team->net.dcis[current_dci].dc_qpex);
            send_atomic_dc(
                (uintptr_t)team->net.remote_ctrl[i].addr +
                (task->seq_index * OP_SEGMENT_SIZE(team)),
                team->net.remote_ctrl[i].rkey, &team->net.dcis[current_dci],
                team->net.remote_dctns[i], team->net.ahs[i], team, task);
            if (ibv_wr_complete(team->net.dcis[current_dci].dc_qpex)) {
                tl_error(UCC_TASK_LIB(task), "can't post atomic request [%d]",
                         i);
                return UCC_ERR_NO_MESSAGE;
            }
        } else {
            status =
                send_atomic_rc(team->net.rc_qps[i],
                               (uintptr_t)team->net.remote_ctrl[i].addr +
                                   (task->seq_index * OP_SEGMENT_SIZE(team)),
                               team->net.remote_ctrl[i].rkey, team, task);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task),
                         "Failed sending atomic to node [%d]", i);
                return status;
            }
        }
    }
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

// add polling mechanism for blocks in order to maintain const qp tx rx
static ucc_status_t ucc_tl_mhba_send_blocks_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *team    = TASK_TEAM(task);
    int node_size                   = team->node.sbgp->group_size;
    int net_size                    = team->net.sbgp->group_size;
    int op_msgsize = node_size * team->max_msg_size * UCC_TL_TEAM_SIZE(team) *
                     team->max_num_of_columns;
    int          node_msgsize  = SQUARED(node_size) * task->msg_size;
    int          block_size    = task->block_size;
    int          col_msgsize   = task->msg_size * block_size * node_size;
    int          block_msgsize = SQUARED(block_size) * task->msg_size;
    int          i, j, k, dest_rank, rank, cyc_rank, current_dci = 0;
    uint64_t     src_addr, remote_addr;
    ucc_status_t status;

    coll_task->super.status              = UCC_INPROGRESS;

    tl_debug(UCC_TASK_LIB(task), "send blocks start");
    rank = team->net.rank_map[team->net.sbgp->group_rank];

    for (i = 0; i < net_size; i++) {
        cyc_rank  = (i + team->net.sbgp->group_rank) % net_size;
        dest_rank = team->net.rank_map[cyc_rank];
        if (team->is_dc) {
            current_dci = cyc_rank % team->num_dci_qps;
            ibv_wr_start(
                team->net.dcis[current_dci]
                    .dc_qpex); //todo pay attention for MT - 2 threads cant write
            // to same QP in same time
        }
        //send all blocks from curr node to some ARR
        for (j = 0; j < (node_size / block_size); j++) {
            for (k = 0; k < (node_size / block_size); k++) {
                src_addr    = (uintptr_t)(node_msgsize * dest_rank +
                                       col_msgsize * j + block_msgsize * k);
                remote_addr = (uintptr_t)(op_msgsize * task->seq_index +
                                          node_msgsize * rank +
                                          block_msgsize * j + col_msgsize * k);

                if (team->is_dc) {
                    send_block_data_dc(
                        src_addr, block_msgsize,
                        team->node.ops[task->seq_index].send_mkeys[0]->lkey,
                        remote_addr, team->net.rkeys[cyc_rank], 0,
                        &team->net.dcis[current_dci],
                        team->net.remote_dctns[cyc_rank],
                        team->net.ahs[cyc_rank]);
                } else {
                    /* printf("rank %d, dest %d, block [%d:%d], block_msgsize %d\n", */
                           /* rank, dest_rank, j, k, block_msgsize); */
                    status = send_block_data_rc(
                        team->net.rc_qps[cyc_rank], src_addr, block_msgsize,
                        team->node.ops[task->seq_index].send_mkeys[0]->lkey,
                        remote_addr, team->net.rkeys[cyc_rank], 0, 0);
                    if (status != UCC_OK) {
                        tl_error(UCC_TASK_LIB(task),
                                 "Failed sending block [%d,%d,%d]", i, j, k);
                        return status;
                    }
                }
            }
        }
        if (team->is_dc) {
            send_atomic_dc((uintptr_t)team->net.remote_ctrl[cyc_rank].addr +
                               (task->seq_index * OP_SEGMENT_SIZE(team)),
                           team->net.remote_ctrl[cyc_rank].rkey,
                           &team->net.dcis[current_dci],
                           team->net.remote_dctns[cyc_rank],
                           team->net.ahs[cyc_rank], team, task);
            if (ibv_wr_complete(team->net.dcis[current_dci].dc_qpex)) {
                tl_error(UCC_TASK_LIB(task), "can't post requests [%d]", i);
                return UCC_ERR_NO_MESSAGE;
            }
        } else {
            status = send_atomic_rc(
                team->net.rc_qps[cyc_rank],
                (uintptr_t)team->net.remote_ctrl[cyc_rank].addr +
                    (task->seq_index * OP_SEGMENT_SIZE(team)),
                team->net.remote_ctrl[cyc_rank].rkey, team, task);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task),
                         "Failed sending atomic to node [%d]", i);
                return status;
            }
        }
    }
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

static ucc_status_t
ucc_tl_mhba_send_blocks_leftovers_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *    team    = TASK_TEAM(task);
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
    int i, j, k, dest_rank, rank, cyc_rank, current_block_msgsize, current_dci=0;
    uint64_t     src_addr, remote_addr;
    ucc_status_t status;

    coll_task->super.status              = UCC_INPROGRESS;

    tl_debug(UCC_TASK_LIB(task), "send blocks start");
    rank = team->net.rank_map[team->net.sbgp->group_rank];

    for (i = 0; i < net_size; i++) {
        cyc_rank  = (i + team->net.sbgp->group_rank) % net_size;
        dest_rank = team->net.rank_map[cyc_rank];
        if (team->is_dc) {
            current_dci = cyc_rank % team->num_dci_qps;
            ibv_wr_start(team->net.dcis[current_dci].dc_qpex);
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

                if (team->is_dc) {
                    send_block_data_dc(
                        src_addr, current_block_msgsize,
                        team->node.ops[task->seq_index].send_mkeys[j]->lkey,
                        remote_addr, team->net.rkeys[cyc_rank], 0,
                        &team->net.dcis[current_dci],
                        team->net.remote_dctns[cyc_rank],
                        team->net.ahs[cyc_rank]);
                } else {
                    status = send_block_data_rc(
                        team->net.rc_qps[cyc_rank], src_addr,
                        current_block_msgsize,
                        team->node.ops[task->seq_index].send_mkeys[j]->lkey,
                        remote_addr, team->net.rkeys[cyc_rank], 0, 0);
                    if (status != UCC_OK) {
                        tl_error(UCC_TASK_LIB(task),
                                 "Failed sending block [%d,%d,%d]", i, j, k);
                        return status;
                    }
                }
            }
        }
        if (team->is_dc) {
            send_atomic_dc((uintptr_t)team->net.remote_ctrl[cyc_rank].addr +
                               (task->seq_index * OP_SEGMENT_SIZE(team)),
                           team->net.remote_ctrl[cyc_rank].rkey,
                           &team->net.dcis[current_dci],
                           team->net.remote_dctns[cyc_rank],
                           team->net.ahs[cyc_rank], team, task);
            if (ibv_wr_complete(team->net.dcis[current_dci].dc_qpex)) {
                tl_error(UCC_TASK_LIB(task), "can't post requests [%d]", i);
                return UCC_ERR_NO_MESSAGE;
            }
        } else {
            status = send_atomic_rc(
                team->net.rc_qps[cyc_rank],
                (uintptr_t)team->net.remote_ctrl[cyc_rank].addr +
                    (task->seq_index * OP_SEGMENT_SIZE(team)),
                team->net.remote_ctrl[cyc_rank].rkey, team, task);
            if (status != UCC_OK) {
                tl_error(UCC_TASK_LIB(task),
                         "Failed sending atomic to node [%d]", i);
                return status;
            }
        }
    }
    ucc_progress_enqueue(UCC_TL_CORE_CTX(team)->pq, coll_task);
    return UCC_OK;
}

ucc_status_t ucc_tl_mhba_send_blocks_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = TASK_SCHEDULE(coll_task);
    ucc_tl_mhba_team_t *    team    = TASK_TEAM(task);
    int                     i, completions_num;
    completions_num = ibv_poll_cq(team->net.cq, team->net.sbgp->group_size,
                                  team->work_completion);
    if (completions_num < 0) {
        tl_error(UCC_TASK_LIB(task),
                 "ibv_poll_cq() failed for RDMA_ATOMIC execution");
        return UCC_ERR_NO_MESSAGE;
    }
    for (i = 0; i < completions_num; i++) {
        if (team->work_completion[i].status != IBV_WC_SUCCESS) {
            tl_error(UCC_TASK_LIB(task),
                     "bad work completion status %s, wr_id %zu",
                     ibv_wc_status_str(team->work_completion[i].status),
                     team->work_completion[i].wr_id);
            return UCC_ERR_NO_MESSAGE;
        }
        team->cq_completions[SEQ_INDEX(team->work_completion[i].wr_id)] += 1;
    }
    if (team->cq_completions[task->seq_index] ==
        team->net.sbgp->group_size) {
        team->cq_completions[task->seq_index] = 0;
        coll_task->super.status                       = UCC_OK;
    }
    return coll_task->super.status;
}

ucc_status_t ucc_tl_mhba_task_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_task_t *task = ucc_derived_of(coll_task, ucc_tl_mhba_task_t);

    tl_trace(UCC_TASK_LIB(task), "finalizing task %p", task);
    ucc_tl_mhba_put_task(task);
    return UCC_OK;
}

static inline ucc_tl_mhba_task_t *
ucc_tl_mhba_init_task(ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
                      ucc_schedule_t *schedule)
{
    ucc_tl_mhba_task_t *task    = ucc_tl_mhba_get_task(coll_args, team);

    task->super.schedule = schedule;
    task->super.finalize = ucc_tl_mhba_task_finalize;
    return task;
}

ucc_status_t ucc_tl_mhba_alltoall_start(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = ucc_derived_of(coll_task,
                                                     ucc_tl_mhba_schedule_t);

    UCC_TL_MHBA_PROFILE_REQUEST_EVENT(task, "mhba_alltoall_start", 0);
    return ucc_schedule_start(&task->super);
}

ucc_status_t ucc_tl_mhba_alltoall_finalize(ucc_coll_task_t *coll_task)
{
    ucc_tl_mhba_schedule_t *task    = ucc_derived_of(coll_task,
                                                     ucc_tl_mhba_schedule_t);
    ucc_tl_mhba_team_t *    team    = TASK_TEAM(task);
    ucc_tl_mhba_context_t *ctx = TASK_CTX(task);
    ucc_status_t        status;

    team->previous_msg_size[task->seq_index] = task->msg_size;
    team->previous_send_address[task->seq_index] =
        task->send_rcache_region_p->mr->addr;
    team->previous_recv_address[task->seq_index] =
        task->recv_rcache_region_p->mr->addr;
    ucc_rcache_region_put(ctx->rcache, task->send_rcache_region_p->region);
    ucc_rcache_region_put(ctx->rcache, task->recv_rcache_region_p->region);
    if (team->transpose) {
        ucc_free(task->tmp_transpose_buf);
        if (task->transpose_buf_mr != team->transpose_buf_mr) {
            ibv_dereg_mr(task->transpose_buf_mr);
            ucc_free(task->transpose_buf_mr->addr);
        }
    }

    UCC_TL_MHBA_PROFILE_REQUEST_EVENT(schedule, "mhba_alltoall_done", 0);
    status = ucc_schedule_finalize(coll_task);
    ucc_tl_mhba_put_schedule(task);
    return status;
}

ucc_status_t ucc_tl_mhba_alltoall_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *     team,
                                       ucc_coll_task_t **    task_h)
{
    ucc_tl_mhba_team_t *    tl_team = ucc_derived_of(team, ucc_tl_mhba_team_t);
    ucc_tl_mhba_schedule_t *task = ucc_tl_mhba_get_schedule(tl_team, coll_args);
    ucc_schedule_t *        schedule    = &task->super;
    ucc_tl_mhba_context_t *ctx = UCC_TL_MHBA_TEAM_CTX(tl_team);
    int              block_msgsize, block_size;
    void *           transpose_buf = NULL;
    ucc_coll_task_t *tasks[4];
    int is_asr = (tl_team->node.sbgp->group_rank == tl_team->node.asr_rank);
    ucc_status_t status = UCC_OK;

    if (UCC_IS_INPLACE(coll_args->args)) {
        status = UCC_ERR_NOT_SUPPORTED;
        goto put_schedule;
    }

    int i, n_tasks = is_asr ? 4 : 2, curr_task = 0;
    for (i = 0; i < n_tasks; i++) {
        tasks[i] = &ucc_tl_mhba_init_task(coll_args, team, schedule)->super;
    }

    task->started   = 0;
    task->seq_num   = tl_team->sequence_number;
    task->seq_index = SEQ_INDEX(tl_team->sequence_number);
    task->op        = &tl_team->node.ops[task->seq_index];
    task->msg_size =
        (size_t)(coll_args->args.src.info.count / UCC_TL_TEAM_SIZE(tl_team)) *
        ucc_dt_size(coll_args->args.src.info.datatype);

    tl_debug(UCC_TL_TEAM_LIB(tl_team), "Seq num is %d", task->seq_num);
    tl_team->sequence_number += 1;
    if (task->msg_size > tl_team->max_msg_size) {
        tl_error(UCC_TL_TEAM_LIB(tl_team), "msg size too long");
        status = UCC_ERR_NO_RESOURCE;
        goto put_schedule;
    }

    block_size =
        (task->msg_size != 1)
            ? tl_team->blocks_sizes[__ucs_ilog2_u32(task->msg_size - 1) + 1]
            : tl_team->blocks_sizes[0];
    block_size = tl_team->requested_block_size ? tl_team->requested_block_size
                                               : block_size;

    //todo following section correct assuming homogenous PPN across all nodes
    task->num_of_blocks_columns =
        (tl_team->node.sbgp->group_size % block_size)
            ? ucc_div_round_up(tl_team->node.sbgp->group_size, block_size)
            : 0;
    block_msgsize = SQUARED(block_size) * task->msg_size;
    if (((tl_team->net.sbgp->group_rank == 0) && is_asr) &&
        (task->msg_size != tl_team->previous_msg_size[task->seq_index])) {
        tl_info(UCC_TL_TEAM_LIB(tl_team), "Block size is %d msg_size is %zu",
                block_size, task->msg_size);
    }

    task->block_size        = block_size;
    task->transpose_buf_mr  = NULL;
    task->tmp_transpose_buf = NULL;

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

    if (is_asr) {
        if (tl_team->transpose) {
            if (UCC_TL_MHBA_TEAM_LIB(tl_team)->cfg.transpose_buf_size >=
                block_msgsize) {
                task->transpose_buf_mr = tl_team->transpose_buf_mr;
            } else {
                transpose_buf = ucc_malloc(block_msgsize);
                if (!transpose_buf) {
                    tl_error(UCC_TL_TEAM_LIB(tl_team),
                             "failed to allocate transpose buffer of %d bytes",
                             block_msgsize);
                    status = UCC_ERR_NO_MEMORY;
                    goto put_schedule;
                }
                task->transpose_buf_mr = ibv_reg_mr(
                    ctx->shared_pd, transpose_buf, block_msgsize,
                    IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE);
                if (!task->transpose_buf_mr) {
                    tl_error(UCC_TL_TEAM_LIB(tl_team),
                             "failed to register transpose buffer, errno %d",
                             errno);
                    status = UCC_ERR_NO_MESSAGE;
                    goto free_transpose;
                }
            }
            task->tmp_transpose_buf = NULL;
            if (task->msg_size > TMP_TRANSPOSE_PREALLOC) {
                task->tmp_transpose_buf = ucc_malloc(task->msg_size);
                if (!task->tmp_transpose_buf) {
                    tl_error(
                        UCC_TL_TEAM_LIB(tl_team),
                        "failed to allocate tmp transpose buffer of %zu bytes",
                        task->msg_size);
                    status = UCC_ERR_NO_MEMORY;
                    goto dereg_transpose;
                }
            }
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

    tasks[curr_task]->post     = ucc_tl_mhba_reg_fanin_start;
    tasks[curr_task]->progress = ucc_tl_mhba_reg_fanin_progress;
    curr_task++;

    if (is_asr) {
        tasks[curr_task]->post     = ucc_tl_mhba_asr_barrier_start;
        tasks[curr_task]->progress = ucc_tl_mhba_asr_barrier_progress;
        curr_task++;
        if (tl_team->transpose) {
            if (task->num_of_blocks_columns) {
                tasks[curr_task]->post =
                    ucc_tl_mhba_send_blocks_leftovers_start_with_transpose;
            } else {
                tasks[curr_task]->post =
                    ucc_tl_mhba_send_blocks_start_with_transpose;
            }
        } else {
            if (task->num_of_blocks_columns) {
                tasks[curr_task]->post =
                    ucc_tl_mhba_send_blocks_leftovers_start;
            } else {
                tasks[curr_task]->post = ucc_tl_mhba_send_blocks_start;
            }
        }
        tasks[curr_task]->progress = ucc_tl_mhba_send_blocks_progress;
        curr_task++;
    }
    tasks[curr_task]->post     = ucc_tl_mhba_fanout_start;
    tasks[curr_task]->progress = ucc_tl_mhba_fanout_progress;

    schedule->super.post           = ucc_tl_mhba_alltoall_start;
    schedule->super.progress       = NULL;
    schedule->super.finalize       = ucc_tl_mhba_alltoall_finalize;
    schedule->super.triggered_post = NULL;

    *task_h                        = &schedule->super;
    return status;

dereg_transpose:
    if (is_asr) {
        ibv_dereg_mr(task->transpose_buf_mr);
    }
free_transpose:
    if (is_asr) {
        ucc_free(transpose_buf);
    }

put_schedule:
    ucc_tl_mhba_put_schedule(task);
    return status;
}


ucc_status_t ucc_tl_mhba_team_get_scores(ucc_base_team_t    *tl_team,
                                         ucc_coll_score_t **score_p)
{
    ucc_tl_mhba_team_t *team  = ucc_derived_of(tl_team, ucc_tl_mhba_team_t);
    ucc_base_lib_t     *lib   = UCC_TL_TEAM_LIB(team);
    ucc_coll_score_t   *score;
    ucc_status_t        status;

    status = ucc_coll_score_alloc(&score);
    if (UCC_OK != status) {
        tl_error(lib, "failed to alloc score_t");
        return status;
    }


    status = ucc_coll_score_add_range(
        score, UCC_COLL_TYPE_ALLTOALL, UCC_MEMORY_TYPE_HOST, 0, 4096,
        UCC_TL_MHBA_DEFAULT_SCORE, ucc_tl_mhba_alltoall_init, tl_team);
    if (UCC_OK != status) {
        tl_error(lib, "faild to add range to score_t");
        return status;
    }


    if (strlen(lib->score_str) > 0) {
        status = ucc_coll_score_update_from_str(
            lib->score_str, score, UCC_TL_TEAM_SIZE(team),
            NULL, tl_team, UCC_TL_MHBA_DEFAULT_SCORE, NULL);

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

ucc_status_t ucc_tl_mhba_coll_init(ucc_base_coll_args_t *coll_args,
                                   ucc_base_team_t *     team,
                                   ucc_coll_task_t **    task)
{
    return UCC_OK;
}
