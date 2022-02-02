/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MHBA_INLINE_H_
#define UCC_TL_MHBA_INLINE_H_

static inline void send_block_data_dc(uint64_t src_addr, uint32_t msg_size,
                                      uint32_t lkey, uint64_t remote_addr,
                                      uint32_t rkey, int send_flags,
                                      struct dci *dci_struct,
                                      uint32_t dct_number, struct ibv_ah *ah, void *dm)
{
    dci_struct->dc_qpex->wr_id    = (uint64_t)(uintptr_t)dm;
    dci_struct->dc_qpex->wr_flags = send_flags;
    ibv_wr_rdma_write(dci_struct->dc_qpex, rkey, remote_addr);
    mlx5dv_wr_set_dc_addr(dci_struct->dc_mqpex, ah, dct_number, DC_KEY);
    ibv_wr_set_sge(dci_struct->dc_qpex, lkey, src_addr, msg_size);
}

static inline ucc_status_t
send_block_data_rc(struct ibv_qp *qp, uint64_t src_addr, uint32_t msg_size,
                   uint32_t lkey, uint64_t remote_addr, uint32_t rkey,
                   int send_flags, int with_imm, void *dm)
{
    struct ibv_send_wr *bad_wr;
    struct ibv_sge      list = {
        .addr   = src_addr,
        .length = msg_size,
        .lkey   = lkey,
    };

    struct ibv_send_wr wr = {
        .wr_id      = (uint64_t)(uintptr_t)dm,
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

static inline ucc_status_t send_block_data(ucc_tl_mhba_team_t *team, ucc_rank_t rank,
                                           uint64_t src_addr, uint32_t msg_size,
                                           uint32_t lkey, uint64_t remote_addr, uint32_t rkey,
                                           int send_flags, int local /*used for sw transpose only */,
                                           void *dm)
{
    struct ibv_qp *qp;
    int            dci;

    if (!team->is_dc || local) {
        qp = local ? team->node.ns_umr_qp.qp : team->net.rc_qps[rank];
        return send_block_data_rc(qp, src_addr, msg_size, lkey, remote_addr, rkey, send_flags,
                                  local ? 1 : 0, dm);
    } else {
        dci = rank % team->num_dci_qps;
        send_block_data_dc(src_addr,  msg_size, lkey, remote_addr, rkey, send_flags,
                           &team->net.dcis[dci], team->net.remote_dctns[rank],
                           team->net.ahs[rank], dm);
    }
    return UCC_OK;
}

static inline void send_start(ucc_tl_mhba_team_t *team, ucc_rank_t rank)
{
    int dci;

    if (team->is_dc) {
        dci = rank % team->num_dci_qps;
        ibv_wr_start(team->net.dcis[dci].dc_qpex);
    }
}

static inline ucc_status_t send_done(ucc_tl_mhba_team_t *team, ucc_rank_t rank)
{
    int dci;

    if (team->is_dc) {
        dci = rank % team->num_dci_qps;
        if (ibv_wr_complete(team->net.dcis[dci].dc_qpex)) {
            return UCC_ERR_NO_MESSAGE;
        }
    }
    return UCC_OK;
}

static inline void
send_atomic_dc(uint64_t remote_addr, uint32_t rkey, struct dci *dci_struct,
               uint32_t dct_number, struct ibv_ah *ah, ucc_tl_mhba_team_t *team,
               uint64_t value)
{
    dci_struct->dc_qpex->wr_id    = value;
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
                                          uint64_t value)
{
    struct ibv_send_wr *bad_wr;
    struct ibv_sge      list = {
        .addr   = (uint64_t)team->dummy_bf_mr->addr,
        .length = team->dummy_bf_mr->length,
        .lkey   = team->dummy_bf_mr->lkey,
    };

    struct ibv_send_wr wr = {
        .wr_id                 = value,
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

static inline ucc_status_t send_atomic(ucc_tl_mhba_team_t *team, ucc_rank_t rank,
                                       void *remote_addr, uint32_t rkey,
                                       uint64_t value)
{
    int dci;

    if (!team->is_dc) {
        return send_atomic_rc(team->net.rc_qps[rank], (uintptr_t)remote_addr,
                              rkey, team, value);
    } else {
        dci = rank % team->num_dci_qps;
        send_atomic_dc((uintptr_t)remote_addr, rkey, &team->net.dcis[dci],
                       team->net.remote_dctns[rank], team->net.ahs[rank],
                       team, value);
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

static inline void *
tl_mhba_atomic_addr(ucc_tl_mhba_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mhba_team_t *team = TASK_TEAM(task);
    void               *remote_atomic;

    remote_atomic = team->net.remote_ctrl[rank].atomic.addr;
    return PTR_OFFSET(remote_atomic, task->seq_index * sizeof(tl_mhba_atomic_t));
}

static inline uint32_t
tl_mhba_atomic_rkey(ucc_tl_mhba_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mhba_team_t *team = TASK_TEAM(task);

    return team->net.remote_ctrl[rank].atomic.rkey;
}

static inline tl_mhba_barrier_t
tl_mhba_barrier_flag(ucc_tl_mhba_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mhba_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          net_size = team->net.net_size;

    return team->net.barrier.flags[(net_size + 1) * task->seq_index + rank];
}

static inline tl_mhba_barrier_t*
tl_mhba_barrier_local_addr(ucc_tl_mhba_schedule_t *task)
{
    ucc_tl_mhba_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          net_size = team->net.net_size;

    return &team->net.barrier.flags[(net_size + 1) * task->seq_index + net_size];
}

static inline uintptr_t
tl_mhba_barrier_my_remote_addr(ucc_tl_mhba_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mhba_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          net_size = team->net.net_size;
    ucc_rank_t          net_rank = team->net.sbgp->group_rank;
    void               *remote_barrier;
    ptrdiff_t           offset;

    remote_barrier = team->net.remote_ctrl[rank].barrier.addr;
    offset = (task->seq_index * (net_size + 1) + net_rank) *
        sizeof(tl_mhba_barrier_t);
    return (uintptr_t)PTR_OFFSET(remote_barrier, offset) ;
}

static inline uint32_t
tl_mhba_barrier_remote_rkey(ucc_tl_mhba_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mhba_team_t *team     = TASK_TEAM(task);

    return team->net.remote_ctrl[rank].barrier.rkey;
}

#endif
