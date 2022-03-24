/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_INLINE_H_
#define UCC_TL_MLX5_INLINE_H_

#include "tl_mlx5_ib.h"
#include "tl_mlx5_coll.h"

static inline struct ibv_qp_ex*
tl_mlx5_get_qp_ex(ucc_tl_mlx5_team_t *team, ucc_rank_t rank)
{
    if (team->is_dc) {
        return team->net.dcis[rank % team->num_dci_qps].dc_qpex;
    }
    return team->net.rc_qps[rank].qp_ex;
}

static inline struct ibv_qp*
tl_mlx5_get_qp(ucc_tl_mlx5_team_t *team, ucc_rank_t rank)
{
    if (team->is_dc) {
        return team->net.dcis[rank % team->num_dci_qps].dci_qp;
    }
    return team->net.rc_qps[rank].qp;
}

static inline ucc_status_t send_block_data(ucc_tl_mlx5_team_t *team, ucc_rank_t rank,
                                           uint64_t src_addr, uint32_t msg_size,
                                           uint32_t lkey, uint64_t remote_addr, uint32_t rkey,
                                           int send_flags, ucc_tl_mlx5_dm_chunk_t *dm)
{
    struct ibv_qp *qp   = tl_mlx5_get_qp(team, rank);
    struct ibv_ah *ah   = NULL;
    uint32_t       dctn = 0;

    if (dm) {
        dm->task->blocks_sent++;
    }
    if (team->is_dc) {
        ah   = team->net.ahs[rank];
        dctn = team->net.remote_dctns[rank];
    }
    return ucc_tl_mlx5_post_rdma(qp, dctn, ah, src_addr, msg_size, lkey,
                                 remote_addr, rkey, send_flags, (uintptr_t)dm);
}

static inline void send_start(ucc_tl_mlx5_team_t *team, ucc_rank_t rank)
{
    ibv_wr_start(tl_mlx5_get_qp_ex(team, rank));
}

static inline ucc_status_t send_done(ucc_tl_mlx5_team_t *team, ucc_rank_t rank)
{
    if (ibv_wr_complete(tl_mlx5_get_qp_ex(team, rank))) {
        tl_error(UCC_TL_TEAM_LIB(team), "ibv_wr_complete failed, errno %d", errno);
        return UCC_ERR_NO_MESSAGE;
    }
    return UCC_OK;
}

static inline ucc_status_t send_atomic(ucc_tl_mlx5_team_t *team, ucc_rank_t rank,
                                       void *remote_addr, uint32_t rkey)
{
    struct ibv_qp_ex    *qp_ex;
    struct mlx5dv_qp_ex *qp_dv;

    qp_ex = tl_mlx5_get_qp_ex(team, rank);
    qp_ex->wr_flags = 0;
    ibv_wr_atomic_fetch_add(qp_ex, rkey, (uintptr_t)remote_addr, 1ULL);
    if (team->is_dc) {
        qp_dv = mlx5dv_qp_ex_from_ibv_qp_ex(qp_ex);
        mlx5dv_wr_set_dc_addr(qp_dv, team->net.ahs[rank],
                              team->net.remote_dctns[rank], DC_KEY);
    }
    ibv_wr_set_sge(qp_ex, team->dummy_bf_mr->lkey,
                   (uint64_t)team->dummy_bf_mr->addr,
                   team->dummy_bf_mr->length);
    return UCC_OK;
}

static inline void *
tl_mlx5_atomic_addr(ucc_tl_mlx5_schedule_t *task, ucc_rank_t rank)
{
    void               *remote_atomic = NULL;
#if !ATOMIC_IN_MEMIC
    ucc_tl_mlx5_team_t *team = TASK_TEAM(task);

    remote_atomic = team->net.remote_ctrl[rank].atomic.addr;
#endif
    return PTR_OFFSET(remote_atomic, task->seq_index * sizeof(tl_mlx5_atomic_t));
}

static inline uint32_t
tl_mlx5_atomic_rkey(ucc_tl_mlx5_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mlx5_team_t *team = TASK_TEAM(task);

    return team->net.remote_ctrl[rank].atomic.rkey;
}

static inline tl_mlx5_barrier_t
tl_mlx5_barrier_flag(ucc_tl_mlx5_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mlx5_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          net_size = team->net.net_size;

    return team->net.barrier.flags[(net_size + 1) * task->seq_index + rank];
}

static inline void
tl_mlx5_barrier_flag_set(ucc_tl_mlx5_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mlx5_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          net_size = team->net.net_size;

    team->net.barrier.flags[(net_size + 1) * task->seq_index + rank]
        = task->seq_num;
}

static inline tl_mlx5_barrier_t*
tl_mlx5_barrier_local_addr(ucc_tl_mlx5_schedule_t *task)
{
    ucc_tl_mlx5_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          net_size = team->net.net_size;

    return &team->net.barrier.flags[(net_size + 1) * task->seq_index + net_size];
}

static inline uintptr_t
tl_mlx5_barrier_my_remote_addr(ucc_tl_mlx5_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mlx5_team_t *team     = TASK_TEAM(task);
    ucc_rank_t          net_size = team->net.net_size;
    ucc_rank_t          net_rank = team->net.sbgp->group_rank;
    void               *remote_barrier;
    ptrdiff_t           offset;

    remote_barrier = team->net.remote_ctrl[rank].barrier.addr;
    offset = (task->seq_index * (net_size + 1) + net_rank) *
        sizeof(tl_mlx5_barrier_t);
    return (uintptr_t)PTR_OFFSET(remote_barrier, offset) ;
}

static inline uint32_t
tl_mlx5_barrier_remote_rkey(ucc_tl_mlx5_schedule_t *task, ucc_rank_t rank)
{
    ucc_tl_mlx5_team_t *team     = TASK_TEAM(task);

    return team->net.remote_ctrl[rank].barrier.rkey;
}

static inline void tl_mlx5_ah_to_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    struct mlx5dv_obj obj;
    struct mlx5dv_ah dv_ah;

    obj.ah.in  = ah;
    obj.ah.out = &dv_ah;
    mlx5dv_init_obj(&obj, MLX5DV_OBJ_AH);

    *av = *(dv_ah.av);
}
#endif
