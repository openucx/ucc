/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLTOALL_H_
#define ALLTOALL_H_

#include "tl_mlx5.h"
#include "tl_mlx5_ib.h"
#include "tl_mlx5_dm.h"

#define SEQ_INDEX(_seq_num) ((_seq_num) % MAX_OUTSTANDING_OPS)

#define MLX5_ASR_RANK                0 // has to be 0 with current implementation of UCC_SBGP_NODE_LEADERS
#define MLX5_NUM_OF_BLOCKS_SIZE_BINS 8
#define MAX_TRANSPOSE_SIZE           8192 // HW transpose unit is limited to matrix size
#define MAX_MSG_SIZE                 128 // HW transpose unit is limited to element size
#define MAX_BLOCK_SIZE               64 // from limit of Transpose unit capabilities
#define MAX_OUTSTANDING_OPS          1 //todo change - according to limitations (52 top)
#define MIN_POLL_WC                  8

typedef uint64_t tl_mlx5_atomic_t;
typedef uint64_t tl_mlx5_barrier_t;

enum
{
    UCC_MLX5_NEED_SEND_MKEY_UPDATE = UCC_BIT(1),
    UCC_MLX5_NEED_RECV_MKEY_UPDATE = UCC_BIT(2),
};

typedef struct ucc_tl_mlx5_alltoall_ctrl {
    union {
        struct {
            volatile int seq_num;
            int          mkey_cache_flag;
        };
        char tmp[UCC_CACHE_LINE_SIZE];
    };
} ucc_tl_mlx5_alltoall_ctrl_t;

typedef struct ucc_tl_mlx5_alltoall_op {
    ucc_tl_mlx5_alltoall_ctrl_t *ctrl;
    ucc_tl_mlx5_alltoall_ctrl_t *my_ctrl;
    struct mlx5dv_mkey **   send_mkeys;
    struct mlx5dv_mkey **   recv_mkeys;
    int *                   blocks_sent;
} ucc_tl_mlx5_alltoall_op_t;

/* This structure holds resources and data related to the "in-node"
   part of the algorithm. */
typedef struct ucc_tl_mlx5_alltoall_node {
    int                       asr_rank;
    ucc_sbgp_t               *sbgp;
    void                     *storage;
    ucc_tl_mlx5_alltoall_op_t ops[MAX_OUTSTANDING_OPS];
    struct mlx5dv_mkey       *team_recv_mkey;
    void                     *umr_entries_buf;
    struct ibv_mr            *umr_entries_mr;
} ucc_tl_mlx5_alltoall_node_t;

typedef struct alltoall_net_ctrl {
    struct {
        void    *addr;
        uint32_t rkey;
    } atomic;
    struct {
        void    *addr;
        uint32_t rkey;
    } barrier;
} alltoall_net_ctrl_t;

typedef struct ucc_tl_mlx5_alltoall_net {
    ucc_sbgp_t       *sbgp;
    int               net_size;
    int              *rank_map;
    ucc_tl_mlx5_qp_t *rc_qps;
    struct ibv_qp    *dct_qp;
    struct ibv_srq   *srq;
    uint32_t         *remote_dctns;
    struct ibv_ah   **ahs;
    struct ibv_cq    *cq;
    struct ibv_cq    *umr_cq;
    struct ibv_qp    *umr_qp;
    struct ibv_mr    *ctrl_mr;
    struct {
#if ATOMIC_IN_MEMIC
        struct ibv_dm *counters;
#else
        tl_mlx5_atomic_t *counters;
#endif
        struct ibv_mr *mr;
    } atomic;
    struct {
        tl_mlx5_barrier_t *flags;
        struct ibv_mr     *mr;
    } barrier;
    int                 *blocks_sent;
    alltoall_net_ctrl_t *remote_ctrl;
    uint32_t            *rkeys;
    ucc_tl_mlx5_dci_t   *dcis;
} ucc_tl_mlx5_alltoall_net_t;

typedef struct ucc_tl_mlx5_a2a_bcast_data {
    int shmid;
    int net_size;
} ucc_tl_mlx5_a2a_bcast_data_t;

enum
{
    TL_MLX5_ALLTOALL_STATE_SHMID,
    TL_MLX5_ALLTOALL_STATE_EXCHANGE_PROGRESS,
    TL_MLX5_ALLTOALL_STATE_EXCHANGE_DONE
};

typedef struct ucc_tl_mlx5_alltoall {
    struct ibv_pd               *pd;
    struct ibv_context          *ctx;
    int                          ib_port;
    int                          state;
    uint64_t                     max_msg_size;
    ucc_tl_mlx5_alltoall_node_t  node;
    ucc_tl_mlx5_alltoall_net_t   net;
    int                          sequence_number;
    int                          op_busy[MAX_OUTSTANDING_OPS];
    int                          num_dci_qps;
    uint8_t                      is_dc;
    int                          previous_msg_size[MAX_OUTSTANDING_OPS];
    void                        *previous_send_address[MAX_OUTSTANDING_OPS];
    void                        *previous_recv_address[MAX_OUTSTANDING_OPS];
    uint64_t                     atomic_scratch_bf;
    int                          requested_block_size;
    int                          max_num_of_columns;
    struct ibv_mr               *atomic_scratch_bf_mr;
    ucc_rank_t                   node_size;
    ucc_tl_mlx5_a2a_bcast_data_t bcast_data;
} ucc_tl_mlx5_alltoall_t;

void         ucc_tl_mlx5_topo_cleanup(ucc_tl_mlx5_team_t *team);
ucc_status_t ucc_tl_mlx5_team_init_alltoall(ucc_tl_mlx5_team_t *team);
ucc_status_t ucc_tl_mlx5_team_test_alltoall_start(ucc_tl_mlx5_team_t *team);
ucc_status_t ucc_tl_mlx5_team_test_alltoall_progress(ucc_tl_mlx5_team_t *team);
ucc_status_t ucc_tl_mlx5_alltoall_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t *     team,
                                       ucc_coll_task_t **    task_h);
void         ucc_tl_mlx5_alltoall_cleanup(ucc_tl_mlx5_team_t *team);

static inline ucc_tl_mlx5_alltoall_ctrl_t*
ucc_tl_mlx5_get_ctrl(ucc_tl_mlx5_alltoall_t *a2a, int op_index, int rank)
{
    ucc_tl_mlx5_alltoall_ctrl_t *ctrl =
        PTR_OFFSET(a2a->node.ops[op_index].ctrl,
                   sizeof(ucc_tl_mlx5_alltoall_ctrl_t) * rank);
    return ctrl;
}

static inline ucc_tl_mlx5_alltoall_ctrl_t*
ucc_tl_mlx5_get_my_ctrl(ucc_tl_mlx5_alltoall_t *a2a, int op_index)
{
    int my_rank = a2a->node.sbgp->group_rank;

    return ucc_tl_mlx5_get_ctrl(a2a, op_index, my_rank);
}

#define OP_SEGMENT_SIZE(_a2a)                                                  \
    (sizeof(ucc_tl_mlx5_alltoall_ctrl_t) * (_a2a)->node_size +                 \
     (sizeof(umr_t) * (_a2a)->max_num_of_columns * (_a2a)->node_size) * 2)

#define UMR_DATA_OFFSET(_a2a)                                                  \
    (sizeof(ucc_tl_mlx5_alltoall_ctrl_t) * (_a2a)->node_size)

#define OP_SEGMENT_STORAGE(_req, _a2a)                                         \
    PTR_OFFSET((_a2a)->node.storage,                                           \
               OP_SEGMENT_SIZE(_a2a) * (_req)->alltoall.seq_index)

#define OP_UMR_DATA(_req, _a2a)                                                \
    PTR_OFFSET(OP_SEGMENT_STORAGE(_req, _a2a), UMR_DATA_OFFSET(_a2a))

#define SEND_UMR_DATA(_req, _a2a, _col)                                        \
    PTR_OFFSET(OP_UMR_DATA(_req, _a2a),                                        \
               _col *(_a2a)->node.sbgp->group_size * sizeof(umr_t))

#define RECV_UMR_DATA(_req, _a2a, _col)                                        \
    PTR_OFFSET(OP_UMR_DATA(_req, _a2a),                                        \
               (_a2a)->max_num_of_columns *(_a2a)->node.sbgp->group_size *     \
                       sizeof(umr_t) +                                         \
                   _col * (_a2a)->node.sbgp->group_size * sizeof(umr_t))

#define MY_SEND_UMR_DATA(_req, _a2a, _col)                                     \
    PTR_OFFSET(SEND_UMR_DATA(_req, _a2a, _col),                                \
               (_a2a)->node.sbgp->group_rank * sizeof(umr_t))

#define MY_RECV_UMR_DATA(_req, _a2a, _col)                                     \
    PTR_OFFSET(RECV_UMR_DATA(_req, _a2a, _col),                                \
               (_a2a)->node.sbgp->group_rank * sizeof(umr_t))

#endif
