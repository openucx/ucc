/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_H_
#define UCC_TL_MLX5_H_
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "core/ucc_service_coll.h"
#include "utils/ucc_mpool.h"
#include "utils/ucc_rcache.h"
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
#include "utils/arch/cpu.h"

#ifndef UCC_TL_MLX5_DEFAULT_SCORE
#define UCC_TL_MLX5_DEFAULT_SCORE 1
#endif

#ifdef HAVE_PROFILING_TL_MLX5
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define MLX5_ASR_RANK                0
#define MLX5_NUM_OF_BLOCKS_SIZE_BINS 8
#define MAX_TRANSPOSE_SIZE           8192 // HW transpose unit is limited to matrix size
#define MAX_MSG_SIZE                 128 // HW transpose unit is limited to element size
#define MAX_BLOCK_SIZE               64 // from limit of Transpose unit capabilities
#define RC_DC_LIMIT                  128
#define DC_KEY                       1
#define MAX_OUTSTANDING_OPS 1 //todo change - according to limitations (52 top)
#define MIN_POLL_WC 8

#define UCC_TL_MLX5_PROFILE_FUNC          UCC_PROFILE_FUNC
#define UCC_TL_MLX5_PROFILE_FUNC_VOID     UCC_PROFILE_FUNC_VOID
#define UCC_TL_MLX5_PROFILE_REQUEST_NEW   UCC_PROFILE_REQUEST_NEW
#define UCC_TL_MLX5_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_MLX5_PROFILE_REQUEST_FREE  UCC_PROFILE_REQUEST_FREE

typedef struct ucc_tl_mlx5_iface {
    ucc_tl_iface_t super;
} ucc_tl_mlx5_iface_t;
/* Extern iface should follow the pattern: ucc_tl_<tl_name> */
extern ucc_tl_mlx5_iface_t ucc_tl_mlx5;

typedef struct ucc_tl_mlx5_lib_config {
    ucc_tl_lib_config_t super;
    int    transpose;
    int    asr_barrier;
    size_t transpose_buf_size;
    int    block_size;
    int    num_dci_qps;
    int    rc_dc;
    size_t dm_buf_size;
    size_t dm_buf_num;
    int    dm_host;
} ucc_tl_mlx5_lib_config_t;

typedef struct ucc_tl_mlx5_context_config {
    ucc_tl_context_config_t  super;
    ucs_config_names_array_t devices;
} ucc_tl_mlx5_context_config_t;

typedef struct ucc_tl_mlx5_lib {
    ucc_tl_lib_t             super;
    ucc_tl_mlx5_lib_config_t cfg;
} ucc_tl_mlx5_lib_t;
UCC_CLASS_DECLARE(ucc_tl_mlx5_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_mlx5_context {
    ucc_tl_context_t             super;
    ucc_tl_mlx5_context_config_t cfg;
    struct ibv_context *         ib_ctx;
    struct ibv_pd *              ib_pd;
    struct ibv_context *shared_ctx;
    struct ibv_pd *     shared_pd;
    ucc_rcache_t *           rcache;
    int                 is_imported;
    int                          ib_port;
    ucc_mpool_t                  req_mp;
} ucc_tl_mlx5_context_t;
UCC_CLASS_DECLARE(ucc_tl_mlx5_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_mlx5_task ucc_tl_mlx5_task_t;

typedef struct ucc_tl_mlx5_ctrl {
    union {
        struct {
            volatile int seq_num;
            int mkey_cache_flag;
        };
        char tmp[UCC_CACHE_LINE_SIZE];
    };
} ucc_tl_mlx5_ctrl_t;

typedef uint64_t tl_mlx5_atomic_t;
typedef uint64_t tl_mlx5_barrier_t;

typedef struct mlx5dv_mr_interleaved umr_t;

typedef struct ucc_tl_mlx5_op {
    ucc_tl_mlx5_ctrl_t *ctrl;
    ucc_tl_mlx5_ctrl_t *my_ctrl;
    struct mlx5dv_mkey **send_mkeys;
    struct mlx5dv_mkey **recv_mkeys;
    int                *blocks_sent;
} ucc_tl_mlx5_op_t;


/* This structure holds resources and data related to the "in-node"
   part of the algorithm. */
typedef struct ucc_tl_mlx5_node {
    int                 asr_rank;
    ucc_sbgp_t *        sbgp;
    void *              storage;
    ucc_tl_mlx5_op_t       ops[MAX_OUTSTANDING_OPS];
    struct mlx5dv_mkey *team_recv_mkey;
    void *umr_entries_buf;
    struct ibv_mr *umr_entries_mr;
} ucc_tl_mlx5_node_t;

typedef struct ucc_tl_mlx5_reg {
    struct ibv_mr *      mr;
    ucs_rcache_region_t *region;
} ucc_tl_mlx5_reg_t;

static inline ucc_tl_mlx5_reg_t *
ucc_tl_mlx5_get_rcache_reg_data(ucc_rcache_region_t *region)
{
	return (ucc_tl_mlx5_reg_t *)((ptrdiff_t)region + sizeof(ucc_rcache_region_t));
}

typedef struct net_ctrl {
    struct {
        void     *addr;
        uint32_t rkey;
    } atomic;
    struct {
        void     *addr;
        uint32_t rkey;
    } barrier;
} net_ctrl_t;

typedef struct ucc_tl_mlx5_net {
    ucc_sbgp_t *    sbgp;
    int             net_size;
    int *           rank_map;
    struct ibv_qp **rc_qps;
    struct ibv_qp * dct_qp;
    struct ibv_srq *srq;
    uint32_t *      remote_dctns;
    struct ibv_ah **ahs;
    struct ibv_cq * cq;
    struct ibv_cq *umr_cq;
    struct ibv_qp *umr_qp;
    struct ibv_mr * ctrl_mr;
    struct {
        tl_mlx5_atomic_t *counters;
        struct ibv_mr    *mr;
    } atomic;
    struct {
        tl_mlx5_barrier_t *flags;
        struct ibv_mr     *mr;
    } barrier;
    int        *blocks_sent; // net_size * MAX_OP_OUTSTANDING
    net_ctrl_t *remote_ctrl;
    uint32_t *rkeys;
    struct dci {
        struct ibv_qp *      dci_qp;
        struct ibv_qp_ex *   dc_qpex;
        struct mlx5dv_qp_ex *dc_mqpex;
    } * dcis;
} ucc_tl_mlx5_net_t;

typedef struct ucc_tl_mlx5_bcast_data {
    int  shmid;
    int  net_size;
    char sock_path[L_tmpnam];
} ucc_tl_mlx5_bcast_data_t;

enum {
    TL_MLX5_TEAM_STATE_SHMID,
    TL_MLX5_TEAM_STATE_EXCHANGE,
};

typedef struct ucc_tl_mlx5_dm_chunk_t {
    ptrdiff_t offset; // 0 based offset from the beginning of
                      // memic_mr (obtained with ibv_reg_dm_mr)
} ucc_tl_mlx5_dm_chunk_t;

typedef struct ucc_tl_mlx5_team {
    ucc_tl_team_t            super;
    int                      state;
    int                      transpose;
    uint64_t                 max_msg_size;
    ucc_tl_mlx5_node_t       node;
    ucc_tl_mlx5_net_t        net;
    void *                   service_bcast_tmp_buf;
    int                      sequence_number;
    int                      op_busy[MAX_OUTSTANDING_OPS];
    int                      cq_completions[MAX_OUTSTANDING_OPS];
    int                      blocks_sizes[MLX5_NUM_OF_BLOCKS_SIZE_BINS];
    int                      num_dci_qps;
    uint8_t                  is_dc;
    int                      previous_msg_size[MAX_OUTSTANDING_OPS];
    void *                   previous_send_address[MAX_OUTSTANDING_OPS];
    void *                   previous_recv_address[MAX_OUTSTANDING_OPS];
    uint64_t                 dummy_atomic_buff;
    int                      requested_block_size;
    int                      max_num_of_columns;
    struct ibv_mr *          dummy_bf_mr;
    struct ibv_wc *          work_completion;
    void *                   transpose_buf;
    struct ibv_mr *          transpose_buf_mr;
    ucc_service_coll_req_t  *scoll_req;
    void *                   oob_req;
    ucc_tl_mlx5_bcast_data_t bcast_data;
    ucc_status_t             status;
    ucc_rank_t               node_size;
    ucc_mpool_t              dm_pool;
    struct ibv_dm           *dm_ptr;
    struct ibv_mr *          dm_mr;
} ucc_tl_mlx5_team_t;
UCC_CLASS_DECLARE(ucc_tl_mlx5_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_TL_MLX5_SUPPORTED_COLLS (UCC_COLL_TYPE_ALLTOALL)

#define UCC_TL_MLX5_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_mlx5_lib_t))

#define UCC_TL_MLX5_TEAM_CTX(_team)                                            \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_mlx5_context_t))

#define UCC_TL_CTX_HAS_OOB(_ctx)                                               \
    ((_ctx)->super.super.ucc_context->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB)

#define UCC_TL_CTX_OOB(_ctx) ((_ctx)->super.super.ucc_context->params.oob)

#define UCC_TL_CTX_LIB(_ctx)                                                   \
    (ucc_derived_of((_ctx)->super.super.lib, ucc_tl_mlx5_lib_t))

#define IS_SERVICE_TEAM(_team)                                                 \
    ((_team)->super.super.params.scope == UCC_CL_LAST + 1)


#define SEQ_INDEX(_seq_num) ((_seq_num) % MAX_OUTSTANDING_OPS)
#define SQUARED(_num)       ((_num) * (_num))

enum
{
    UCC_MLX5_NEED_SEND_MKEY_UPDATE = UCC_BIT(1),
    UCC_MLX5_NEED_RECV_MKEY_UPDATE = UCC_BIT(2),
};

ucc_status_t tl_mlx5_create_rcache(ucc_tl_mlx5_context_t *ctx);

ucc_status_t ucc_tl_mlx5_asr_socket_init(ucc_tl_mlx5_context_t *ctx, ucc_rank_t group_size,
                                         int *socket, const char *sock_path);

static inline ucc_tl_mlx5_ctrl_t *ucc_tl_mlx5_get_ctrl(ucc_tl_mlx5_team_t *team,
                                                       int op_index, int rank)
{
	ucc_tl_mlx5_ctrl_t * ctrl =  PTR_OFFSET(team->node.ops[op_index].ctrl,
                                            sizeof(ucc_tl_mlx5_ctrl_t) * rank);
	return ctrl;
}

static inline ucc_tl_mlx5_ctrl_t *ucc_tl_mlx5_get_my_ctrl(ucc_tl_mlx5_team_t *team,
                                                          int op_index)
{
    int my_rank = team->node.sbgp->group_rank;
    return ucc_tl_mlx5_get_ctrl(team, op_index, my_rank);
}


#define OP_SEGMENT_SIZE(_team) \
    ( sizeof(ucc_tl_mlx5_ctrl_t) * (_team)->node_size +                  \
     (sizeof(umr_t) * (_team)->max_num_of_columns * (_team)->node_size) * 2)


#define UMR_DATA_OFFSET(_team) (sizeof(ucc_tl_mlx5_ctrl_t) * (_team)->node_size)

#define OP_SEGMENT_STORAGE(_req, _team) \
    PTR_OFFSET((_team)->node.storage, OP_SEGMENT_SIZE(_team) * (_req)->seq_index)

#define OP_UMR_DATA(_req, _team) \
    PTR_OFFSET(OP_SEGMENT_STORAGE(_req, _team), UMR_DATA_OFFSET(_team))

#define SEND_UMR_DATA(_req, _team, _col)                                     \
    PTR_OFFSET(OP_UMR_DATA(_req, _team),                                \
               _col * (_team)->node.sbgp->group_size * sizeof(umr_t))

#define RECV_UMR_DATA(_req, _team, _col)                                \
    PTR_OFFSET(OP_UMR_DATA(_req, _team),                                \
               (_team)->max_num_of_columns * (_team)->node.sbgp->group_size * sizeof(umr_t) + \
               _col * (_team)->node.sbgp->group_size * sizeof(umr_t))

#define MY_SEND_UMR_DATA(_req, _team, _col)                     \
    PTR_OFFSET(SEND_UMR_DATA(_req, _team, _col),                \
               (_team)->node.sbgp->group_rank * sizeof(umr_t))

#define MY_RECV_UMR_DATA(_req, _team, _col)     \
    PTR_OFFSET(RECV_UMR_DATA(_req, _team, _col),                \
               (_team)->node.sbgp->group_rank * sizeof(umr_t))

#endif
