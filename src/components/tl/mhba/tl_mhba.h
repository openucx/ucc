/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MHBA_H_
#define UCC_TL_MHBA_H_
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "core/ucc_service_coll.h"
#include "utils/ucc_mpool.h"
#include "utils/ucc_rcache.h"
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>
/* #include "tl_mhba_mkeys.h" */

#ifndef UCC_TL_MHBA_DEFAULT_SCORE
#define UCC_TL_MHBA_DEFAULT_SCORE 1
#endif

#ifdef HAVE_PROFILING_TL_MHBA
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define MHBA_ASR_RANK                0
#define MHBA_CTRL_SIZE               128 //todo change according to arch
#define MHBA_DATA_SIZE               sizeof(struct mlx5dv_mr_interleaved)
#define MHBA_NUM_OF_BLOCKS_SIZE_BINS 8
#define MAX_TRANSPOSE_SIZE           8192 // HW transpose unit is limited to matrix size
#define MAX_MSG_SIZE                 128 // HW transpose unit is limited to element size
#define MAX_BLOCK_SIZE               64 // from limit of Transpose unit capabilities
#define RC_DC_LIMIT                  128
#define DC_KEY                       1
#define MAX_OUTSTANDING_OPS 1 //todo change - according to limitations (52 top)

#define UCC_TL_MHBA_PROFILE_FUNC          UCC_PROFILE_FUNC
#define UCC_TL_MHBA_PROFILE_FUNC_VOID     UCC_PROFILE_FUNC_VOID
#define UCC_TL_MHBA_PROFILE_REQUEST_NEW   UCC_PROFILE_REQUEST_NEW
#define UCC_TL_MHBA_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_MHBA_PROFILE_REQUEST_FREE  UCC_PROFILE_REQUEST_FREE

typedef struct ucc_tl_mhba_iface {
    ucc_tl_iface_t super;
} ucc_tl_mhba_iface_t;
/* Extern iface should follow the pattern: ucc_tl_<tl_name> */
extern ucc_tl_mhba_iface_t ucc_tl_mhba;

typedef struct ucc_tl_mhba_lib_config {
    ucc_tl_lib_config_t super;

    int    transpose;
    size_t transpose_buf_size;
    int    block_size;
    int    num_dci_qps;
    int    rc_dc;
} ucc_tl_mhba_lib_config_t;

typedef struct ucc_tl_mhba_context_config {
    ucc_tl_context_config_t  super;
    ucs_config_names_array_t devices;
} ucc_tl_mhba_context_config_t;

typedef struct ucc_tl_mhba_lib {
    ucc_tl_lib_t             super;
    ucc_tl_mhba_lib_config_t cfg;
} ucc_tl_mhba_lib_t;
UCC_CLASS_DECLARE(ucc_tl_mhba_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_mhba_context {
    ucc_tl_context_t             super;
    ucc_tl_mhba_context_config_t cfg;
    struct ibv_context *         ib_ctx;
    struct ibv_pd *              ib_pd;
    struct ibv_context *shared_ctx;
    struct ibv_pd *     shared_pd;
    ucc_rcache_t *           rcache;
    int                 is_imported;
    int                          ib_port;
    ucc_mpool_t                  req_mp;
} ucc_tl_mhba_context_t;
UCC_CLASS_DECLARE(ucc_tl_mhba_context_t, const ucc_base_context_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_mhba_task ucc_tl_mhba_task_t;

typedef struct ucc_tl_mhba_ctrl {
    int     seq_num;
    uint8_t mkey_cache_flag;
} ucc_tl_mhba_ctrl_t;

typedef struct ucc_tl_mhba_op {
    void *               ctrl;
    ucc_tl_mhba_ctrl_t * my_ctrl;
    void **              send_umr_data;
    void **              my_send_umr_data;
    void **              recv_umr_data;
    void **              my_recv_umr_data;
    struct mlx5dv_mkey **send_mkeys;
    struct mlx5dv_mkey **recv_mkeys;
} ucc_tl_mhba_op_t;

struct ucc_tl_mhba_internal_qp {
    int                       nreq;
    uint32_t                  cur_size;
    struct mlx5_wqe_ctrl_seg *cur_ctrl;
    uint8_t                   fm_cache;
    void *                    sq_start;
    struct mlx5dv_qp          qp;
    void *                    sq_qend;
    unsigned                  sq_cur_post;
    uint32_t                  qp_num;
    ucs_spinlock_t            qp_spinlock;
    unsigned                  offset;
};

struct ucc_tl_mhba_mlx5_qp {
    struct ibv_qp *      qp;
    struct ibv_qp_ex *   qpx;
    struct mlx5dv_qp_ex *mlx5dv_qp_ex;
};

struct ucc_tl_mhba_qp {
    struct ucc_tl_mhba_mlx5_qp     mlx5_qp;
    struct ucc_tl_mhba_internal_qp in_qp;
};

/* This structure holds resources and data related to the "in-node"
   part of the algorithm. */
typedef struct ucc_tl_mhba_node {
    int                 asr_rank;
    ucc_sbgp_t *        sbgp;
    void *              storage;
    ucc_tl_mhba_op_t       ops[MAX_OUTSTANDING_OPS];
    struct mlx5dv_mkey *team_recv_mkey;
    struct ibv_cq *     umr_cq;
    struct ucc_tl_mhba_mlx5_qp
        ns_umr_qp; // Non-strided - used for team UMR hirerchy
    struct ucc_tl_mhba_qp
          s_umr_qp; // Strided - used for operation send/recv mkey hirerchy
    void *umr_entries_buf;
    struct ibv_mr *umr_entries_mr;
} ucc_tl_mhba_node_t;

typedef struct ucc_tl_mhba_reg {
    struct ibv_mr *      mr;
    ucs_rcache_region_t *region;
} ucc_tl_mhba_reg_t;

static inline ucc_tl_mhba_reg_t *
ucc_tl_mhba_get_rcache_reg_data(ucc_rcache_region_t *region)
{
	return (ucc_tl_mhba_reg_t *)((ptrdiff_t)region + sizeof(ucc_rcache_region_t));
}

typedef struct ucc_tl_mhba_net {
    ucc_sbgp_t *    sbgp;
    int             net_size;
    int *           rank_map;
    struct ibv_qp **rc_qps;
    struct ibv_qp * dct_qp;
    struct ibv_srq *srq;
    uint32_t *      remote_dctns;
    struct ibv_ah **ahs;
    struct ibv_cq * cq;
    struct ibv_mr * ctrl_mr;
    struct {
        void *   addr;
        uint32_t rkey;
    } * remote_ctrl;
    uint32_t *rkeys;
    struct dci {
        struct ibv_qp *      dci_qp;
        struct ibv_qp_ex *   dc_qpex;
        struct mlx5dv_qp_ex *dc_mqpex;
    } * dcis;
} ucc_tl_mhba_net_t;

typedef struct ucc_tl_mhba_bcast_data {
    int  shmid;
    int  net_size;
    char sock_path[L_tmpnam];
} ucc_tl_mhba_bcast_data_t;

enum {
    TL_MHBA_TEAM_STATE_SHMID,
    TL_MHBA_TEAM_STATE_EXCHANGE,
};

typedef struct ucc_tl_mhba_team {
    ucc_tl_team_t            super;
    int                      state;
    int                      transpose;
    uint64_t                 max_msg_size;
    ucc_tl_mhba_node_t       node;
    ucc_tl_mhba_net_t        net;
    void *                   service_bcast_tmp_buf;
    int                      sequence_number;
    int                      op_busy[MAX_OUTSTANDING_OPS];
    int                      cq_completions[MAX_OUTSTANDING_OPS];
    int                      blocks_sizes[MHBA_NUM_OF_BLOCKS_SIZE_BINS];
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
    ucc_tl_mhba_bcast_data_t bcast_data;
    ucc_status_t             status;
} ucc_tl_mhba_team_t;
UCC_CLASS_DECLARE(ucc_tl_mhba_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

#define UCC_TL_MHBA_SUPPORTED_COLLS (UCC_COLL_TYPE_ALLTOALL)

#define UCC_TL_MHBA_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_mhba_lib_t))

#define UCC_TL_MHBA_TEAM_CTX(_team)                                            \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_mhba_context_t))

#define UCC_TL_CTX_HAS_OOB(_ctx)                                               \
    ((_ctx)->super.super.ucc_context->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB)

#define UCC_TL_CTX_OOB(_ctx) ((_ctx)->super.super.ucc_context->params.oob)

#define UCC_TL_CTX_LIB(_ctx)                                                   \
    (ucc_derived_of((_ctx)->super.super.lib, ucc_tl_mhba_lib_t))

#define IS_SERVICE_TEAM(_team)                                                 \
    ((_team)->super.super.params.scope == UCC_CL_LAST + 1)


#define SEQ_INDEX(_seq_num) ((_seq_num) % MAX_OUTSTANDING_OPS)
#define SQUARED(_num)       ((_num) * (_num))

enum
{
    UCC_MHBA_NEED_SEND_MKEY_UPDATE = UCS_BIT(1),
    UCC_MHBA_NEED_RECV_MKEY_UPDATE = UCS_BIT(2),
};

ucc_status_t tl_mhba_create_rcache(ucc_tl_mhba_context_t *ctx);

ucc_status_t ucc_tl_mhba_asr_socket_init(ucc_tl_mhba_context_t *ctx, ucc_rank_t group_size,
                                         int *socket, const char *sock_path);

static inline ucc_tl_mhba_ctrl_t *ucc_tl_mhba_get_ctrl(ucc_tl_mhba_team_t *team,
                                                       int op_index, int rank)
{
	ucc_tl_mhba_ctrl_t * ctrl =  (ucc_tl_mhba_ctrl_t *)((ptrdiff_t)team->node.ops[op_index].ctrl +
                                  MHBA_CTRL_SIZE * rank);
	return ctrl;
}

static inline ucc_tl_mhba_ctrl_t *ucc_tl_mhba_get_my_ctrl(ucc_tl_mhba_team_t *team,
                                                          int op_index)
{
    int my_rank = team->node.sbgp->group_rank;
    return ucc_tl_mhba_get_ctrl(team, op_index, my_rank);
}

#endif
