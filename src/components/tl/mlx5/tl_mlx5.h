/**
 * Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "mcast/tl_mlx5_mcast.h"

#ifndef UCC_TL_MLX5_DEFAULT_SCORE
#define UCC_TL_MLX5_DEFAULT_SCORE 1
#endif

#ifdef HAVE_PROFILING_TL_MLX5
#include "utils/profile/ucc_profile.h"
#else
#include "utils/profile/ucc_profile_off.h"
#endif

#define UCC_TL_MLX5_PROFILE_FUNC          UCC_PROFILE_FUNC
#define UCC_TL_MLX5_PROFILE_FUNC_VOID     UCC_PROFILE_FUNC_VOID
#define UCC_TL_MLX5_PROFILE_REQUEST_NEW   UCC_PROFILE_REQUEST_NEW
#define UCC_TL_MLX5_PROFILE_REQUEST_EVENT UCC_PROFILE_REQUEST_EVENT
#define UCC_TL_MLX5_PROFILE_REQUEST_FREE  UCC_PROFILE_REQUEST_FREE

#define ATOMIC_IN_MEMIC 1
#define DC_KEY 1

typedef struct ucc_tl_mlx5_iface {
    ucc_tl_iface_t super;
} ucc_tl_mlx5_iface_t;
/* Extern iface should follow the pattern: ucc_tl_<tl_name> */
extern ucc_tl_mlx5_iface_t ucc_tl_mlx5;

typedef struct ucc_tl_mlx5_ib_qp_conf {
    uint8_t             qp_sl;
    uint32_t            qp_rnr_retry;
    uint32_t            qp_rnr_timer;
    uint32_t            qp_retry_cnt;
    uint32_t            qp_timeout;
    uint32_t            qp_max_atomic;
} ucc_tl_mlx5_ib_qp_conf_t;

typedef enum ucc_tl_mlx5_alltoall_block_shape_modes
{
    UCC_TL_MLX5_ALLTOALL_BLOCK_SHAPE_LONG,
    UCC_TL_MLX5_ALLTOALL_BLOCK_SHAPE_WIDE,
    UCC_TL_MLX5_ALLTOALL_BLOCK_SHAPE_SQUARE,
    UCC_TL_MLX5_ALLTOALL_BLOCK_SHAPE_LAST,
} ucc_tl_mlx5_alltoall_block_shape_modes_t;

typedef struct ucc_tl_mlx5_lib_config {
    ucc_tl_lib_config_t                      super;
    int                                      asr_barrier;
    int                                      block_size;
    int                                      num_dci_qps;
    int                                      dc_threshold;
    size_t                                   dm_buf_size;
    unsigned long                            dm_buf_num;
    int                                      dm_host;
    ucc_tl_mlx5_ib_qp_conf_t                 qp_conf;
    ucc_tl_mlx5_mcast_coll_comm_init_spec_t  mcast_conf;
    int                                      num_serialized_batches;
    int                                      num_batches_per_passage;
    int                                      block_batch_size;
    int                                      force_regular;
    ucc_tl_mlx5_alltoall_block_shape_modes_t block_shape_mode;
} ucc_tl_mlx5_lib_config_t;

typedef struct ucc_tl_mlx5_context_config {
    ucc_tl_context_config_t         super;
    ucs_config_names_array_t        devices;
    ucc_tl_mlx5_mcast_ctx_params_t  mcast_ctx_conf;
    int                             enable_alltoall;
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
    struct ibv_context          *shared_ctx;
    struct ibv_pd               *shared_pd;
    ucc_rcache_t                *rcache;
    int                          is_imported;
    int                          ib_port;
    int                          sock;
    ucc_mpool_t                  req_mp;
    ucc_tl_mlx5_mcast_context_t  mcast;
    uint16_t                     supported_mem_types;
} ucc_tl_mlx5_context_t;
UCC_CLASS_DECLARE(ucc_tl_mlx5_context_t, const ucc_base_context_params_t*,
                  const ucc_base_config_t*);

typedef struct ucc_tl_mlx5_task ucc_tl_mlx5_task_t;
typedef struct ucc_tl_mlx5_schedule ucc_tl_mlx5_schedule_t;
typedef struct ucc_tl_mlx5_dm_chunk_t {
    uintptr_t addr; // 0 based offset from the beginning of
                    // memic_mr (obtained with ibv_reg_dm_mr)
    ucc_tl_mlx5_schedule_t *task;
    int                     posted_sends;
    int                     posted_all;
    int                     completed_sends;
} ucc_tl_mlx5_dm_chunk_t;

typedef struct ucc_tl_mlx5_alltoall ucc_tl_mlx5_alltoall_t;

typedef enum
{
    TL_MLX5_TEAM_STATE_ALLTOALL_CTX_CHECK,
    TL_MLX5_TEAM_STATE_ALLTOALL_INIT,
    TL_MLX5_TEAM_STATE_ALLTOALL_POSTED,
    TL_MLX5_TEAM_STATE_ALLTOALL_READY,
    TL_MLX5_TEAM_STATE_ALLTOALL_NOT_AVAILABLE,
} ucc_tl_mlx5_team_a2a_state_t;

typedef enum
{
    TL_MLX5_TEAM_STATE_MCAST_CTX_CHECK,
    TL_MLX5_TEAM_STATE_MCAST_INIT,
    TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_TEST,
    TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_READY,
    TL_MLX5_TEAM_STATE_MCAST_GRP_JOIN_FAILED,
    TL_MLX5_TEAM_STATE_MCAST_GRP_BCAST_POST,
    TL_MLX5_TEAM_STATE_MCAST_RELIAB_SYNC,
    TL_MLX5_TEAM_STATE_MCAST_RELIABLITY,
    TL_MLX5_TEAM_STATE_MCAST_READY,
    TL_MLX5_TEAM_STATE_MCAST_NOT_AVAILABLE
} ucc_tl_mlx5_team_mcast_state_t;

typedef struct ucc_tl_mlx5_team_status {
    ucc_status_t local;
    ucc_status_t global;
} ucc_tl_mlx5_team_status_t;

#define UCC_TL_MLX5_FEATURES_COUNT     2 /* Currently only alltoall and mcast are supported */
#define UCC_TL_MLX5_A2A_STATUS_INDEX   0
#define UCC_TL_MLX5_MCAST_STATUS_INDEX 1

typedef struct ucc_tl_mlx5_team {
    ucc_tl_team_t                   super;
    ucc_service_coll_req_t         *scoll_req;
    ucc_service_coll_req_t         *global_sync_req;
    ucc_tl_mlx5_team_a2a_state_t    a2a_state;
    ucc_tl_mlx5_team_mcast_state_t  mcast_state;
    void                           *dm_offset;
    ucc_mpool_t                     dm_pool;
    struct ibv_dm                  *dm_ptr;
    struct ibv_mr                  *dm_mr;
    ucc_tl_mlx5_team_status_t       a2a_status;
    ucc_tl_mlx5_alltoall_t         *a2a;
    ucc_topo_t                     *topo;
    ucc_ep_map_t                    ctx_map;
    int                             local_mcast_team_ready;
    ucc_tl_mlx5_mcast_team_t       *mcast;
    ucc_status_t                    local_status_array[UCC_TL_MLX5_FEATURES_COUNT];
    ucc_status_t                    global_status_array[UCC_TL_MLX5_FEATURES_COUNT];
} ucc_tl_mlx5_team_t;
UCC_CLASS_DECLARE(ucc_tl_mlx5_team_t, ucc_base_context_t *,
                  const ucc_base_team_params_t *);

ucc_status_t tl_mlx5_rcache_create(ucc_tl_mlx5_context_t *ctx);

typedef struct ucc_tl_mlx5_reg {
    struct ibv_mr       *mr;
} ucc_tl_mlx5_reg_t;

typedef struct ucc_tl_mlx5_rcache_region {
    ucc_rcache_region_t super;
    ucc_tl_mlx5_reg_t   reg;
} ucc_tl_mlx5_rcache_region_t;

#define UCC_TL_MLX5_SUPPORTED_COLLS                                            \
    (UCC_COLL_TYPE_ALLTOALL | UCC_COLL_TYPE_BCAST | UCC_COLL_TYPE_ALLGATHER)

#define UCC_TL_MLX5_TEAM_LIB(_team)                                            \
    (ucc_derived_of((_team)->super.super.context->lib, ucc_tl_mlx5_lib_t))

#define UCC_TL_MLX5_TEAM_CTX(_team)                                            \
    (ucc_derived_of((_team)->super.super.context, ucc_tl_mlx5_context_t))

#define UCC_TL_CTX_HAS_OOB(_ctx)                                               \
    ((_ctx)->super.super.ucc_context->params.mask & UCC_CONTEXT_PARAM_FIELD_OOB)

#define UCC_TL_CTX_OOB(_ctx) ((_ctx)->super.super.ucc_context->params.oob)

#define UCC_TL_CTX_LIB(_ctx)                                                   \
    (ucc_derived_of((_ctx)->super.super.lib, ucc_tl_mlx5_lib_t))

#define SQUARED(_num) ((_num) * (_num))

ucc_status_t tl_mlx5_create_rcache(ucc_tl_mlx5_context_t *ctx);

ucc_status_t ucc_tl_mlx5_asr_socket_init(ucc_tl_mlx5_context_t *ctx,
                                         ucc_rank_t group_size, int *socket,
                                         const char *sock_path);

ucc_status_t ucc_tl_mlx5_dm_alloc_reg(struct ibv_context *ib_ctx,
                                      struct ibv_pd *pd, int dm_host,
                                      size_t buf_size, size_t *buf_num_p,
                                      struct ibv_dm **ptr, struct ibv_mr **mr,
                                      ucc_base_lib_t *lib);

#endif
