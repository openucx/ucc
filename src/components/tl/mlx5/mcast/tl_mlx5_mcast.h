/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MCAST_H
#define UCC_MCAST_H

#include <infiniband/ib.h>
#include <infiniband/umad.h>
#include <infiniband/verbs.h>
#include <rdma/rdma_verbs.h>
#include "utils/ucc_list.h"
#include "utils/ucc_mpool.h"
#include "components/tl/ucc_tl.h"
#include "components/tl/ucc_tl_log.h"
#include "utils/ucc_rcache.h"


#define UCC_TL_MLX5_MCAST_ENABLE_BLOCKING true

struct ucc_tl_mlx5_mcast_p2p_completion_obj;
typedef int (*ucc_tl_mlx5_mcast_p2p_completion_cb_fn_t)(struct ucc_tl_mlx5_mcast_p2p_completion_obj *obj);
typedef struct ucc_tl_mlx5_mcast_p2p_completion_obj {
    ucc_list_link_t                          super;
    ucc_tl_mlx5_mcast_p2p_completion_cb_fn_t compl_cb;
    uint64_t                                 data[3];
    ucc_coll_req_h                           req;
} ucc_tl_mlx5_mcast_p2p_completion_obj_t;

typedef struct mcast_coll_comm_init_spec {
} mcast_coll_comm_init_spec_t;

typedef int (*ucc_tl_mlx5_mcast_p2p_wait_cb_fn_t)(void *wait_arg);

typedef int (*ucc_tl_mlx5_mcast_p2p_send_nb_fn_t)(void* src, size_t size,
                                                  ucc_rank_t rank, void *context,
                                                  ucc_tl_mlx5_mcast_p2p_completion_obj_t *compl_obj);


typedef int (*ucc_tl_mlx5_mcast_p2p_recv_nb_fn_t)(void* src, size_t size,
                                                  ucc_rank_t rank, void *context,
                                                  ucc_tl_mlx5_mcast_p2p_completion_obj_t *compl_obj);

typedef struct ucc_tl_mlx5_mcast_context_config {
    ucc_tl_context_config_t  super;
    char                    *dev_list;
    int                      use_rcache;
    size_t                   reg_threshold;
    unsigned int             rand_seed;
    unsigned int             uprogress_num_polls;
    int                      context_per_team;
} ucc_tl_mlx5_mcast_context_config_t;

typedef struct ucc_tl_mlx5_mcast_lib {
} ucc_tl_mlx5_mcast_lib_t;
UCC_CLASS_DECLARE(ucc_tl_mlx5_mcast_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_mlx5_mcast_ctx_params {
} ucc_tl_mlx5_mcast_ctx_params_t;

typedef struct ucc_tl_mlx5_mcast_coll_context {
    struct ibv_context            *ctx;
    struct ibv_pd                 *pd;
    char                          *devname;
    int                            max_qp_wr;
    int                            ib_port;
    int                            pkey_index;
    int                            mtu;
    struct rdma_cm_id             *id;
    struct rdma_event_channel     *channel;
    ucc_mpool_t                    compl_objects_mp;
    ucc_mpool_t                    nack_reqs_mp;
    ucc_list_link_t                pending_nacks_list;
    ucc_rcache_t                  *rcache;
    ucc_tl_mlx5_mcast_ctx_params_t params;
    ucc_base_lib_t                *lib;
} ucc_tl_mlx5_mcast_coll_context_t;

typedef struct ucc_tl_mlx5_mcast_oob_ctx {
    void               *ctx;
    union {
        ucc_oob_coll_t *oob;
        ucc_subset_t    subset;
    };
} ucc_tl_mlx5_mcast_oob_ctx_t;

typedef struct ucc_tl_mlx5_mcast_context {
    ucc_thread_mode_t                  tm;
    ucc_tl_mlx5_mcast_coll_context_t   mcast_context;
    ucc_tl_mlx5_mcast_context_config_t cfg;
    ucc_mpool_t                        req_mp;
    ucc_tl_mlx5_mcast_oob_ctx_t        oob_ctx;
} ucc_tl_mlx5_mcast_context_t;

typedef struct ucc_tl_mlx5_mcast_reg {
    void *mr;
} ucc_tl_mlx5_mcast_reg_t;

typedef struct ucc_tl_mlx5_mcast_rcache_region {
    ucc_rcache_region_t     super;
    ucc_tl_mlx5_mcast_reg_t reg;
} ucc_tl_mlx5_mcast_rcache_region_t;


typedef struct mcast_coll_comm { /* Stuff at a per-communicator sort of level */
} mcast_coll_comm_t;

typedef struct ucc_tl_mlx5_mcast_team {
    void *mcast_comm;
} ucc_tl_mlx5_mcast_team_t;

typedef struct ucc_tl_mlx5_mcast_coll_req { /* Stuff that has to happen per call */
} ucc_tl_mlx5_mcast_coll_req_t;

typedef struct ucc_tl_mlx5_mcast_oob_p2p_context {
    ucc_context_h base_ctx;
    ucc_team_h    base_team;
    ucc_rank_t    my_team_rank;
    ucc_subset_t  subset;
} ucc_tl_mlx5_mcast_oob_p2p_context_t;

#define TASK_TEAM_MCAST(_task)                                                       \
    (ucc_derived_of((_task)->super.team, ucc_tl_mlx5_mcast_team_t))
#define TASK_CTX_MCAST(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context, ucc_tl_mlx5_mcast_context_t))
#define TASK_LIB_MCAST(_task)                                                        \
    (ucc_derived_of((_task)->super.team->context->lib, ucc_tl_mlx5_mcast_lib_t))
#define TASK_ARGS_MCAST(_task) (_task)->super.bargs.args

ucc_status_t ucc_tl_mlx5_mcast_team_init(ucc_base_context_t      *tl_context,
                                         ucc_tl_mlx5_mcast_team_t     **mcast_team,
                                         ucc_tl_mlx5_mcast_context_t  *ctx,
                                         const ucc_base_team_params_t *params,
                                         mcast_coll_comm_init_spec_t  *mcast_conf);

ucc_status_t ucc_tl_mlx5_mcast_coll_init(ucc_base_coll_args_t *coll_args,
                                         ucc_base_team_t *team,
                                         ucc_coll_task_t **task_h);

ucc_status_t ucc_tl_mlx5_mcast_context_init(ucc_tl_mlx5_mcast_context_t *mcast_ctx,
                                            ucc_tl_mlx5_mcast_ctx_params_t *mcast_ctx_conf);

#endif
