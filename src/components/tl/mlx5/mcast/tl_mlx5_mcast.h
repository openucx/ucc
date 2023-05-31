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

typedef struct mcast_coll_comm_init_spec {
} mcast_coll_comm_init_spec_t;

typedef struct ucc_tl_mlx5_mcast_lib {
} ucc_tl_mlx5_mcast_lib_t;
UCC_CLASS_DECLARE(ucc_tl_mlx5_mcast_lib_t, const ucc_base_lib_params_t *,
                  const ucc_base_config_t *);

typedef struct ucc_tl_mlx5_mcast_ctx_params {
} ucc_tl_mlx5_mcast_ctx_params_t;

typedef struct mcast_coll_context_t {
} mcast_coll_context_t;

typedef struct ucc_tl_mlx5_mcast_context_t {
} ucc_tl_mlx5_mcast_context_t;


typedef struct mcast_coll_comm { /* Stuff at a per-communicator sort of level */
} mcast_coll_comm_t;

typedef struct ucc_tl_mlx5_mcast_team {
    void *mcast_comm;
} ucc_tl_mlx5_mcast_team_t;

typedef struct ucc_tl_mlx5_mcast_coll_req { /* Stuff that has to happen per call */
} ucc_tl_mlx5_mcast_coll_req_t;

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
