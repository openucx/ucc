/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_MCAST_COLL_H_
#define UCC_TL_MLX5_MCAST_COLL_H_

#include "tl_mlx5_mcast.h"
#include "tl_mlx5_coll.h"

ucc_status_t ucc_tl_mlx5_mcast_bcast_init(ucc_tl_mlx5_task_t *task);

ucc_status_t ucc_tl_mlx5_mcast_test(ucc_tl_mlx5_mcast_coll_req_t* _req);

ucc_status_t ucc_tl_mlx5_mcast_check_support(ucc_base_coll_args_t *coll_args,
                                             ucc_base_team_t *team);

#endif
