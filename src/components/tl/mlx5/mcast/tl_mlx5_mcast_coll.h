/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_MCAST_COLL_H_
#define UCC_TL_MLX5_MCAST_COLL_H_

#include "tl_mlx5_mcast.h"
#include "tl_mlx5_coll.h"

ucc_status_t ucc_tl_mlx5_mcast_bcast_init(ucc_tl_mlx5_task_t *task);

ucc_status_t ucc_tl_mlx5_mcast_test(ucc_tl_mlx5_mcast_coll_req_t* _req);

#endif
