/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_MCAST_ALLGATHER_H_
#define UCC_TL_MLX5_MCAST_ALLGATHER_H_

#include "tl_mlx5_mcast.h"
#include "tl_mlx5_coll.h"

ucc_status_t ucc_tl_mlx5_mcast_allgather_init(ucc_tl_mlx5_task_t *task);

#endif
