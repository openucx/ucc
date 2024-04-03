/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <ucc/api/ucc.h>
#include "components/tl/mlx5/mcast/tl_mlx5_mcast.h"

ucc_status_t ucc_tl_mlx5_mcast_p2p_send_nb(void* src, size_t size, ucc_rank_t
                                           rank, void *context,
                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t
                                           *obj);

ucc_status_t ucc_tl_mlx5_mcast_p2p_recv_nb(void* dst, size_t size, ucc_rank_t
                                           rank, void *context,
                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t
                                           *obj);
