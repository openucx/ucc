/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast.h"
#include "utils/ucc_rcache.h"

ucc_status_t ucc_tl_mlx5_mcast_setup_rcache(ucc_tl_mlx5_mcast_coll_context_t *ctx);

ucc_status_t ucc_tl_mlx5_mcast_mem_register(ucc_tl_mlx5_mcast_coll_context_t
                                           *ctx, void *addr, size_t length,
                                            ucc_tl_mlx5_mcast_reg_t **reg);

ucc_status_t ucc_tl_mlx5_mcast_mem_deregister(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                              ucc_tl_mlx5_mcast_reg_t *reg);
