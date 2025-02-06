/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5_mcast.h"
#include "utils/ucc_rcache.h"

ucc_status_t ucc_tl_mlx5_mcast_setup_rcache(ucc_tl_mlx5_mcast_coll_context_t *ctx);

ucc_status_t ucc_tl_mlx5_mcast_mem_register(ucc_tl_mlx5_mcast_coll_context_t
                                           *ctx, void *addr, size_t length,
                                            ucc_tl_mlx5_mcast_reg_t **reg);

void ucc_tl_mlx5_mcast_mem_deregister(ucc_tl_mlx5_mcast_coll_context_t *ctx,
                                      ucc_tl_mlx5_mcast_reg_t *reg);
