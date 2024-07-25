/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_mlx5.h"

ucc_status_t ucc_tl_mlx5_dm_alloc_reg(struct ibv_context *ib_ctx,
                                      struct ibv_pd *pd, int dm_host,
                                      size_t buf_size, size_t *buf_num_p,
                                      struct ibv_dm **ptr, struct ibv_mr **mr,
                                      ucc_base_lib_t *lib);

void ucc_tl_mlx5_dm_cleanup(ucc_tl_mlx5_team_t *team);

ucc_status_t ucc_tl_mlx5_dm_init(ucc_tl_mlx5_team_t *team);
