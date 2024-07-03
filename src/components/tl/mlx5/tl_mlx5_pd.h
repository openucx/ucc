/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef TL_MLX5_PD_H
#define TL_MLX5_PD_H

#include <sys/un.h>

typedef struct ucc_tl_mlx5_team ucc_tl_mlx5_team_t;

ucc_status_t ucc_tl_mlx5_share_ctx_pd(ucc_tl_mlx5_context_t *ctx,
                                      const char *           sock_path,
                                      ucc_rank_t group_size, int is_ctx_owner,
                                      int ctx_owner_sock);

ucc_status_t ucc_tl_mlx5_remove_shared_ctx_pd(ucc_tl_mlx5_context_t *ctx);

ucc_status_t ucc_tl_mlx5_socket_init(ucc_tl_mlx5_context_t *ctx,
                                     ucc_rank_t group_size, int *sock_p,
                                     const char *sock_path);
#endif
