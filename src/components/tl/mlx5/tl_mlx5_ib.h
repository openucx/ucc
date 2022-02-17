/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_IB_H_
#define UCC_TL_MLX5_IB_H_

#include "tl_mlx5.h"

int  ucc_tl_mlx5_get_active_port(struct ibv_context *ctx);

int  ucc_tl_mlx5_check_port_active(struct ibv_context *ctx, int port_num);

ucc_status_t ucc_tl_mlx5_create_ibv_ctx(char **              ib_devname,
                                        struct ibv_context **ctx,ucc_base_lib_t *lib);


ucc_status_t ucc_tl_mlx5_qp_connect(struct ibv_qp *qp, uint32_t qp_num,
                                    uint16_t lid, int port, ucc_base_lib_t *lib);
ucc_status_t ucc_tl_mlx5_init_dc_qps_and_connect(ucc_tl_mlx5_team_t *team,
                                                 uint32_t *          local_data,
                                                 uint8_t             port_num);
ucc_status_t ucc_tl_mlx5_create_rc_qps(ucc_tl_mlx5_team_t *team,
                                       uint32_t *          local_data);
ucc_status_t ucc_tl_mlx5_create_ah(struct ibv_ah **ah_ptr, uint16_t lid,
                                   uint8_t port_num, ucc_tl_mlx5_team_t *team);

ucc_status_t ucc_tl_mlx5_init_mkeys(ucc_tl_mlx5_team_t *team);

ucc_status_t ucc_tl_mlx5_post_transpose(struct ibv_qp *qp, uint32_t src_mr_lkey, uint32_t dst_mr_key,
                                        uintptr_t src_mkey_addr, uintptr_t dst_addr,
                                        uint32_t element_size, uint16_t ncols, uint16_t nrows);

ucc_status_t ucc_tl_mlx5_post_umr(
    struct ibv_qp *qp, struct mlx5dv_mkey *dv_mkey,
    uint32_t access_flags, uint32_t repeat_count, uint16_t num_entries,
    struct mlx5dv_mr_interleaved *data, uint32_t ptr_mkey, void *ptr_address);
#endif
