/*
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MHBA_IB_H_
#define UCC_TL_MHBA_IB_H_

#include "tl_mhba.h"

int  ucc_tl_mhba_get_active_port(struct ibv_context *ctx);

int  ucc_tl_mhba_check_port_active(struct ibv_context *ctx, int port_num);

ucc_status_t ucc_tl_mhba_create_ibv_ctx(char **              ib_devname,
                                        struct ibv_context **ctx,ucc_base_lib_t *lib);


ucc_status_t ucc_tl_mhba_qp_connect(struct ibv_qp *qp, uint32_t qp_num,
                                    uint16_t lid, int port, ucc_base_lib_t *lib);
ucc_status_t ucc_tl_mhba_init_dc_qps_and_connect(ucc_tl_mhba_team_t *team,
                                                 uint32_t *          local_data,
                                                 uint8_t             port_num);
ucc_status_t ucc_tl_mhba_create_rc_qps(ucc_tl_mhba_team_t *team,
                                       uint32_t *          local_data);
ucc_status_t ucc_tl_mhba_create_ah(struct ibv_ah **ah_ptr, uint16_t lid,
                                   uint8_t port_num, ucc_tl_mhba_team_t *team);
ucc_status_t
ucc_tl_mhba_ibv_qp_to_mlx5dv_qp(struct ibv_qp *                 umr_qp,
                                struct ucc_tl_mhba_internal_qp *mqp, ucc_tl_mhba_lib_t *lib);

ucc_status_t ucc_tl_mhba_destroy_mlxdv_qp(struct ucc_tl_mhba_internal_qp *mqp);

void ucc_tl_mhba_wr_start(struct ucc_tl_mhba_internal_qp *mqp);

void ucc_tl_mhba_wr_complete(struct ucc_tl_mhba_internal_qp *mqp);

ucc_status_t ucc_tl_mhba_send_wr_mr_noninline(
    struct ucc_tl_mhba_internal_qp *mqp, struct mlx5dv_mkey *dv_mkey,
    uint32_t access_flags, uint32_t repeat_count, uint16_t num_entries,
    struct mlx5dv_mr_interleaved *data, uint32_t ptr_mkey, void *ptr_address,
    struct ibv_qp_ex *ibqp);

ucc_status_t ucc_tl_mhba_init_umr(ucc_tl_mhba_context_t *ctx,
                                  ucc_tl_mhba_node_t *   node);

ucc_status_t ucc_tl_mhba_init_mkeys(ucc_tl_mhba_team_t *team);
#endif
