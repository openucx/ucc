/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_IB_H_
#define UCC_TL_MLX5_IB_H_

#include "tl_mlx5.h"

typedef struct mlx5dv_mr_interleaved umr_t;

typedef struct ucc_tl_mlx5_qp {
    struct ibv_qp *   qp;
    struct ibv_qp_ex *qp_ex;
} ucc_tl_mlx5_qp_t;

typedef struct ucc_tl_mlx5_dci {
    struct ibv_qp *      dci_qp;
    struct ibv_qp_ex *   dc_qpex;
    struct mlx5dv_qp_ex *dc_mqpex;
} ucc_tl_mlx5_dci_t;

int ucc_tl_mlx5_get_active_port(struct ibv_context *ctx);

int ucc_tl_mlx5_check_port_active(struct ibv_context *ctx, int port_num);

ucc_status_t ucc_tl_mlx5_create_ibv_ctx(char **              ib_devname,
                                        struct ibv_context **ctx,
                                        ucc_base_lib_t *     lib);

ucc_status_t ucc_tl_mlx5_qp_connect(struct ibv_qp *qp, uint32_t qp_num,
                                    uint16_t lid, int port,
                                    ucc_tl_mlx5_ib_qp_conf_t *qp_conf,
                                    ucc_base_lib_t           *lib);

ucc_status_t ucc_tl_mlx5_init_dct(struct ibv_pd *pd, struct ibv_context *ctx,
                                  struct ibv_cq *cq, struct ibv_srq *srq,
                                  uint8_t port_num, struct ibv_qp **dct_qp,
                                  uint32_t                 *qpn,
                                  ucc_tl_mlx5_ib_qp_conf_t *qp_conf,
                                  ucc_base_lib_t           *lib);

ucc_status_t ucc_tl_mlx5_init_dci(ucc_tl_mlx5_dci_t *dci, struct ibv_pd *pd,
                                  struct ibv_context *ctx, struct ibv_cq *cq,
                                  uint8_t port_num, int tx_depth,
                                  ucc_tl_mlx5_ib_qp_conf_t *qp_conf,
                                  ucc_base_lib_t           *lib);

ucc_status_t ucc_tl_mlx5_create_rc_qp(struct ibv_context *ctx,
                                      struct ibv_pd *pd, struct ibv_cq *cq,
                                      int tx_depth, ucc_tl_mlx5_qp_t *qp,
                                      uint32_t *qpn, ucc_base_lib_t *lib);

ucc_status_t ucc_tl_mlx5_create_ah(struct ibv_pd *pd, uint16_t lid,
                                   uint8_t port_num, struct ibv_ah **ah_ptr,
                                   ucc_base_lib_t *lib);

ucc_status_t ucc_tl_mlx5_create_umr_qp(struct ibv_context *ctx,
                                       struct ibv_pd *pd, struct ibv_cq *cq,
                                       int ib_port, struct ibv_qp **qp,
                                       ucc_tl_mlx5_ib_qp_conf_t *qp_conf,
                                       ucc_base_lib_t           *lib);

static inline ucc_status_t tl_mlx5_ah_to_av(struct ibv_ah *ah, struct mlx5_wqe_av *av)
{
    struct mlx5dv_obj obj;
    struct mlx5dv_ah  dv_ah;

    obj.ah.in  = ah;
    obj.ah.out = &dv_ah;
    if (ucc_unlikely(mlx5dv_init_obj(&obj, MLX5DV_OBJ_AH))) {
        return UCC_ERR_NO_MESSAGE;
    }

    *av = *(dv_ah.av);
    return UCC_OK;
}

#endif
