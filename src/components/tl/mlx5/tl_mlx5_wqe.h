/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_WQE_H_
#define UCC_TL_MLX5_WQE_H_

#include "ucc/api/ucc_status.h"
#include <infiniband/verbs.h>
#include <infiniband/mlx5dv.h>

ucc_status_t ucc_tl_mlx5_post_transpose(struct ibv_qp *qp, uint32_t src_mr_lkey,
                                        uint32_t  dst_mr_key,
                                        uintptr_t src_mkey_addr,
                                        uintptr_t dst_addr,
                                        uint32_t element_size, uint16_t ncols,
                                        uint16_t nrows, int send_flags);

ucc_status_t ucc_tl_mlx5_post_umr(struct ibv_qp *     qp,
                                  struct mlx5dv_mkey *dv_mkey,
                                  uint32_t access_flags, uint32_t repeat_count,
                                  uint16_t                      num_entries,
                                  struct mlx5dv_mr_interleaved *data,
                                  uint32_t ptr_mkey, void *ptr_address);

ucc_status_t ucc_tl_mlx5_post_rdma(struct ibv_qp *qp, uint32_t qpn,
                                   struct ibv_ah *ah, uintptr_t src_mkey_addr,
                                   size_t len, uint32_t src_mr_lkey,
                                   uintptr_t dst_addr, uint32_t dst_mr_key,
                                   int send_flags, uint64_t wr_id);

ucc_status_t ucc_tl_mlx5_post_wait_on_data(struct ibv_qp *qp, uint64_t value,
                                           uint32_t lkey, uintptr_t addr,
                                           void *task_ptr);

#endif
