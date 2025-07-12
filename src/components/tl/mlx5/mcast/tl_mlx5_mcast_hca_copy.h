/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_MCAST_HCA_COPY_H_
#define UCC_TL_MLX5_MCAST_HCA_COPY_H_

#include "tl_mlx5_mcast.h"

typedef struct ucc_tl_mlx5_mcast_hca_copy_task {
    void                             *dst;
    void                             *src;
    size_t                            size;
    ucc_memory_type_t                 dst_mtype;
    ucc_memory_type_t                 src_mtype;
    ucc_tl_mlx5_mcast_coll_comm_t    *comm;

    /* Use RDMA write for HCA copy with rcache optimization */
    ucc_tl_mlx5_mcast_reg_t          *src_reg;
    ucc_tl_mlx5_mcast_reg_t          *dst_reg;
    struct ibv_send_wr                rdma_wr;
    struct ibv_sge                    rdma_sge;
    volatile int                      completed;
    ucc_status_t                      status;
    ucc_rank_t                        target_rank;
} ucc_tl_mlx5_mcast_hca_copy_task_t;

/* HCA copy API functions */
ucc_status_t ucc_tl_mlx5_mcast_hca_copy_post(void *dst, ucc_memory_type_t dst_mtype,
                                             void *src, ucc_memory_type_t src_mtype,
                                             size_t size,
                                             ucc_tl_mlx5_mcast_coll_comm_t *comm,
                                             ucc_tl_mlx5_mcast_hca_copy_task_t **copy_task);

ucc_status_t ucc_tl_mlx5_mcast_hca_copy_test(ucc_tl_mlx5_mcast_hca_copy_task_t *copy_task);

ucc_status_t ucc_tl_mlx5_mcast_hca_copy_finalize(ucc_tl_mlx5_mcast_hca_copy_task_t *copy_task);

/* Helper function to choose between HCA copy and mc copy */
ucc_status_t ucc_tl_mlx5_mcast_memcpy(void *dst, ucc_memory_type_t dst_mtype,
                                      void *src, ucc_memory_type_t src_mtype,
                                      size_t size,
                                      ucc_tl_mlx5_mcast_coll_comm_t *comm);

#endif /* UCC_TL_MLX5_MCAST_HCA_COPY_H_ */
