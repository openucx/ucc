/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <ucc/api/ucc.h>
#include "components/tl/mlx5/mcast/tl_mlx5_mcast.h"

ucc_status_t ucc_tl_mlx5_mcast_p2p_send_nb(void* src, size_t size, ucc_rank_t
                                           rank, ucc_memory_type_t mem_type, void *context,
                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t
                                           *obj);

ucc_status_t ucc_tl_mlx5_mcast_p2p_recv_nb(void* dst, size_t size, ucc_rank_t
                                           rank, ucc_memory_type_t mem_type, void *context,
                                           ucc_tl_mlx5_mcast_p2p_completion_obj_t
                                           *obj);

ucc_status_t ucc_tl_mlx5_one_sided_p2p_put(void* src, void* remote_addr, size_t length,
                                           uint32_t lkey, uint32_t rkey, ucc_rank_t target_rank,
                                           uint64_t wr_id, ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_one_sided_p2p_get(void* src, void* remote_addr, size_t length,
                                           uint32_t lkey, uint32_t rkey, ucc_rank_t target_rank,
                                           uint64_t wr_id, ucc_tl_mlx5_mcast_coll_comm_t *comm);

