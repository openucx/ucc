/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_MLX5_MCAST_ALLGATHER_H_
#define UCC_TL_MLX5_MCAST_ALLGATHER_H_

#include "tl_mlx5_mcast.h"
#include "tl_mlx5_coll.h"

#define MCAST_ALLGATHER_IN_PROGRESS(_req, _comm)                                     \
        (_req->to_send || _req->to_recv || _comm->pending_send ||                    \
         _comm->one_sided.rdma_read_in_progress || (NULL != _req->allgather_rkeys_req))      \

ucc_status_t ucc_tl_mlx5_mcast_allgather_init(ucc_tl_mlx5_task_t *task);

ucc_status_t ucc_tl_mlx5_mcast_test_allgather(ucc_tl_mlx5_mcast_coll_req_t* _req);

#endif
