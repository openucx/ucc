/**
 * Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include "tl_mlx5.h"
#include "tl_mlx5_mcast_coll.h"
#include "coll_score/ucc_coll_score.h"
#include "tl_mlx5_mcast_helper.h"
#include "p2p/ucc_tl_mlx5_mcast_p2p.h"
#include "mcast/tl_mlx5_mcast_helper.h"
#include "mcast/tl_mlx5_mcast_service_coll.h"


ucc_status_t ucc_tl_mlx5_mcast_one_sided_reliability_init(ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_one_sided_reliability_test(ucc_tl_mlx5_mcast_coll_comm_t *comm);

ucc_status_t ucc_tl_mlx5_mcast_one_sided_cleanup(ucc_tl_mlx5_mcast_coll_comm_t *comm);
