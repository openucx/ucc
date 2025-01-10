/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
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


ucc_status_t ucc_tl_mlx5_mcast_one_sided_reliability_init(ucc_base_team_t *team);

ucc_status_t ucc_tl_mlx5_mcast_one_sided_reliability_test(ucc_base_team_t *team);
