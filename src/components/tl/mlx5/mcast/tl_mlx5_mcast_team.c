/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include "tl_mlx5.h"
#include "tl_mlx5_mcast_coll.h"
#include "coll_score/ucc_coll_score.h"

ucc_status_t ucc_tl_mlx5_mcast_team_init(ucc_base_context_t           *base_context, /* NOLINT */
                                         ucc_tl_mlx5_mcast_team_t    **mcast_team, /* NOLINT */
                                         ucc_tl_mlx5_mcast_context_t  *ctx, /* NOLINT */
                                         const ucc_base_team_params_t *params, /* NOLINT */
                                         ucc_tl_mlx5_mcast_coll_comm_init_spec_t  *mcast_conf /* NOLINT */)
{
    return UCC_OK;
}

