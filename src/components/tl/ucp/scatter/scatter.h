/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */
#ifndef SCATTER_H_
#define SCATTER_H_
#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

/* Base interface signature: uses scatter_kn_radix from config. */

ucc_status_t
ucc_tl_ucp_scatter_knomial_init(ucc_base_coll_args_t *coll_args,
                                ucc_base_team_t      *team,
                                ucc_coll_task_t     **task_h);

/* Internal interface to KN scatter with custom radix */
ucc_status_t ucc_tl_ucp_scatter_knomial_init_r(
    ucc_base_coll_args_t *coll_args, ucc_base_team_t *team,
    ucc_coll_task_t **task_h, ucc_kn_radix_t radix);
#endif
