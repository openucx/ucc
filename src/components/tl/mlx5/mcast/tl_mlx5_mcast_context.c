/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include <inttypes.h>
#include "tl_mlx5_mcast.h"
#include "utils/arch/cpu.h"
#include <ucs/sys/string.h>
#include "core/ucc_service_coll.h"
#include "tl_mlx5.h"

ucc_status_t ucc_tl_mlx5_mcast_context_init(ucc_tl_mlx5_mcast_context_t    *context, /* NOLINT */
                                            ucc_tl_mlx5_mcast_ctx_params_t *mcast_ctx_conf /* NOLINT */)
{
    return UCC_OK;
}
