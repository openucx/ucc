/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <ucc/api/ucc.h>
#include "components/tl/mlx5/mcast/tl_mlx5_mcast.h"

int ucc_tl_mlx5_mcast_p2p_send_nb(void* src, size_t size, int rank, int tag, void *context,
                                  ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj);

int ucc_tl_mlx5_mcast_p2p_recv_nb(void* src, size_t size, int rank, int tag, void *context,
                                  ucc_tl_mlx5_mcast_p2p_completion_obj_t *obj);

int ucc_tl_mlx5_mcast_p2p_progress(void* ctx);
