/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef ALLTOALL_H_
#define ALLTOALL_H_

#include "tl_cuda.h"
#include "tl_cuda_coll.h"

ucc_status_t ucc_tl_cuda_alltoall_init(ucc_base_coll_args_t *coll_args,
                                       ucc_base_team_t      *tl_team,
                                       ucc_coll_task_t     **task_p);

#endif
