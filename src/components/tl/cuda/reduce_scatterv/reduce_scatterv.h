/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef REDUCE_SCATTERV_H_
#define REDUCE_SCATTERV_H_

#include "../tl_cuda.h"
#include "../tl_cuda_coll.h"

ucc_status_t ucc_tl_cuda_reduce_scatterv_init(ucc_base_coll_args_t *coll_args,
                                              ucc_base_team_t *tl_team,
                                              ucc_coll_task_t **task_p);

#endif
