/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_DPU_COLL_H_
#define UCC_TL_DPU_COLL_H_

#include "components/tl/ucc_tl.h"
#include "tl_dpu.h"

#define UCC_TL_DPU_COLL_POLL 100

ucc_status_t ucc_tl_dpu_allreduce_progress(ucc_coll_task_t *coll_task);
ucc_status_t ucc_tl_dpu_allreduce_start(ucc_coll_task_t *coll_task);
ucc_status_t ucc_tl_dpu_allreduce_init(ucc_tl_dpu_task_t *coll_task);
ucc_status_t ucc_tl_dpu_coll_init(ucc_base_coll_args_t  *coll_args,
                                  ucc_base_team_t       *team,
                                  ucc_coll_task_t      **task_h);
#endif
