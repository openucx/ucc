/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef SCATTERV_H_
#define SCATTERV_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_SCATTERV_ALG_LINEAR,
    UCC_TL_UCP_SCATTERV_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_scatterv_algs[UCC_TL_UCP_SCATTERV_ALG_LAST + 1];

ucc_status_t ucc_tl_ucp_scatterv_init(ucc_tl_ucp_task_t *task);

#endif
