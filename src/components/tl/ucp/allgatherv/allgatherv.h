/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef ALLGATHERV_H_
#define ALLGATHERV_H_

#include "../tl_ucp.h"
#include "../tl_ucp_coll.h"

enum {
    UCC_TL_UCP_ALLGATHERV_ALG_RING,
    UCC_TL_UCP_ALLGATHERV_ALG_LAST
};

extern ucc_base_coll_alg_info_t
             ucc_tl_ucp_allgatherv_algs[UCC_TL_UCP_ALLGATHERV_ALG_LAST + 1];

ucc_status_t ucc_tl_ucp_allgatherv_init(ucc_tl_ucp_task_t *task);

#endif
