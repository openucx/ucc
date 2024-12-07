/**
 * Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TLCP_UCP_EXAMPLE_CTX_H_
#define UCC_TLCP_UCP_EXAMPLE_CTX_H_

#include "components/tl/ucp/tl_ucp.h"
#include "components/tl/ucp/tl_ucp_coll.h"

typedef struct ucc_tlcp_ucp_example_config {
    char *score_str;
} ucc_tlcp_ucp_example_config_t;

typedef struct ucc_tlcp_ucp_example_am_msg {
    ucc_list_link_t  list_elem;
    uint64_t         tag;
    void            *msg;
} ucc_tlcp_ucp_example_am_msg_t;

typedef struct ucc_tlcp_ucp_example_context {
    ucc_list_link_t am_list;
} ucc_tlcp_ucp_example_context_t;

ucc_status_t ucc_tl_ucp_allreduce_knomial_am_init(ucc_base_coll_args_t *coll_args,
                                                  ucc_base_team_t *tl_team,
                                                  ucc_coll_task_t **task_h);

#endif
