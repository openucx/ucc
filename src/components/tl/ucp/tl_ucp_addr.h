/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_UCP_ADDR_H_
#define UCC_TL_UCP_ADDR_H_
#include "components/tl/ucc_tl.h"

typedef struct ucc_tl_ucp_context ucc_tl_ucp_context_t;
typedef struct ucc_tl_ucp_addr_storage {
    void                 *oob_req;
    size_t                max_addrlen;
    int                   state;
    int                   is_ctx;
    size_t               *addrlens;
    void                 *addresses;
    ucc_oob_coll_t        oob;
    ucc_tl_ucp_context_t *ctx;
} ucc_tl_ucp_addr_storage_t;

ucc_status_t ucc_tl_ucp_addr_exchange_start(ucc_tl_ucp_context_t *ctx,
                                            ucc_oob_coll_t oob,
                                            ucc_tl_ucp_addr_storage_t **storage);

ucc_status_t ucc_tl_ucp_addr_exchange_test(ucc_tl_ucp_addr_storage_t *storage);

void ucc_tl_ucp_addr_storage_free(ucc_tl_ucp_addr_storage_t *storage);

#endif
