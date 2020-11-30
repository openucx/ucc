/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_UCP_TAG_H_
#define UCC_UCP_TAG_H_
/*
 * UCP tag structure:
 *
 *    01234           567     01234567 01234567 01234567 01234567 01234567 01234567 01234567
 *                |         |                  |                          |
 *   RESERVED (5) | CTX (3) | message tag (16) |     source rank (24)     |  team id (16)
 */

typedef enum {
    UCC_UCP_CTX_TL_UCX = 0,
    UCC_UCP_CTX_CL_HIER,
    UCC_UCP_CTX_CL_UCG,
    UCC_UCP_CTX_LAST,
} ucc_ucp_ctx_type_t;

#define UCC_UCP_RESERVED_BITS 5
#define UCC_UCP_CTX_BITS 3
#define UCC_UCP_TAG_BITS 16
#define UCC_UCP_SENDER_BITS 24
#define UCC_UCP_ID_BITS 16

#define UCC_UCP_RESERVED_BITS_OFFSET                                           \
    (UCC_UCP_ID_BITS + UCC_UCP_SENDER_BITS + UCC_UCP_TAG_BITS +                \
     UCC_UCP_CTX_BITS)

#define UCC_UCP_CTX_BITS_OFFSET                                                \
    (UCC_UCP_ID_BITS + UCC_UCP_SENDER_BITS + UCC_UCP_TAG_BITS)

#define UCC_UCP_TAG_BITS_OFFSET    (UCC_UCP_ID_BITS + UCC_UCP_SENDER_BITS)
#define UCC_UCP_SENDER_BITS_OFFSET (UCC_UCP_ID_BITS)
#define UCC_UCP_ID_BITS_OFFSET     0

#define UCC_UCP_MAX_TAG    UCC_MASK(UCC_UCP_TAG_BITS)
#define UCC_UCP_MAX_SENDER UCC_MASK(UCC_UCP_SENDER_BITS)
#define UCC_UCP_MAX_ID     UCC_MASK(UCC_UCP_ID_BITS)

#define UCC_UCP_TAG_SENDER_MASK UCC_MASK(UCC_UCP_ID_BITS + UCC_UCP_SENDER_BITS)
#endif
