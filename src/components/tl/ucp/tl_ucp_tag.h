/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Meta Platforms, Inc. and affiliates. 2022.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_TL_UCP_TAG_H_
#define UCC_TL_UCP_TAG_H_
#include "utils/ucc_compiler_def.h"

/*
 * UCP tag structure:
 *
 *     01     |      2       | 34567 01234567 01 |    234   |      567    | 01234567 01234567 01234567 | 01234567 01234567
 *            |              |                   |          |             |                            |
 *  RESERV(2) | user tag (1) |  message tag (15) | SCOPE(3) | SCOPE_ID(3) |     source rank (24)       |    team id (16)
 */

#define UCC_TL_UCP_RESERVED_BITS 2
#define UCC_TL_UCP_SCOPE_BITS    3
#define UCC_TL_UCP_SCOPE_ID_BITS 3
#define UCC_TL_UCP_USER_TAG_BITS 1
#define UCC_TL_UCP_TAG_BITS      15
#define UCC_TL_UCP_SENDER_BITS   24
#define UCC_TL_UCP_ID_BITS       16

#define UCC_TL_UCP_RESERVED_BITS_OFFSET                                        \
    (UCC_TL_UCP_ID_BITS + UCC_TL_UCP_SENDER_BITS + UCC_TL_UCP_SCOPE_ID_BITS +  \
     UCC_TL_UCP_SCOPE_BITS + UCC_TL_UCP_TAG_BITS + UCC_TL_UCP_USER_TAG_BITS)

#define UCC_TL_UCP_USER_TAG_BITS_OFFSET                                        \
    (UCC_TL_UCP_ID_BITS + UCC_TL_UCP_SENDER_BITS + UCC_TL_UCP_SCOPE_ID_BITS +  \
     UCC_TL_UCP_SCOPE_BITS + UCC_TL_UCP_TAG_BITS)

#define UCC_TL_UCP_TAG_BITS_OFFSET                                             \
    (UCC_TL_UCP_ID_BITS + UCC_TL_UCP_SENDER_BITS + UCC_TL_UCP_SCOPE_ID_BITS +  \
     UCC_TL_UCP_SCOPE_BITS)

#define UCC_TL_UCP_SCOPE_BITS_OFFSET                                        \
    (UCC_TL_UCP_ID_BITS + UCC_TL_UCP_SENDER_BITS + UCC_TL_UCP_SCOPE_ID_BITS)

#define UCC_TL_UCP_SCOPE_ID_BITS_OFFSET (UCC_TL_UCP_ID_BITS + UCC_TL_UCP_SENDER_BITS)
#define UCC_TL_UCP_SENDER_BITS_OFFSET   (UCC_TL_UCP_ID_BITS)
#define UCC_TL_UCP_ID_BITS_OFFSET       0

#define UCC_TL_UCP_MAX_TAG         UCC_MASK(UCC_TL_UCP_TAG_BITS)
#define UCC_TL_UCP_RESERVED_TAGS   8
#define UCC_TL_UCP_MAX_COLL_TAG   (UCC_TL_UCP_MAX_TAG - UCC_TL_UCP_RESERVED_TAGS)
#define UCC_TL_UCP_SERVICE_TAG    (UCC_TL_UCP_MAX_COLL_TAG + 1)
#define UCC_TL_UCP_ACTIVE_SET_TAG (UCC_TL_UCP_MAX_COLL_TAG + 2)
#define UCC_TL_UCP_MAX_SENDER      UCC_MASK(UCC_TL_UCP_SENDER_BITS)
#define UCC_TL_UCP_MAX_ID          UCC_MASK(UCC_TL_UCP_ID_BITS)

#define UCC_TL_UCP_TAG_SENDER_MASK                                             \
    UCC_MASK(UCC_TL_UCP_ID_BITS + UCC_TL_UCP_SENDER_BITS + \
             UCC_TL_UCP_SCOPE_ID_BITS + UCC_TL_UCP_SCOPE_BITS)

#define UCC_TL_UCP_GET_SENDER(_tag) ((uint32_t)(((_tag) >> UCC_TL_UCP_SENDER_BITS_OFFSET) & \
                                                UCC_MASK(UCC_TL_UCP_SENDER_BITS)))
#endif
