/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_DATASTRUCT_H_
#define UCC_DATASTRUCT_H_
#include <ucc/api/ucc.h>
#include <ucs/datastruct/list.h>
#include <stdint.h>

#define UCC_LIST_HEAD UCS_LIST_HEAD
typedef uint32_t ucc_rank_t;
#define UCC_RANK_MAX UINT32_MAX

typedef struct ucc_subset {
    ucc_ep_map_t map;
    ucc_rank_t   myrank;
} ucc_subset_t ;

#endif
