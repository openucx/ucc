/**
 * Copyright (c) 2001-2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_DATASTRUCT_H_
#define UCC_DATASTRUCT_H_
#include <ucc/api/ucc.h>
#include <ucs/datastruct/list.h>
#include <stdint.h>

#define UCC_LIST_HEAD UCS_LIST_HEAD
typedef uint32_t ucc_rank_t;
#define UCC_RANK_INVALID UINT32_MAX
#define UCC_RANK_MAX UCC_RANK_INVALID - 1

typedef struct ucc_subset {
    ucc_ep_map_t map;
    ucc_rank_t   myrank;
} ucc_subset_t;

static inline ucc_rank_t ucc_subset_size(ucc_subset_t *set)
{
    return (ucc_rank_t)set->map.ep_num;
}

#endif
