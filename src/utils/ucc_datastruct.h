/**
 * Copyright (c) 2001-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_DATASTRUCT_H_
#define UCC_DATASTRUCT_H_
#include <ucc/api/ucc.h>
#include "ucc_list.h"
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

typedef struct ucc_mrange {
    ucc_list_link_t list_elem;
    size_t          start;
    size_t          end;
    uint32_t        mtypes;
    unsigned        value;
} ucc_mrange_t;

typedef struct ucc_mrange_uint {
    ucc_list_link_t ranges;
    unsigned        default_value;
} ucc_mrange_uint_t;

ucc_status_t ucc_mrange_uint_copy(ucc_mrange_uint_t       *dst,
                                  const ucc_mrange_uint_t *src);

void ucc_mrange_uint_destroy(ucc_mrange_uint_t *param);

static inline unsigned ucc_mrange_uint_get(ucc_mrange_uint_t *param,
                                           size_t             range_value,
                                           ucc_memory_type_t  mem_type)
{
    ucc_mrange_t *r;

    ucc_list_for_each(r, &param->ranges, list_elem) {
        if (r->start <= range_value && range_value <= r->end &&
            (UCC_BIT(mem_type) & r->mtypes)) {
            return r->value;
        }
    }
    return param->default_value;
}

#endif
