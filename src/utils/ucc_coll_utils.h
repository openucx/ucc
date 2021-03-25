/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_COLL_UTILS_H_
#define UCC_COLL_UTILS_H_

#include "config.h"
#include "ucc_datastruct.h"

#define UCC_IS_INPLACE(_args) \
    (((_args).mask & UCC_COLL_ARGS_FIELD_FLAGS) && \
     ((_args).flags & UCC_COLL_ARGS_FLAG_IN_PLACE))

static inline size_t
ucc_coll_args_get_count(const ucc_coll_args_t *args, const ucc_count_t *counts,
                        ucc_rank_t idx)
{
    if ((args->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
        (args->flags & UCC_COLL_ARGS_FLAG_COUNT_64BIT)) {
        return ((uint64_t *)counts)[idx];
    }
    return ((uint32_t *)counts)[idx];
}

static inline size_t
ucc_coll_args_get_displacement(const ucc_coll_args_t *args,
                               const ucc_aint_t *displacements, ucc_rank_t idx)
{
    if ((args->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
        (args->flags & UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT)) {
        return ((uint64_t *)displacements)[idx];
    }
    return ((uint32_t *)displacements)[idx];
}

static inline size_t
ucc_coll_args_get_total_count(const ucc_coll_args_t *args,
                              const ucc_count_t *counts, ucc_rank_t size)
{
    size_t count = 0;
    int i;

    if ((args->mask & UCC_COLL_ARGS_FIELD_FLAGS) &&
        (args->flags & UCC_COLL_ARGS_FLAG_COUNT_64BIT)) {
        for (i = 0; i < size; i++) {
            count += ((uint64_t *)counts)[i];
        }
    } else {
        for (i = 0; i < size; i++) {
            count += ((uint32_t *)counts)[i];
        }
    }

    return count;
}


static inline ucc_rank_t ucc_ep_map_eval(ucc_ep_map_t map, ucc_rank_t rank)
{
    ucc_rank_t r;
    switch(map.type) {
    case UCC_EP_MAP_FULL:
        r = rank;
        break;
    case UCC_EP_MAP_STRIDED:
        r = map.strided.start + rank*map.strided.stride;
        break;
    case UCC_EP_MAP_ARRAY:
        r = *((ucc_rank_t*)((ptrdiff_t)map.array.map + rank*map.array.elem_size));
        break;
    case UCC_EP_MAP_CB:
        r = (ucc_rank_t)map.cb.cb(rank, map.cb.cb_ctx);
        break;
    default:
        r = -1;
    }
    return r;
}

#endif
