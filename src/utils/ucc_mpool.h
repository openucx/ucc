/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_MPOOL_H_
#define UCC_MPOOL_H_

#include "config.h"
#include <ucs/datastruct/mpool.h>
#include "ucc_compiler_def.h"

typedef ucs_mpool_t ucc_mpool_t;
#define ucc_mpool_get(_mp) ucs_mpool_get((_mp))
#define ucc_mpool_put(_obj) ucs_mpool_put((_obj))

typedef void (*ucc_mpool_obj_init_fn_t)(ucc_mpool_t *mp, void *obj,
                                        void *chunk);
typedef void (*ucc_mpool_obj_cleanup_fn_t)(ucc_mpool_t *mp, void *obj);

static inline ucc_status_t
ucc_mpool_init(ucc_mpool_t *mp, size_t elem_size, size_t alignment,
               unsigned elems_per_chunk, unsigned max_elems,
               ucc_mpool_obj_init_fn_t    init_fn,
               ucc_mpool_obj_cleanup_fn_t cleanup_fn, const char *name)
{
    ucs_mpool_ops_t *ops = ucc_malloc(sizeof(*ops), "mpool_ops");
    if (!ops) {
        ucc_error("failed to allocate %zd bytes for mpool ops", sizeof(*ops));
        return UCC_ERR_NO_MEMORY;
    }

    ops->chunk_alloc   = ucs_mpool_hugetlb_malloc;
    ops->chunk_release = ucs_mpool_hugetlb_free;
    ops->obj_init      = init_fn;
    ops->obj_cleanup   = cleanup_fn;
    return ucs_status_to_ucc_status(ucs_mpool_init(
        mp, 0, elem_size, 0, alignment, elems_per_chunk, max_elems, ops, name));
}

static inline void ucc_mpool_cleanup(ucc_mpool_t *mp, int leak_check)
{
    ucs_mpool_ops_t *ops = mp->data->ops;
    ucs_mpool_cleanup(mp, leak_check);
    ucc_free(ops);
}

#endif
