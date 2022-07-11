/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_MPOOL_H_
#define UCC_MPOOL_H_

#include "config.h"
#include "ucc/api/ucc.h"
#include <ucs/datastruct/mpool.h>
#include "ucc_compiler_def.h"
#include "ucc_spinlock.h"

typedef struct ucc_mpool ucc_mpool_t;

typedef struct ucc_mpool_ops {
    ucc_status_t (*chunk_alloc)(ucc_mpool_t *mp, size_t *size_p,
                                void **chunk_p);
    void (*chunk_release)(ucc_mpool_t *mp, void *chunk);
    void (*obj_init)(ucc_mpool_t *mp, void *obj, void *chunk);
    void (*obj_cleanup)(ucc_mpool_t *mp, void *obj);
} ucc_mpool_ops_t;

struct ucc_mpool {
    ucs_mpool_t       super;
    ucc_mpool_ops_t * ucc_ops;
    ucc_thread_mode_t tm;
    ucc_spinlock_t    lock;
};

ucc_status_t ucc_mpool_init(ucc_mpool_t *mp, size_t priv_size, size_t elem_size,
                            size_t align_offset, size_t alignment,
                            unsigned elems_per_chunk, unsigned max_elems,
                            ucc_mpool_ops_t *ops, ucc_thread_mode_t tm,
                            const char *name);

void ucc_mpool_cleanup(ucc_mpool_t *mp, int leak_check);

ucc_status_t ucc_mpool_hugetlb_malloc(ucc_mpool_t *mp, size_t *size_p,
                                      void **chunk_p);

void ucc_mpool_hugetlb_free(ucc_mpool_t *mp, void *chunk);

static inline void *ucc_mpool_get(ucc_mpool_t *mp)
{
    void *ret;

    if (UCC_THREAD_SINGLE == mp->tm) {
        return ucs_mpool_get(&mp->super);
    }
    ucc_spin_lock(&mp->lock);
    ret = ucs_mpool_get(&mp->super);
    ucc_spin_unlock(&mp->lock);
    return ret;
}

static inline void ucc_mpool_put(void *obj)
{
    ucs_mpool_elem_t *elem = (ucs_mpool_elem_t *)obj - 1;
    ucc_mpool_t *     mp   = ucc_derived_of(elem->mpool, ucc_mpool_t);

    if (UCC_THREAD_SINGLE == mp->tm) {
        ucs_mpool_put(obj);
        return;
    }
    ucc_spin_lock(&mp->lock);
    ucs_mpool_put(obj);
    ucc_spin_unlock(&mp->lock);
}

#endif
