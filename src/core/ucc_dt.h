/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_DT_H_
#define UCC_DT_H_
#include "config.h"
#include "ucc/api/ucc.h"

typedef struct ucc_dt_generic {
    void                     *context;
    ucc_generic_dt_ops_t     ops;
} ucc_dt_generic_t;

#define UCC_DT_PREDEFINED_ID(_dt) ((_dt) >> UCC_DATATYPE_SHIFT)

#define UCC_DT_IS_GENERIC(_dt)                                    \
    (((_dt) & UCC_DATATYPE_CLASS_MASK) == UCC_DATATYPE_GENERIC)

#define UCC_DT_IS_PREDEFINED(_dt) \
    (((_dt) & UCC_DATATYPE_CLASS_MASK) == UCC_DATATYPE_PREDEFINED)

#define UCC_DT_GENERIC_IS_CONTIG(_dt) (((_dt)->ops.mask & UCC_GENERIC_DT_OPS_FIELD_FLAGS) && \
                                       ((_dt)->ops.flags & UCC_GENERIC_DT_OPS_FLAG_CONTIG))

#define UCC_DT_GENERIC_HAS_REDUCE(_dt) (((_dt)->ops.mask & UCC_GENERIC_DT_OPS_FIELD_FLAGS) && \
                                       ((_dt)->ops.flags & UCC_GENERIC_DT_OPS_FLAG_REDUCE))

#define UCC_DT_IS_CONTIG(_dt) (UCC_DT_IS_GENERIC(_dt) && \
                               UCC_DT_GENERIC_IS_CONTIG(ucc_dt_to_generic(_dt)))

#define UCC_DT_HAS_REDUCE(_dt) (UCC_DT_IS_GENERIC(_dt) && \
                                UCC_DT_GENERIC_HAS_REDUCE(ucc_dt_to_generic(_dt)))

static inline
ucc_dt_generic_t* ucc_dt_to_generic(ucc_datatype_t datatype)
{
    return (ucc_dt_generic_t*)(void*)(datatype & ~UCC_DATATYPE_CLASS_MASK);
}

static inline
ucc_datatype_t ucc_dt_from_generic(ucc_dt_generic_t* dt_gen)
{
    return ((uintptr_t)dt_gen) | UCC_DATATYPE_GENERIC;
}

static inline size_t ucc_contig_dt_size(ucc_datatype_t datatype)
{
    return ucc_dt_to_generic(datatype)->ops.contig_size;
}

extern size_t ucc_dt_predefined_sizes[UCC_DT_PREDEFINED_LAST];

static inline size_t ucc_dt_size(ucc_datatype_t dt)
{
    if (UCC_DT_IS_PREDEFINED(dt)) {
        return ucc_dt_predefined_sizes[UCC_DT_PREDEFINED_ID(dt)];
    } else if (UCC_DT_IS_CONTIG(dt)) {
        return ucc_contig_dt_size(dt);
    }
    // GENERIC callck pack/unpack
    // TODO remove ucc_likely once custom datatype is implemented
    return 0;
}
#endif
