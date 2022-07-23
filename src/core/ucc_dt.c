/**
 * Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "ucc_dt.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_math.h"

size_t ucc_dt_predefined_sizes[UCC_DT_PREDEFINED_LAST] = {
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT8)]             = 1,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT8)]            = 1,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT16)]            = 2,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT16)]           = 2,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT16)]          = 2,
    [UCC_DT_PREDEFINED_ID(UCC_DT_BFLOAT16)]         = 2,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT32)]            = 4,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT32)]           = 4,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32)]          = 4,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT64)]            = 8,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT64)]           = 8,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64)]          = 8,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT128)]         = 16,
    [UCC_DT_PREDEFINED_ID(UCC_DT_INT128)]           = 16,
    [UCC_DT_PREDEFINED_ID(UCC_DT_UINT128)]          = 16,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT32_COMPLEX)]  = 8,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT64_COMPLEX)]  = 16,
    [UCC_DT_PREDEFINED_ID(UCC_DT_FLOAT128_COMPLEX)] = 32};

ucc_status_t ucc_dt_create_generic(const ucc_generic_dt_ops_t *ops, void *context,
                                   ucc_datatype_t *datatype_p)
{
    ucc_dt_generic_t *dt_gen;
    int ret;

    ret = ucc_posix_memalign((void **)&dt_gen,
                             ucc_max(sizeof(void *), UCC_BIT(UCC_DATATYPE_SHIFT)),
                             sizeof(*dt_gen), "generic_dt");
    if (ret != 0) {
        return UCC_ERR_NO_MEMORY;
    }

    dt_gen->ops     = *ops;
    dt_gen->context = context;
    *datatype_p     = ucc_dt_from_generic(dt_gen);
    return UCC_OK;
}

void ucc_dt_destroy(ucc_datatype_t datatype)
{
    ucc_dt_generic_t *dt_gen;

    if (UCC_DT_IS_GENERIC(datatype)) {
        dt_gen = ucc_dt_to_generic(datatype);
        ucc_free(dt_gen);
    }
}
