/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "ucc/api/ucc.h"
#include "ucc_math.h"

size_t ucc_dt_sizes[UCC_DT_USERDEFINED] = {
    [UCC_DT_INT8]     = 1,
    [UCC_DT_UINT8]    = 1,
    [UCC_DT_INT16]    = 2,
    [UCC_DT_UINT16]   = 2,
    [UCC_DT_FLOAT16]  = 2,
    [UCC_DT_BFLOAT16] = 2,
    [UCC_DT_INT32]    = 4,
    [UCC_DT_UINT32]   = 4,
    [UCC_DT_FLOAT32]  = 4,
    [UCC_DT_INT64]    = 8,
    [UCC_DT_UINT64]   = 8,
    [UCC_DT_FLOAT64]  = 8,
    [UCC_DT_INT128]   = 16,
    [UCC_DT_UINT128]  = 16,
};

static int _compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

static int _compare_inv(const void *a, const void *b)
{
    return (*(int *)b - *(int *)a);
}

static inline int _unique(int *first, int *last)
{
    int *begin  = first;
    int *result = first;
    if (first == last) {
        return 1;
    }
    while (++first != last) {
        if (!(*result == *first)) {
            *(++result) = *first;
        }
    }
    return (++result - begin);
}

int ucc_sort_uniq(int *array, int len, int inverse)
{
    qsort(array, len, sizeof(int), inverse ? _compare_inv : _compare);
    return _unique(array, array + len);
}
