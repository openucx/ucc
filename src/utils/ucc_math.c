/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "ucc/api/ucc.h"
#include "ucc_math.h"

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
