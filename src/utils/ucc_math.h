/**
 * Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MATH_H_
#define UCC_MATH_H_

#include "config.h"
#include <ucs/sys/math.h>
#include "ucc_datastruct.h"
#include "ucc/api/ucc.h"
#include "ucc_compiler_def.h"

#define ucc_min(_a, _b) ucs_min((_a), (_b))
#define ucc_max(_a, _b) ucs_max((_a), (_b))
#define ucc_ilog2(_v)   ucs_ilog2((_v))
#define ucc_ceil(_a, _b) (((_a)%(_b))?((_a)/(_b)+1)*(_b):(_a))

#define DO_OP_MAX(_v1, _v2) (_v1 > _v2 ? _v1 : _v2)
#define DO_OP_MIN(_v1, _v2) (_v1 < _v2 ? _v1 : _v2)
#define DO_OP_SUM(_v1, _v2) (_v1 + _v2)
#define DO_OP_PROD(_v1, _v2) (_v1 * _v2)
#define DO_OP_LAND(_v1, _v2) (_v1 && _v2)
#define DO_OP_BAND(_v1, _v2) (_v1 & _v2)
#define DO_OP_LOR(_v1, _v2) (_v1 || _v2)
#define DO_OP_BOR(_v1, _v2) (_v1 | _v2)
#define DO_OP_LXOR(_v1, _v2) ((!_v1) != (!_v2))
#define DO_OP_BXOR(_v1, _v2) (_v1 ^ _v2)

#define PTR_OFFSET(_ptr, _offset)                                              \
    ((void *)((ptrdiff_t)(_ptr) + (size_t)(_offset)))

/* Returns the number of trailing 0-bits in x, starting at the least
 * significant bit position.  If x is 0, the result is undefined.
 */
#define ucc_count_trailing_zero_bits(_n) \
    ((sizeof(_n) <= 4) ? __builtin_ctz((uint32_t)(_n)) : __builtin_ctzl(_n))

/* Returns the number of leading 0-bits in _n.
 * If _n is 0, the result is undefined
 */
#define ucc_count_leading_zero_bits(_n) \
    ((sizeof(_n) <= 4) ? __builtin_clz((uint32_t)(_n)) : __builtin_clzl(_n))

/* http://www.cse.yorku.ca/~oz/hash.html - Dan Bernstein string
   hash function */
static inline unsigned long ucc_str_hash_djb2(const char *str)
{
    unsigned long hash = 5381;
    int           c;

    while ('\0' != (c = *str++)) {
        hash = ((hash << 5) + hash) + c;
    }
    return hash;
}

/* Sorts the input integer array in-place keeping only
   unique elements */
int ucc_sort_uniq(int *array, int len, int inverse);

#define SWAP(_x, _y, _type)                                                    \
    do {                                                                       \
        _type _tmp = (_x);                                                     \
        (_x)       = (_y);                                                     \
        (_y)       = _tmp;                                                     \
    } while (0)

#define ucc_div_round_up(_n, _d) (((_n) + (_d) - 1) / (_d))

// compute the greatest exponent v such that p**v divides n
static inline uint32_t calc_valuation(uint32_t n, uint32_t p)
{
    uint32_t v = 0;
    uint32_t q         = n;
    while (q % p == 0) {
        v++;
        q /= p;
    }
    return v;
}

static inline float bfloat16tofloat32(const void *bfloat16_ptr)
{
    float res = 0;
#if UCC_BIG_ENDIAN
    ((uint16_t *)(&res))[0] = *((uint16_t *)bfloat16_ptr);
#else
    ((uint16_t *)(&res))[1] = *((uint16_t *)bfloat16_ptr);
#endif
    return res;
}

static inline void float32tobfloat16(float float_val, void *bfloat16_ptr)
{
#if UCC_BIG_ENDIAN
    *((uint16_t *)bfloat16_ptr) = ((uint16_t *)(&float_val))[0];
#else
    *((uint16_t *)bfloat16_ptr) = ((uint16_t *)(&float_val))[1];
#endif
}

#define ucc_padding(_n, _alignment)                                            \
    ( ((_alignment) - (_n) % (_alignment)) % (_alignment) )

#define ucc_align_down(_n, _alignment)                                         \
    ( (_n) - ((_n) % (_alignment)) )

#define ucc_align_up(_n, _alignment)                                           \
    ( (_n) + ucc_padding(_n, _alignment) )

#define ucc_align_down_pow2(_n, _alignment)                                    \
    ( (_n) & ~((_alignment) - 1) )

#define ucc_align_up_pow2(_n, _alignment)                                      \
    ucc_align_down_pow2((_n) + (_alignment) - 1, _alignment)

/* compute the log2 of n, rounded up */
static inline int lognum(int n)
{
    int count  = 1;
    int lognum = 0;

    while (count < n) {
        count = count << 1;
        lognum++;
    }
    return lognum;
}

/* return log2 of nearest greater or equal power of 2 */
static inline uint32_t ucc_ilog2_ceil(uint32_t n)
{
    int x = ucc_count_leading_zero_bits(n); // leading zeros
    if (!(n & (n - 1))) {
        return 31 - x; // pow2 case
    }
    return 31 - x + 1;
}

/* return the lowest greater or equal power of 2 */
static inline int ucc_lowest_greater_power2(int value)
{
    int p = 1;
    while (p < value) {
        p *= 2;
    }
    return p;
}

#endif
