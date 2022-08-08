/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MATH_OP_H_
#define UCC_MATH_OP_H_
#include "ucc_math.h"

#define DO_OP_2(_op, _v1, _v2)           (_v1 _op _v2)
#define DO_OP_3(_op, _v1, _v2, _v3)      (_v1 _op _v2 _op _v3)
#define DO_OP_4(_op, _v1, _v2, _v3, _v4) (_v1 _op _v2 _op _v3 _op _v4)
#define DO_OP_5(_op, _v1, _v2, _v3, _v4, _v5)                                  \
    (_v1 _op _v2 _op _v3 _op _v4 _op _v5)
#define DO_OP_6(_op, _v1, _v2, _v3, _v4, _v5, _v6)                             \
    (_v1 _op _v2 _op _v3 _op _v4 _op _v5 _op _v6)
#define DO_OP_7(_op, _v1, _v2, _v3, _v4, _v5, _v6, _v7)                        \
    (_v1 _op _v2 _op _v3 _op _v4 _op _v5 _op _v6 _op _v7)
#define DO_OP_8(_op, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8)                   \
    (_v1 _op _v2 _op _v3 _op _v4 _op _v5 _op _v6 _op _v7 _op _v8)

#define DO_OP_SUM_2(...) DO_OP_2(+, __VA_ARGS__)
#define DO_OP_SUM_3(...) DO_OP_3(+, __VA_ARGS__)
#define DO_OP_SUM_4(...) DO_OP_4(+, __VA_ARGS__)
#define DO_OP_SUM_5(...) DO_OP_5(+, __VA_ARGS__)
#define DO_OP_SUM_6(...) DO_OP_6(+, __VA_ARGS__)
#define DO_OP_SUM_7(...) DO_OP_7(+, __VA_ARGS__)
#define DO_OP_SUM_8(...) DO_OP_8(+, __VA_ARGS__)

#define DO_OP_PROD_2(...) DO_OP_2(*, __VA_ARGS__)
#define DO_OP_PROD_3(...) DO_OP_3(*, __VA_ARGS__)
#define DO_OP_PROD_4(...) DO_OP_4(*, __VA_ARGS__)
#define DO_OP_PROD_5(...) DO_OP_5(*, __VA_ARGS__)
#define DO_OP_PROD_6(...) DO_OP_6(*, __VA_ARGS__)
#define DO_OP_PROD_7(...) DO_OP_7(*, __VA_ARGS__)
#define DO_OP_PROD_8(...) DO_OP_8(*, __VA_ARGS__)

#define DO_OP_LAND_2(...) DO_OP_2(&&, __VA_ARGS__)
#define DO_OP_LAND_3(...) DO_OP_3(&&, __VA_ARGS__)
#define DO_OP_LAND_4(...) DO_OP_4(&&, __VA_ARGS__)
#define DO_OP_LAND_5(...) DO_OP_5(&&, __VA_ARGS__)
#define DO_OP_LAND_6(...) DO_OP_6(&&, __VA_ARGS__)
#define DO_OP_LAND_7(...) DO_OP_7(&&, __VA_ARGS__)
#define DO_OP_LAND_8(...) DO_OP_8(&&, __VA_ARGS__)

#define DO_OP_BAND_2(...) DO_OP_2(&, __VA_ARGS__)
#define DO_OP_BAND_3(...) DO_OP_3(&, __VA_ARGS__)
#define DO_OP_BAND_4(...) DO_OP_4(&, __VA_ARGS__)
#define DO_OP_BAND_5(...) DO_OP_5(&, __VA_ARGS__)
#define DO_OP_BAND_6(...) DO_OP_6(&, __VA_ARGS__)
#define DO_OP_BAND_7(...) DO_OP_7(&, __VA_ARGS__)
#define DO_OP_BAND_8(...) DO_OP_8(&, __VA_ARGS__)

#define DO_OP_LOR_2(...) DO_OP_2(||, __VA_ARGS__)
#define DO_OP_LOR_3(...) DO_OP_3(||, __VA_ARGS__)
#define DO_OP_LOR_4(...) DO_OP_4(||, __VA_ARGS__)
#define DO_OP_LOR_5(...) DO_OP_5(||, __VA_ARGS__)
#define DO_OP_LOR_6(...) DO_OP_6(||, __VA_ARGS__)
#define DO_OP_LOR_7(...) DO_OP_7(||, __VA_ARGS__)
#define DO_OP_LOR_8(...) DO_OP_8(||, __VA_ARGS__)

#define DO_OP_BOR_2(...) DO_OP_2(|, __VA_ARGS__)
#define DO_OP_BOR_3(...) DO_OP_3(|, __VA_ARGS__)
#define DO_OP_BOR_4(...) DO_OP_4(|, __VA_ARGS__)
#define DO_OP_BOR_5(...) DO_OP_5(|, __VA_ARGS__)
#define DO_OP_BOR_6(...) DO_OP_6(|, __VA_ARGS__)
#define DO_OP_BOR_7(...) DO_OP_7(|, __VA_ARGS__)
#define DO_OP_BOR_8(...) DO_OP_8(|, __VA_ARGS__)

#define DO_OP_BXOR_2(...) DO_OP_2(^, __VA_ARGS__)
#define DO_OP_BXOR_3(...) DO_OP_3(^, __VA_ARGS__)
#define DO_OP_BXOR_4(...) DO_OP_4(^, __VA_ARGS__)
#define DO_OP_BXOR_5(...) DO_OP_5(^, __VA_ARGS__)
#define DO_OP_BXOR_6(...) DO_OP_6(^, __VA_ARGS__)
#define DO_OP_BXOR_7(...) DO_OP_7(^, __VA_ARGS__)
#define DO_OP_BXOR_8(...) DO_OP_8(^, __VA_ARGS__)

#define DO_OP__3(_OP, _v1, _v2, _v3)      _OP(_OP(_v1, _v2), _v3)
#define DO_OP__4(_OP, _v1, _v2, _v3, _v4) _OP(_OP(_v1, _v2), _OP(_v3, _v4))
#define DO_OP__5(_OP, _v1, _v2, _v3, _v4, _v5)                                 \
    _OP(_OP(_v1, _v2), DO_OP__3(_OP, _v3, _v4, _v5))
#define DO_OP__6(_OP, _v1, _v2, _v3, _v4, _v5, _v6)                            \
    _OP(DO_OP__3(_OP, _v1, _v2, _v3), DO_OP__3(_OP, _v4, _v5, _v6))
#define DO_OP__7(_OP, _v1, _v2, _v3, _v4, _v5, _v6, _v7)                       \
    _OP(DO_OP__3(_OP, _v1, _v2, _v3), DO_OP__4(_OP, _v4, _v5, _v6, _v7))
#define DO_OP__8(_OP, _v1, _v2, _v3, _v4, _v5, _v6, _v7, _v8)                  \
    _OP(DO_OP__4(_OP, _v1, _v2, _v3, _v4), DO_OP__4(_OP, _v5, _v6, _v7, _v8))

#define DO_OP_MAX_2(_v1, _v2) DO_OP_MAX(_v1, _v2)
#define DO_OP_MAX_3(...)      DO_OP__3(DO_OP_MAX, __VA_ARGS__)
#define DO_OP_MAX_4(...)      DO_OP__4(DO_OP_MAX, __VA_ARGS__)
#define DO_OP_MAX_5(...)      DO_OP__5(DO_OP_MAX, __VA_ARGS__)
#define DO_OP_MAX_6(...)      DO_OP__6(DO_OP_MAX, __VA_ARGS__)
#define DO_OP_MAX_7(...)      DO_OP__7(DO_OP_MAX, __VA_ARGS__)
#define DO_OP_MAX_8(...)      DO_OP__8(DO_OP_MAX, __VA_ARGS__)

#define DO_OP_MIN_2(_v1, _v2) DO_OP_MIN(_v1, _v2)
#define DO_OP_MIN_3(...)      DO_OP__3(DO_OP_MIN, __VA_ARGS__)
#define DO_OP_MIN_4(...)      DO_OP__4(DO_OP_MIN, __VA_ARGS__)
#define DO_OP_MIN_5(...)      DO_OP__5(DO_OP_MIN, __VA_ARGS__)
#define DO_OP_MIN_6(...)      DO_OP__6(DO_OP_MIN, __VA_ARGS__)
#define DO_OP_MIN_7(...)      DO_OP__7(DO_OP_MIN, __VA_ARGS__)
#define DO_OP_MIN_8(...)      DO_OP__8(DO_OP_MIN, __VA_ARGS__)

#define DO_OP_LXOR_2(_v1, _v2) DO_OP_LXOR(_v1, _v2)
#define DO_OP_LXOR_3(...)      DO_OP__3(DO_OP_LXOR, __VA_ARGS__)
#define DO_OP_LXOR_4(...)      DO_OP__4(DO_OP_LXOR, __VA_ARGS__)
#define DO_OP_LXOR_5(...)      DO_OP__5(DO_OP_LXOR, __VA_ARGS__)
#define DO_OP_LXOR_6(...)      DO_OP__6(DO_OP_LXOR, __VA_ARGS__)
#define DO_OP_LXOR_7(...)      DO_OP__7(DO_OP_LXOR, __VA_ARGS__)
#define DO_OP_LXOR_8(...)      DO_OP__8(DO_OP_LXOR, __VA_ARGS__)

#endif
