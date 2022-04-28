/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <components/mc/ucc_mc.h>
#include <utils/ucc_math.h>
}
#include <common/test.h>

template<ucc_datatype_t, template <typename P> class op>
struct TypeOpPair;

#define DECLARE_TYPE_OP_PAIR(_type, _TYPE, _EQ)                     \
    template<template <typename P> class op>                        \
    struct TypeOpPair<UCC_DT_ ## _TYPE, op>                         \
    {                                                               \
        using type = _type;                                         \
        const static ucc_datatype_t dt = UCC_DT_ ## _TYPE;          \
        const static ucc_reduction_op_t redop = op<type>::redop;    \
        static void assert_equal(type arg1, type arg2) {            \
            _EQ(arg1, arg2);                                        \
        }                                                           \
        static type do_op(type arg1, type arg2) {                   \
            op<type> _op;                                           \
            return _op(arg1, arg2);                                 \
        }                                                           \
    };

DECLARE_TYPE_OP_PAIR(int8_t, INT8, ASSERT_EQ);
DECLARE_TYPE_OP_PAIR(int16_t, INT16, ASSERT_EQ);
DECLARE_TYPE_OP_PAIR(int32_t, INT32, ASSERT_EQ);
DECLARE_TYPE_OP_PAIR(int64_t, INT64, ASSERT_EQ);
DECLARE_TYPE_OP_PAIR(uint8_t, UINT8, ASSERT_EQ);
DECLARE_TYPE_OP_PAIR(uint16_t, UINT16, ASSERT_EQ);
DECLARE_TYPE_OP_PAIR(uint32_t, UINT32, ASSERT_EQ);
DECLARE_TYPE_OP_PAIR(uint64_t, UINT64, ASSERT_EQ);

//TODO Bfloat Custom
DECLARE_TYPE_OP_PAIR(float, FLOAT32, ASSERT_FLOAT_EQ);
DECLARE_TYPE_OP_PAIR(double, FLOAT64, ASSERT_FLOAT_EQ);

template <template <typename P> class op>
struct TypeOpPair<UCC_DT_BFLOAT16, op> {
    using type                            = uint16_t;
    const static ucc_datatype_t     dt    = UCC_DT_BFLOAT16;
    const static ucc_reduction_op_t redop = op<float>::redop;
    static void                     assert_equal(type arg1, type arg2)
    {
        // near because of different calculation methods - in CPU op across all
        // vectors as fp32, and only then convert to bfloat16. here conversion to fp32
        // is per couple. In CUDA, op is as "real" bfloat16 op.
        // For example - 15*29=435, which doesn't fit in bfloat16.
        // casting from fp32 will result in 434, while in GPU we will get 436.
        ASSERT_NEAR(bfloat16tofloat32(&arg1), bfloat16tofloat32(&arg2),
                    1e-33);
    }
    static type do_op(type arg1, type arg2)
    {
        op<float>  _op;
        uint16_t   res;
        float32tobfloat16(
            _op(bfloat16tofloat32(&arg1), bfloat16tofloat32(&arg2)), &res);
        return res;
    }
};

#define DECLARE_OP_(_op, _UCC_OP, _OP)                          \
    template<typename T>                                        \
    class _op {                                                 \
    public:                                                     \
    const static ucc_reduction_op_t redop = UCC_OP_ ## _UCC_OP; \
    T operator()(T arg1, T arg2) {                              \
        return DO_OP_ ## _OP(arg1, arg2);                       \
    }                                                           \
    };                                                          \

#define DECLARE_OP(_op, _OP) DECLARE_OP_(_op, _OP, _OP)

DECLARE_OP(sum, SUM);
DECLARE_OP(prod, PROD);
DECLARE_OP(min, MIN);
DECLARE_OP(max, MAX);
DECLARE_OP(land, LAND);
DECLARE_OP(lor, LOR);
DECLARE_OP(lxor, LXOR);
DECLARE_OP(band, BAND);
DECLARE_OP(bor, BOR);
DECLARE_OP(bxor, BXOR);
DECLARE_OP_(avg, AVG, SUM);

#define ARITHMETIC_OP_PAIRS(_TYPE) TypeOpPair<UCC_DT_ ## _TYPE, sum>,   \
        TypeOpPair<UCC_DT_ ## _TYPE, prod>,                             \
        TypeOpPair<UCC_DT_ ## _TYPE, min>,                              \
        TypeOpPair<UCC_DT_ ## _TYPE, max>

/* TypeOp pairs that MC Cuda supports */
#define CUDA_OP_PAIRS TypeOpPair<UCC_DT_INT16, lor>,    \
        TypeOpPair<UCC_DT_INT16, sum>,                  \
        TypeOpPair<UCC_DT_INT32, prod>,                 \
        TypeOpPair<UCC_DT_INT64, bxor>,                 \
        TypeOpPair<UCC_DT_FLOAT32, avg>,                \
        ARITHMETIC_OP_PAIRS(INT32),                     \
        ARITHMETIC_OP_PAIRS(FLOAT32),                   \
        ARITHMETIC_OP_PAIRS(FLOAT64),                   \
        ARITHMETIC_OP_PAIRS(BFLOAT16)

using CollReduceTypeOpsCuda = ::testing::Types<CUDA_OP_PAIRS>;

/* Host supports all combos, so add more to tests */
using CollReduceTypeOpsHost = ::testing::Types<CUDA_OP_PAIRS,
                                           TypeOpPair<UCC_DT_UINT16, band>,
                                           TypeOpPair<UCC_DT_UINT32, bor>,
                                           TypeOpPair<UCC_DT_UINT64, land>,
                                           TypeOpPair<UCC_DT_UINT8, lxor>,
                                           TypeOpPair<UCC_DT_INT8, lor>>;

using CollReduceTypeOpsAvg = ::testing::Types<TypeOpPair<UCC_DT_FLOAT32, avg>,
                                              TypeOpPair<UCC_DT_FLOAT64, avg>,
                                              TypeOpPair<UCC_DT_BFLOAT16, avg>>;

static inline bool ucc_reduction_is_supported(ucc_datatype_t dt,
                                              ucc_reduction_op_t op,
                                              ucc_memory_type_t mt)
{
    switch(op) {
    case UCC_OP_AVG:
        switch(dt) {
        case UCC_DT_FLOAT32:
        case UCC_DT_FLOAT64:
        case UCC_DT_BFLOAT16:
            break;
        default:
            return false;
        }
        break;
    default:
        break;
    }
    return true;
}

#define CHECK_TYPE_OP_SKIP(_dt, _op, _mt) do {              \
        if (!ucc_reduction_is_supported(_dt, _op, _mt)) {   \
            GTEST_SKIP();                                   \
        }                                                   \
    } while(0)
