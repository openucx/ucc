/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include <common/test.h>

template<ucc_datatype_t, template <typename P> class op>
struct ReductionTest;

template<template <typename P> class op>
struct ReductionTest<UCC_DT_INT16, op>
{
    using type = int16_t;
    const static ucc_datatype_t dt = UCC_DT_INT16;
    const static ucc_reduction_op_t redop = op<type>::redop;
    static void assert_equal(type arg1, type arg2) {
        ASSERT_EQ(arg1, arg2);
    }
    static type do_op(type arg1, type arg2) {
        op<type> _op;
        return _op(arg1, arg2);
    }
};

template<template <typename P> class op>
struct ReductionTest<UCC_DT_INT32, op>
{
    using type = int32_t;
    const static ucc_datatype_t dt = UCC_DT_INT32;
    const static ucc_reduction_op_t redop = op<type>::redop;
    static void assert_equal(type arg1, type arg2) {
        ASSERT_EQ(arg1, arg2);
    }
    static type do_op(type arg1, type arg2) {
        op<type> _op;
        return _op(arg1, arg2);
    }
};

template<template <typename P> class op>
struct ReductionTest<UCC_DT_INT64, op>
{
    using type = int64_t;
    const static ucc_datatype_t dt = UCC_DT_INT64;
    const static ucc_reduction_op_t redop = op<type>::redop;
    static void assert_equal(type arg1, type arg2) {
        ASSERT_EQ(arg1, arg2);
    }
    static type do_op(type arg1, type arg2) {
        op<type> _op;
        return _op(arg1, arg2);
    }
};

template<template <typename P> class op>
struct ReductionTest<UCC_DT_FLOAT32, op>
{
    using type = float;
    const static ucc_datatype_t dt = UCC_DT_FLOAT32;
    const static ucc_reduction_op_t redop = op<type>::redop;
    static void assert_equal(type arg1, type arg2) {
        ASSERT_FLOAT_EQ(arg1, arg2);
    }
    static type do_op(type arg1, type arg2) {
        op<type> _op;
        return _op(arg1, arg2);
    }
};

template<template <typename P> class op>
struct ReductionTest<UCC_DT_FLOAT64, op>
{
    using type = double;
    const static ucc_datatype_t dt = UCC_DT_FLOAT64;
    const static ucc_reduction_op_t redop = op<type>::redop;
    static void assert_equal(type arg1, type arg2) {
        ASSERT_DOUBLE_EQ(arg1, arg2);
    }
    static type do_op(type arg1, type arg2) {
        op<type> _op;
        return _op(arg1, arg2);
    }
};

template<typename T>
class sum {
public:
    const static ucc_reduction_op_t redop = UCC_OP_SUM;
    T operator()(T arg1, T arg2) {
        return arg1 + arg2;
    }
};

template<typename T>
class prod {
public:
    const static ucc_reduction_op_t redop = UCC_OP_PROD;
    T operator()(T arg1, T arg2) {
        return arg1 * arg2;
    }
};

template<typename T>
class max {
public:
    const static ucc_reduction_op_t redop = UCC_OP_MAX;
    T operator()(T arg1, T arg2) {
        return arg1 > arg2 ? arg1: arg2;
    }
};

template<typename T>
class min {
public:
    const static ucc_reduction_op_t redop = UCC_OP_MIN;
    T operator()(T arg1, T arg2) {
        return arg1 < arg2 ? arg1: arg2;
    }
};

template<typename T>
class land {
public:
    const static ucc_reduction_op_t redop = UCC_OP_LAND;
    T operator()(T arg1, T arg2) {
        return arg1 && arg2;
    }
};

template<typename T>
class band {
public:
    const static ucc_reduction_op_t redop = UCC_OP_BAND;
    T operator()(T arg1, T arg2) {
        return arg1 & arg2;
    }
};

template<typename T>
class lor {
public:
    const static ucc_reduction_op_t redop = UCC_OP_LOR;
    T operator()(T arg1, T arg2) {
        return arg1 || arg2;
    }
};

template<typename T>
class bor {
public:
    const static ucc_reduction_op_t redop = UCC_OP_BOR;
    T operator()(T arg1, T arg2) {
        return arg1 | arg2;
    }
};

template<typename T>
class lxor {
public:
    const static ucc_reduction_op_t redop = UCC_OP_LXOR;
    T operator()(T arg1, T arg2) {
        return !arg1 != !arg2;
    }
};

template<typename T>
class bxor {
public:
    const static ucc_reduction_op_t redop = UCC_OP_BXOR;
    T operator()(T arg1, T arg2) {
        return arg1 ^ arg2;
    }
};

template<typename T>
class test_mc_reduce : public testing::Test {
  protected:
    const int COUNT = 1024;
    virtual void SetUp() override
    {
        ucc_constructor();
        ucc_mc_init();
        ucc_mc_alloc((void**)&buf1, COUNT*sizeof(*buf1), UCC_MEMORY_TYPE_HOST);
        ucc_mc_alloc((void**)&buf2, COUNT*sizeof(*buf2), UCC_MEMORY_TYPE_HOST);
        ucc_mc_alloc((void**)&res, COUNT*sizeof(*res), UCC_MEMORY_TYPE_HOST);
        for (int i = 0 ; i < COUNT; i++) {
            buf1[i] = (typename T::type)(i + 1);
            buf2[i] = (typename T::type)(2 * i + 1);
            res[i]  = (typename T::type)(0);
        }

    }
    virtual void TearDown() override
    {
        ucc_mc_free((void*)buf1, UCC_MEMORY_TYPE_HOST);
        ucc_mc_free((void*)buf2, UCC_MEMORY_TYPE_HOST);
        ucc_mc_free((void*)res, UCC_MEMORY_TYPE_HOST);
        ucc_mc_finalize();
    }
    typename T::type *buf1;
    typename T::type *buf2;
    typename T::type *res;
};

using ReductionTypesOps = ::testing::Types<ReductionTest<UCC_DT_INT16, max>,
                                           ReductionTest<UCC_DT_INT32, max>,
                                           ReductionTest<UCC_DT_INT64, max>,
                                           ReductionTest<UCC_DT_INT16, min>,
                                           ReductionTest<UCC_DT_INT32, min>,
                                           ReductionTest<UCC_DT_INT64, min>,
                                           ReductionTest<UCC_DT_INT16, sum>,
                                           ReductionTest<UCC_DT_INT32, sum>,
                                           ReductionTest<UCC_DT_INT64, sum>,
                                           ReductionTest<UCC_DT_INT16, prod>,
                                           ReductionTest<UCC_DT_INT32, prod>,
                                           ReductionTest<UCC_DT_INT64, prod>,
                                           ReductionTest<UCC_DT_INT16, land>,
                                           ReductionTest<UCC_DT_INT32, land>,
                                           ReductionTest<UCC_DT_INT64, land>,
                                           ReductionTest<UCC_DT_INT16, band>,
                                           ReductionTest<UCC_DT_INT32, band>,
                                           ReductionTest<UCC_DT_INT64, band>,
                                           ReductionTest<UCC_DT_INT16, lor>,
                                           ReductionTest<UCC_DT_INT32, lor>,
                                           ReductionTest<UCC_DT_INT64, lor>,
                                           ReductionTest<UCC_DT_INT16, bor>,
                                           ReductionTest<UCC_DT_INT32, bor>,
                                           ReductionTest<UCC_DT_INT64, bor>,
                                           ReductionTest<UCC_DT_INT16, lxor>,
                                           ReductionTest<UCC_DT_INT32, lxor>,
                                           ReductionTest<UCC_DT_INT64, lxor>,
                                           ReductionTest<UCC_DT_INT16, bxor>,
                                           ReductionTest<UCC_DT_INT32, bxor>,
                                           ReductionTest<UCC_DT_INT64, bxor>,
                                           ReductionTest<UCC_DT_FLOAT32, max>,
                                           ReductionTest<UCC_DT_FLOAT64, max>,
                                           ReductionTest<UCC_DT_FLOAT32, min>,
                                           ReductionTest<UCC_DT_FLOAT64, min>,
                                           ReductionTest<UCC_DT_FLOAT32, sum>,
                                           ReductionTest<UCC_DT_FLOAT64, sum>,
                                           ReductionTest<UCC_DT_FLOAT32, prod>,
                                           ReductionTest<UCC_DT_FLOAT64, prod>>;

TYPED_TEST_CASE(test_mc_reduce, ReductionTypesOps);
TYPED_TEST(test_mc_reduce, ucc_op_sum) {
    ucc_mc_reduce(this->buf1, this->buf2, this->res,
                  this->COUNT, TypeParam::dt,
                  TypeParam::redop, UCC_MEMORY_TYPE_HOST);
    for (int i = 0; i < this->COUNT; i++) {
        TypeParam::assert_equal(TypeParam::do_op(this->buf1[i],this->buf2[i]),
                                this->res[i]);
    }
}
