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
    ucc_memory_type_t mem_type;
    virtual void SetUp() override
    {
        ucc_constructor();
        ucc_mc_init();
        buf1_h = buf2_h = res_h = nullptr;
        buf1_d = buf2_d = res_d = nullptr;
    }

    ucc_status_t alloc_bufs(ucc_memory_type_t mtype, size_t n)
    {
        size_t n_bytes = COUNT*sizeof(typename T::type);
        mem_type = mtype;

        ucc_mc_alloc((void**)&res_h, n_bytes, UCC_MEMORY_TYPE_HOST);
        ucc_mc_alloc((void**)&buf1_h, n_bytes, UCC_MEMORY_TYPE_HOST);
        ucc_mc_alloc((void**)&buf2_h, n * n_bytes, UCC_MEMORY_TYPE_HOST);

        for (int i = 0; i < COUNT; i++) {
            res_h[i] = (typename T::type)(0);
        }
        for (int i = 0; i < COUNT; i++) {
            buf1_h[i] = (typename T::type)(i + 1);
        }
        for (int j = 0; j < n; j++) {
            for (int i = 0; i < COUNT; i++) {
                buf2_h[i + j * COUNT] =  (typename T::type)(2 * i + j + 1);
            }
        }
        if (mtype != UCC_MEMORY_TYPE_HOST) {
            ucc_mc_alloc((void**)&res_d, n_bytes, mtype);
            ucc_mc_alloc((void**)&buf1_d, n_bytes, mtype);
            ucc_mc_alloc((void**)&buf2_d, n*n_bytes, mtype);
            ucc_mc_memcpy(res_d, res_h, n_bytes, mtype, UCC_MEMORY_TYPE_HOST);
            ucc_mc_memcpy(buf1_d, buf1_h, n_bytes, mtype, UCC_MEMORY_TYPE_HOST);
            ucc_mc_memcpy(buf2_d, buf2_h, n * n_bytes, mtype,
                          UCC_MEMORY_TYPE_HOST);
        }

        return UCC_OK;
    }

    ucc_status_t free_bufs(ucc_memory_type_t mtype)
    {
        if (buf1_h != nullptr) {
            ucc_mc_free((void*)buf1_h, UCC_MEMORY_TYPE_HOST);
        }
        if (buf2_h != nullptr) {
            ucc_mc_free((void*)buf2_h, UCC_MEMORY_TYPE_HOST);
        }
        if (res_h != nullptr) {
            ucc_mc_free((void*)res_h, UCC_MEMORY_TYPE_HOST);
        }
        if (buf1_d != nullptr) {
            ucc_mc_free((void*)buf1_d, mtype);
        }
        if (buf2_d != nullptr) {
            ucc_mc_free((void*)buf2_d, mtype);
        }
        if (res_d != nullptr) {
            ucc_mc_free((void*)res_d, mtype);
        }

        return UCC_OK;
    }

    virtual void TearDown() override
    {
        free_bufs(mem_type);
        ucc_mc_finalize();
    }
    typename T::type *buf1_h;
    typename T::type *buf2_h;
    typename T::type *res_h;

    typename T::type *buf1_d;
    typename T::type *buf2_d;
    typename T::type *res_d;

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
