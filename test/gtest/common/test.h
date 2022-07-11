/**
 * Copyright (c) 2001-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_BASE_H
#define UCC_TEST_BASE_H

extern "C" {
}

#include <complex.h>
#undef I
#define _Imag _Imaginary_I
#include "gtest.h"

#define UCC_CHECK(_call)    EXPECT_EQ(UCC_OK, (_call))

#define UCC_TEST_SKIP_R(_str) GTEST_SKIP_(_str)

namespace ucc {

class test : public testing::Test {
};

#define UCC_TEST_F(...) TEST_F(__VA_ARGS__)
#define UCC_TEST_P(...) TEST_P(__VA_ARGS__)
}

#define ASSERT_FLOAT32_COMPLEX_EQ(expected, actual)                            \
    do {                                                                       \
        static float expected_real      = crealf(expected);                    \
        static float expected_imaginary = cimagf(expected);                    \
        static float actual_real        = crealf(actual);                      \
        static float actual_imaginary   = cimagf(actual);                      \
        ASSERT_PRED_FORMAT2(                                                   \
            ::testing::internal::CmpHelperFloatingPointEQ<float>,              \
            expected_real, actual_real);                                       \
        ASSERT_PRED_FORMAT2(                                                   \
            ::testing::internal::CmpHelperFloatingPointEQ<float>,              \
            expected_imaginary, actual_imaginary);                             \
    } while (0)
#define ASSERT_FLOAT64_COMPLEX_EQ(expected, actual)                            \
    do {                                                                       \
        static double expected_real      = creal(expected);                    \
        static double expected_imaginary = cimag(expected);                    \
        static double actual_real        = creal(actual);                      \
        static double actual_imaginary   = cimag(actual);                      \
        ASSERT_PRED_FORMAT2(                                                   \
            ::testing::internal::CmpHelperFloatingPointEQ<double>,             \
            expected_real, actual_real);                                       \
        ASSERT_PRED_FORMAT2(                                                   \
            ::testing::internal::CmpHelperFloatingPointEQ<double>,             \
            expected_imaginary, actual_imaginary);                             \
    } while (0)
#define ASSERT_FLOAT128_COMPLEX_EQ(expected, actual)                           \
    do {                                                                       \
        static long double expected_real      = creall(expected);              \
        static long double expected_imaginary = cimagl(expected);              \
        static long double actual_real        = creall(actual);                \
        static long double actual_imaginary   = cimagl(actual);                \
        ASSERT_PRED_FORMAT2(                                                   \
            ::testing::internal::CmpHelperFloatingPointEQ<double>,             \
            expected_real, actual_real);                                       \
        ASSERT_PRED_FORMAT2(                                                   \
            ::testing::internal::CmpHelperFloatingPointEQ<double>,             \
            expected_imaginary, actual_imaginary);                             \
    } while (0)

#define EXPECT_FLOAT32_COMPLEX_EQ(expected, actual)                            \
    static float expected_real      = crealf(expected);                        \
    static float expected_imaginary = cimagf(expected);                        \
    static float actual_real        = crealf(actual);                          \
    static float actual_imaginary   = cimagf(actual);                          \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<float>,  \
                        expected_real, actual_real)                            \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<float>,  \
                        expected_imaginary, actual_imaginary)
#define EXPECT_FLOAT64_COMPLEX_EQ(expected, actual)                            \
    static double expected_real      = creal(expected);                        \
    static double expected_imaginary = cimag(expected);                        \
    static double actual_real        = creal(actual);                          \
    static double actual_imaginary   = cimag(actual);                          \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<double>, \
                        expected_real, actual_real)                            \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<double>, \
                        expected_imaginary, actual_imaginary)
#define EXPECT_FLOAT128_COMPLEX_EQ(expected, actual)                           \
    static long double expected_real      = creall(expected);                  \
    static long double expected_imaginary = cimagl(expected);                  \
    static long double actual_real        = creall(actual);                    \
    static long double actual_imaginary   = cimagl(actual);                    \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<double>, \
                        expected_real, actual_real)                            \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperFloatingPointEQ<double>, \
                        expected_imaginary, actual_imaginary)

#endif
