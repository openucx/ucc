/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
extern "C" {
#include "utils/ucc_math.h"
}
#include <common/test.h>

using floatParams = float;
class test_floats_cast : public ucc::test,
                         public ::testing::WithParamInterface<floatParams> {
};

UCC_TEST_P(test_floats_cast, start_with_float)
{
    float    p = GetParam();
    uint16_t bfloat16Val;
    float32tobfloat16(p, &bfloat16Val);
    // All values in test_floats_cast use only first 16 bits of the 32. rest are 0.
    EXPECT_EQ(p, bfloat16tofloat32(&bfloat16Val));
}

INSTANTIATE_TEST_CASE_P(, test_floats_cast,
                        ::testing::Values(-4.26941244675e+18,
                                          -6.95441104568e+13, 2.015625));

using bfloat16Params = uint16_t;
class test_bfloats16_cast
    : public ucc::test,
      public ::testing::WithParamInterface<bfloat16Params> {
};

UCC_TEST_P(test_bfloats16_cast, start_with_bfloat16)
{
    uint16_t p = GetParam();
    uint16_t res;
    float32tobfloat16(bfloat16tofloat32(&p), &res);
    EXPECT_EQ(p, res);
}

INSTANTIATE_TEST_CASE_P(, test_bfloats16_cast,
                        ::testing::Values(31000, 400, 17, 13569, 0));
