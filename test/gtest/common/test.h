/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2022.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_BASE_H
#define UCC_TEST_BASE_H

extern "C" {
}
#include "gtest.h"


#define UCC_CHECK(_call)    EXPECT_EQ(UCC_OK, (_call))

#define UCC_TEST_SKIP_R(_str) GTEST_SKIP_(_str)

namespace ucc {

class test : public testing::Test {
};

#define UCC_TEST_F(...) TEST_F(__VA_ARGS__)
#define UCC_TEST_P(...) TEST_P(__VA_ARGS__)
}
#endif
