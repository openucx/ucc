/**
 * Copyright (c) 2001-2019, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <common/test.h>

extern "C" {
#include <schedule/ucc_schedule.h>
}

class test_obj_size : public ucc::test {
};

#define EXPECTED_SIZE(_obj, _size) EXPECT_EQ((size_t)_size, sizeof(_obj))

UCC_TEST_F(test_obj_size, size) {
    EXPECTED_SIZE(ucc_coll_task_t, 472);
}
