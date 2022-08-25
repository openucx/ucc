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

UCC_TEST_F(test_obj_size, size) {
    /* lets try to keep it within 8 cache lines
       currently 480b */
    EXPECT_LT(sizeof(ucc_coll_task_t), 64 * 8);
}
