/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2019.  ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include <common/test.h>

extern "C" {
#include <core/ucc_global_opts.h>
}

class test_obj_size : public ucc::test {
};

#define EXPECTED_SIZE(_obj, _size) EXPECT_EQ((size_t)_size, sizeof(_obj))

UCC_TEST_F(test_obj_size, size) {

#if ENABLE_DEBUG_DATA
    UCC_TEST_SKIP_R("Debug data");
#elif defined (ENABLE_STATS)
    UCC_TEST_SKIP_R("Statistic enabled");
#elif UCC_ENABLE_ASSERT
    UCC_TEST_SKIP_R("Assert enabled");
#else

    EXPECTED_SIZE(ucc_global_config_t, 168);

#endif
}
