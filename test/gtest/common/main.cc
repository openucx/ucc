/**
 * Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
 * Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
 * Copyright (C) Huawei Technologies Co., Ltd. 2020.  All rights reserved.
 * See file LICENSE for terms.
 */

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif

#include "test_helpers.h"
#include "test_ucc.h"

static int ucc_gtest_random_seed = -1;

void parse_test_opts(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "s:")) != -1) {
        switch (c) {
        case 's':
            ucc_gtest_random_seed = atoi(optarg);
            break;
        default:
            fprintf(stderr, "Usage: gtest [ -s rand-seed ]\n");
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    int ret;
    // coverity[fun_call_w_exception]: uncaught exceptions cause nonzero exit anyway, so don't warn.
    ::testing::InitGoogleTest(&argc, argv);

    parse_test_opts(argc, argv);
    if (ucc_gtest_random_seed == -1) {
        ucc_gtest_random_seed = time(NULL) % 32768;
    }
    UCC_TEST_MESSAGE << "Using random seed of " << ucc_gtest_random_seed;
    srand(ucc_gtest_random_seed);

    ret = RUN_ALL_TESTS();
    UccJob::cleanup();
    return ret;
}
