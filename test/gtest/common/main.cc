/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2013.  ALL RIGHTS RESERVED.
* Copyright (C) UT-Battelle, LLC. 2014. ALL RIGHTS RESERVED.
* See file LICENSE for terms.
*/

#ifdef HAVE_CONFIG_H
#  include "config.h"
#endif
//#include <ucs/config/parser.h>
//#include <core/global_opts.h>

#include "test_helpers.h"
#include "tap.h"


static int ucs_gtest_random_seed = -1;
int ucc::perf_retry_count        = 0; /* 0 - don't check performance */
double ucc::perf_retry_interval  = 1.0;


void parse_test_opts(int argc, char **argv) {
    int c;
    while ((c = getopt(argc, argv, "s:p:i:")) != -1) {
        switch (c) {
        case 's':
            ucs_gtest_random_seed = atoi(optarg);
            break;
        case 'p':
            ucc::perf_retry_count = atoi(optarg);
            break;
        case 'i':
            ucc::perf_retry_interval = atof(optarg);
            break;
        default:
            fprintf(stderr, "Usage: gtest [ -s rand-seed ] [ -p count ] [ -i interval ]\n");
            exit(1);
        }
    }
}

int main(int argc, char **argv) {
    // coverity[fun_call_w_exception]: uncaught exceptions cause nonzero exit anyway, so don't warn.
    ::testing::InitGoogleTest(&argc, argv);

    char *str = getenv("GTEST_TAP");
    int ret;

    /* Append TAP Listener */
    if (str) {
        if (0 < strtol(str, NULL, 0)) {
            testing::TestEventListeners& listeners = testing::UnitTest::GetInstance()->listeners();
            if (1 == strtol(str, NULL, 0)) {
                delete listeners.Release(listeners.default_result_printer());
            }
            listeners.Append(new tap::TapListener());
        }
    }

    parse_test_opts(argc, argv);
    if (ucs_gtest_random_seed == -1) {
        ucs_gtest_random_seed = time(NULL) % 32768;
    }
    UCS_TEST_MESSAGE << "Using random seed of " << ucs_gtest_random_seed;
    srand(ucs_gtest_random_seed);

    ret = ucc::watchdog_start();
    if (ret != 0) {
        ADD_FAILURE() << "Unable to start watchdog - abort";
        return ret;
    }

    ret = RUN_ALL_TESTS();

    ucc::watchdog_stop();

    ucc::analyze_test_results();

    return ret;
}
