
/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include <chrono>

/* Disabled by default since it will produce UCC and UCX
   errors due to unmatched p2p */
class DISABLED_test_timeout : public ucc::test
{
public:
    ucc_coll_args_t coll;
    DISABLED_test_timeout() {
        coll.mask      = UCC_COLL_ARGS_FIELD_FLAGS;
        coll.coll_type = UCC_COLL_TYPE_BARRIER;
        coll.flags    |= UCC_COLL_ARGS_FLAG_TIMEOUT;
        coll.timeout   = 3; /* 3 sec timout */
    }
};

UCC_TEST_F(DISABLED_test_timeout, single_skip)
{
    UccJob    job(4);
    UccTeam_h team = job.create_team(4);
    UccReq    req(team, &coll);
    int       i;

    /* start request on all procs except 0
       to emulate slow peer for timeout */
    for (i = 1; i < req.reqs.size(); i++) {
        ASSERT_EQ(UCC_OK, ucc_collective_post(req.reqs[i]));
    }

    EXPECT_EQ(UCC_ERR_TIMED_OUT, req.wait());
}

UCC_TEST_F(DISABLED_test_timeout, timeout_not_exceeded)
{
    UccJob       job(4);
    UccTeam_h    team = job.create_team(4);
    UccReq       req(team, &coll);
    ucc_status_t status;
    int          i;
    int          all_posted;

    /* start request on all procs except 0
       to emulate slow peer for timeout */
    for (i = 1; i < req.reqs.size(); i++) {
        ASSERT_EQ(UCC_OK, ucc_collective_post(req.reqs[i]));
    }
    auto start = std::chrono::high_resolution_clock::now();
    all_posted = 0;

    while (UCC_OK != (status = req.test())) {
        if (status < 0) {
            break;
        }
        team->progress();
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start);
        if (!all_posted && (elapsed > std::chrono::seconds(1))) {
            all_posted = 1;
            /* post late req */
            ASSERT_EQ(UCC_OK, ucc_collective_post(req.reqs[0]));
        }
    }

    EXPECT_EQ(UCC_OK, status);
}
