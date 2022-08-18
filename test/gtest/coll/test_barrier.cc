
/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"

class test_barrier : public ucc::test
{
public:
    ucc_coll_args_t coll;
    test_barrier() {
        coll.mask      = 0;
        coll.coll_type = UCC_COLL_TYPE_BARRIER;
    }
};

UCC_TEST_F(test_barrier, single_2proc)
{
    UccTeam_h team = UccJob::getStaticJob()->create_team(2);
    UccReq    req(team, &coll);
    req.start();
    req.wait();
}

UCC_TEST_F(test_barrier, single_max_procs)
{
    UccTeam_h team = UccJob::getStaticTeams().back();
    UccReq    req(team, &coll);
    req.start();
    req.wait();
}

UCC_TEST_F(test_barrier, multiple)
{
    std::vector<UccReq> reqs;
    for (auto &team : UccJob::getStaticTeams()) {
        reqs.push_back(UccReq(team, &coll));
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);
}
