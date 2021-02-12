/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#include "common/test_ucc.h"
extern "C" {
#include "core/ucc_team.h"
}
#include <algorithm>
#include <random>

class test_team : public ucc::test, public::testing::WithParamInterface<int> {
};

/* Create and destroy team of different size, 1 at a time */
UCC_TEST_P(test_team, team_create_destroy)
{
    UccTeam_h team = UccJob::getStaticJob()->create_team(GetParam());
}

INSTANTIATE_TEST_CASE_P(, test_team,
                        ::testing::Values(
                            2, /* Minimal team size   */
                            8, /* Some power of 2     */
                            7  /* Some non-power of 2 */
                            ));

/* Create and destroy several coexisting teams */
UCC_TEST_F(test_team, team_create_multiple)
{
    UccJob *job  = UccJob::getStaticJob();
    int job_size = job->n_procs;
    int n_teams  = 4; /* how many teams to create */
    std::vector<UccTeam_h> teams;
    for (int i = 0; i < n_teams; i++) {
        int team_size = 2 + (rand() % (job_size - 2 + 1));
        teams.push_back(job->create_team(team_size));
    }
    /* shuffle vector so that teams are destroyed in different order */
    std::shuffle(teams.begin(), teams.end(), std::default_random_engine());
}

