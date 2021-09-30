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
UCC_TEST_P(test_team, team_create_destroy_ctx_global)
{
    /* static job uses global ctx mode: oob provided */
    UccTeam_h team = UccJob::getStaticJob()->create_team(GetParam());
}

UCC_TEST_P(test_team, team_create_destroy_ctx_local)
{
    UccJob job(8, UccJob::UCC_JOB_CTX_LOCAL);
    UccTeam_h team = job.create_team(GetParam());
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

/* Create and destroy several coexisting teams */
UCC_TEST_F(test_team, team_create_multiple_preconnect_ctx_local)
{
    int job_size = 16;
    UccJob job(job_size, UccJob::UCC_JOB_CTX_LOCAL,
               {ucc_env_var_t("UCC_TL_UCP_PRECONNECT", "inf")});
    int n_teams  = 4; /* how many teams to create */
    std::vector<UccTeam_h> teams;
    for (int i = 0; i < n_teams; i++) {
        int team_size = 2 + (rand() % (job_size - 2 + 1));
        teams.push_back(job.create_team(team_size));
    }
    /* shuffle vector so that teams are destroyed in different order */
    std::shuffle(teams.begin(), teams.end(), std::default_random_engine());
}

/* Create and destroy several coexisting teams */
UCC_TEST_F(test_team, team_create_multiple_preconnect_ctx_global)
{
    int job_size = 16;
    UccJob job(job_size, UccJob::UCC_JOB_CTX_GLOBAL,
               {ucc_env_var_t("UCC_TL_UCP_PRECONNECT", "inf")});
    int n_teams  = 4; /* how many teams to create */
    std::vector<UccTeam_h> teams;
    for (int i = 0; i < n_teams; i++) {
        int team_size = 2 + (rand() % (job_size - 2 + 1));
        teams.push_back(job.create_team(team_size));
    }
    /* shuffle vector so that teams are destroyed in different order */
    std::shuffle(teams.begin(), teams.end(), std::default_random_engine());
}

UCC_TEST_F(test_team, team_create_no_ep)
{
    UccTeam_h team = UccJob::getStaticJob()->create_team(
        UccJob::staticUccJobSize, false, false);
}
