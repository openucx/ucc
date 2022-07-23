/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
#include <common/test.h>
#include <common/test_ucc.h>
#include <vector>

extern "C" {
#include "core/ucc_team.h"
#include "core/ucc_service_coll.h"
}

class test_service_coll {
  public:
    int *                                 array;
    UccTeam_h                             team;
    std::vector<ucc_subset_t>             subsets;
    std::vector<ucc_service_coll_req_t *> reqs;
    test_service_coll(std::vector<int> _subset, UccTeam_h _team)
    {
        team  = _team;
        array = new int[_subset.size()];
        memcpy(array, _subset.data(), sizeof(int) * _subset.size());
        subsets.resize(_subset.size());
        reqs.resize(_subset.size());
        for (auto i = 0; i < _subset.size(); i++) {
            subsets[i].myrank              = i;
            subsets[i].map.type            = UCC_EP_MAP_ARRAY;
            subsets[i].map.array.map       = (void *)array;
            subsets[i].map.array.elem_size = sizeof(int);
            subsets[i].map.ep_num          = _subset.size();
        }
    }
    ~test_service_coll()
    {
        delete[] array;
    }
    void progress()
    {
        for (auto i = 0; i < reqs.size(); i++) {
            ucc_context_progress(team.get()->procs[array[i]].p->ctx_h);
        }
    }
    void wait()
    {
        ucc_status_t status;
        bool         ready;
        do {
            ready = true;
            progress();
            for (auto &r : reqs) {
                status = ucc_service_coll_test(r);
                EXPECT_GE(status, 0);
                if (UCC_INPROGRESS == status) {
                    ready = false;
                }
            }
        } while (!ready);
        for (auto &r : reqs) {
            ucc_service_coll_finalize(r);
        }
    }
};

class test_service_allreduce : public test_service_coll {
    std::vector<std::vector<int>> sbuf;
    std::vector<std::vector<int>> rbuf;

  public:
    test_service_allreduce(std::vector<int> _subset, size_t count,
                           UccTeam_h _team)
        : test_service_coll(_subset, _team)
    {
        sbuf.resize(_subset.size());
        rbuf.resize(_subset.size());
        for (auto i = 0; i < _subset.size(); i++) {
            sbuf[i].resize(count);
            rbuf[i].resize(count);
            for (auto j = 0; j < count; j++) {
                sbuf[i][j] = i + j + 1;
                rbuf[i][j] = 0;
            }
        }
    }
    void start()
    {
        ucc_status_t status;
        for (auto i = 0; i < reqs.size(); i++) {
            auto r = array[i];
            status = ucc_service_allreduce(
                team.get()->procs[r].team, sbuf[i].data(), rbuf[i].data(),
                UCC_DT_INT32, sbuf[i].size(), UCC_OP_SUM, subsets[i], &reqs[i]);
            EXPECT_EQ(UCC_OK, status);
        }
    }
    void check()
    {
        int size = reqs.size();
        for (auto i = 0; i < size; i++) {
            for (auto j = 0; j < sbuf[i].size(); j++) {
                int check = size * (size + 1) / 2 + j * size;
                EXPECT_EQ(check, rbuf[i][j]);
            }
        }
    }
};

class test_scoll_allreduce
    : public ucc::test,
      public ::testing::WithParamInterface<std::vector<int>> {
};

UCC_TEST_P(test_scoll_allreduce, allreduce)
{
    /* Reversed team of size staticUccJobSize - last one in static teawms array */
    auto team = UccJob::getStaticTeams().back();
    ASSERT_EQ(team.get()->procs.size(), 16);
    std::vector<int>       subset = GetParam();
    test_service_allreduce t(subset, 4, team);
    t.start();
    t.wait();
    t.check();
}

INSTANTIATE_TEST_CASE_P(
    , test_scoll_allreduce,
    ::testing::Values(std::vector<int>({1, 0}), std::vector<int>({2, 3}),
                      std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                        12, 13, 14, 15}),
                      std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8}),
                      std::vector<int>({15, 14, 13, 12, 11, 10, 9}),
                      std::vector<int>({0, 2, 4, 6, 8})));

class test_service_allgather : public test_service_coll {
    std::vector<std::vector<int>> sbuf;
    std::vector<std::vector<int>> rbuf;

  public:
    test_service_allgather(std::vector<int> _subset, size_t count,
                           UccTeam_h _team)
        : test_service_coll(_subset, _team)
    {
        sbuf.resize(_subset.size());
        rbuf.resize(_subset.size());
        for (auto i = 0; i < _subset.size(); i++) {
            sbuf[i].resize(count);
            rbuf[i].resize(count * _subset.size());
            for (auto j = 0; j < count; j++) {
                sbuf[i][j] = i + j + 1;
            }
            for (auto j = 0; j < count * _subset.size(); j++) {
                rbuf[i][j] = 0;
            }
        }
    }
    void start()
    {
        ucc_status_t status;
        for (auto i = 0; i < reqs.size(); i++) {
            auto r = array[i];
            status = ucc_service_allgather(
                team.get()->procs[r].team, sbuf[i].data(), rbuf[i].data(),
                sbuf[i].size() * sizeof(int), subsets[i], &reqs[i]);
            EXPECT_EQ(UCC_OK, status);
        }
    }
    void check()
    {
        int size  = reqs.size();
        int count = sbuf[0].size();
        for (auto i = 0; i < size; i++) {
            for (auto j = 0; j < rbuf[i].size(); j++) {
                int check = (j % count) + 1 + (j / count);
                EXPECT_EQ(check, rbuf[i][j]);
            }
        }
    }
};

class test_scoll_allgather
    : public ucc::test,
      public ::testing::WithParamInterface<std::vector<int>> {
};

UCC_TEST_P(test_scoll_allgather, allgather)
{
    /* Reversed team of size staticUccJobSize - last one in static teawms array */
    auto team = UccJob::getStaticTeams().back();
    ASSERT_EQ(team.get()->procs.size(), 16);
    std::vector<int>       subset = GetParam();
    test_service_allgather t(subset, 4, team);
    t.start();
    t.wait();
    t.check();
}

INSTANTIATE_TEST_CASE_P(
    , test_scoll_allgather,
    ::testing::Values(std::vector<int>({1, 0}), std::vector<int>({2, 3}),
                      std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                        12, 13, 14, 15}),
                      std::vector<int>({0, 1, 2, 3, 4, 5, 6, 7, 8}),
                      std::vector<int>({15, 14, 13, 12, 11, 10, 9}),
                      std::vector<int>({0, 2, 4, 6, 8})));
