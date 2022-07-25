/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
extern "C" {
#include <core/ucc_team.h>
#include <utils/ucc_coll_utils.h>
}

#include <common/test.h>
using Param = std::tuple<int, ucc_datatype_t, int>;

class test_coll_args_msgsize : public ucc::test, public::testing::WithParamInterface<Param> {
public:
    ucc_team_t               team;
    ucc_base_coll_args_t     args;
    std::vector<int> counts;
    ucc_count_t              total;
    ucc_datatype_t           dt;
    void _init(int team_size, ucc_datatype_t _dt, int c) {
        team.size = team_size;
        team.rank = 1;
        args.team = &team;
        total     = 0;
        dt        = _dt;
        counts.resize(team.size);
        for (int i = 0; i < team.size; i++) {
            counts[i] = c + i;
            total += counts[i];
        }
        memset(&args.args, 0x1, sizeof(args.args));
    };
    size_t total_size() {
        return ucc_dt_size(dt) * total;
    }
};

UCC_TEST_P(test_coll_args_msgsize, dst_vector)
{
    auto colls = {UCC_COLL_TYPE_ALLGATHERV, UCC_COLL_TYPE_REDUCE_SCATTERV};
    auto p     = GetParam();

    _init(std::get<0>(p), std::get<1>(p), std::get<2>(p));
    args.args.dst.info_v.counts   = (ucc_count_t*)counts.data();
    args.args.dst.info_v.datatype = dt;

    for (auto c : colls) {
        args.args.coll_type = c;
        EXPECT_EQ(total_size(), ucc_coll_args_msgsize(&args));
    }
}

UCC_TEST_P(test_coll_args_msgsize, always_zero)
{
    auto colls = {UCC_COLL_TYPE_BARRIER, UCC_COLL_TYPE_FANIN,
                  UCC_COLL_TYPE_FANOUT};
    auto p     = GetParam();

    _init(std::get<0>(p), std::get<1>(p), std::get<2>(p));
    for (auto c : colls) {
        args.args.coll_type = c;
        EXPECT_EQ(0, ucc_coll_args_msgsize(&args));
    }
}

UCC_TEST_P(test_coll_args_msgsize, scalar)
{
    auto colls = {UCC_COLL_TYPE_ALLREDUCE, UCC_COLL_TYPE_REDUCE_SCATTER,
                  UCC_COLL_TYPE_ALLTOALL, UCC_COLL_TYPE_ALLGATHER};
    auto p     = GetParam();

    _init(std::get<0>(p), std::get<1>(p), std::get<2>(p));
    /* Dont fill src - there is garbage there, also will check for INPLACE */
    args.args.dst.info.count    = total;
    args.args.dst.info.datatype = dt;

    for (auto c : colls) {
        args.args.coll_type = c;
        args.args.coll_type = UCC_COLL_TYPE_ALLREDUCE;
        EXPECT_EQ(total_size(), ucc_coll_args_msgsize(&args));
    }
}

UCC_TEST_P(test_coll_args_msgsize, reduce)
{
    auto p     = GetParam();

    for (int r = 0; r < 2; r++) {
        _init(std::get<0>(p), std::get<1>(p), std::get<2>(p));
        args.args.coll_type = UCC_COLL_TYPE_REDUCE;
        args.args.root      = r;
        if (team.rank != r) {
            args.args.src.info.count    = total;
            args.args.src.info.datatype = dt;
        } else {
            args.args.dst.info.count    = total;
            args.args.dst.info.datatype = dt;
        }
        EXPECT_EQ(total_size(), ucc_coll_args_msgsize(&args));
    }

}

UCC_TEST_P(test_coll_args_msgsize, scatter)
{
    auto p     = GetParam();

    for (int r = 0; r < 2; r++) {
        _init(std::get<0>(p), std::get<1>(p), std::get<2>(p));
        args.args.coll_type = UCC_COLL_TYPE_SCATTER;
        args.args.root      = r;
        if (team.rank != r) {
            args.args.dst.info.count    = total;
            args.args.dst.info.datatype = dt;
        } else {
            args.args.src.info.count    = total * team.size;
            args.args.src.info.datatype = dt;
        }
        EXPECT_EQ(total_size() * team.size, ucc_coll_args_msgsize(&args));
    }
}

UCC_TEST_P(test_coll_args_msgsize, gather)
{
    auto p     = GetParam();

    for (int r = 0; r < 2; r++) {
        _init(std::get<0>(p), std::get<1>(p), std::get<2>(p));
        args.args.coll_type = UCC_COLL_TYPE_GATHER;
        args.args.root      = r;
        if (team.rank != r) {
            args.args.src.info.count    = total;
            args.args.src.info.datatype = dt;
        } else {
            args.args.dst.info.count    = total * team.size;
            args.args.dst.info.datatype = dt;
        }
        EXPECT_EQ(total_size() * team.size, ucc_coll_args_msgsize(&args));
    }

}

INSTANTIATE_TEST_CASE_P(
    , test_coll_args_msgsize,
    ::testing::Combine(
        ::testing::Values(2, 8, 11),
        ::testing::Values(UCC_DT_INT32, UCC_DT_UINT128, UCC_DT_FLOAT64),
        ::testing::Values(32, 4096, 65533)));
