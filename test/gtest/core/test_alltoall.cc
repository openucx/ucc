/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include "src/utils/ucc_math.h"

using Param_0 = std::tuple<int, int, int>;
using Param_1 = std::tuple<int, int>;

class test_alltoall : public UccCollArgs, public ucc::test
{
public:
    UccCollArgsVec data_init(int nprocs, ucc_datatype_t dtype,
                                     size_t count) {
        UccCollArgsVec args(nprocs);
        for (auto i = 0; i < nprocs; i++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));
            coll->mask = 0;
            coll->coll_type = UCC_COLL_TYPE_ALLTOALL;
            coll->src.info.mem_type = UCC_MEMORY_TYPE_HOST;
            coll->src.info.count   = (ucc_count_t)count;
            coll->src.info.datatype = dtype;
            coll->dst.info.mem_type = UCC_MEMORY_TYPE_HOST;
            coll->dst.info.count   = (ucc_count_t)count;
            coll->dst.info.datatype = dtype;

            coll->src.info.buffer = malloc(ucc_dt_size(dtype) * count * nprocs);
            coll->dst.info.buffer = malloc(ucc_dt_size(dtype) * count * nprocs);

            for (int r = 0; r < nprocs; r++) {
                size_t rank_size = ucc_dt_size(dtype) * count;
                alltoallx_init_buf(r, i, (uint8_t*)coll->src.info.buffer +
                              r * rank_size, rank_size);
            }
            args[i] = coll;
        }
        return args;
    }
    void data_fini(UccCollArgsVec args) {
        for (ucc_coll_args_t* coll : args) {
            free(coll->src.info.buffer);
            free(coll->dst.info.buffer);
            free(coll);
        }
        args.clear();
    }
    void data_validate(UccCollArgsVec args)
    {
        for (int r = 0; r < args.size(); r++) {
            ucc_coll_args_t* coll = args[r];
            for (int i = 0; i < args.size(); i++) {
                size_t rank_size = ucc_dt_size(coll->dst.info.datatype) *
                        (size_t)coll->dst.info.count;
                EXPECT_EQ(0,
                          alltoallx_validate_buf(i, r,
                          (uint8_t*)coll->dst.info.buffer + rank_size * i,
                          rank_size));
            }
        }
    }
};

class test_alltoall_0 : public test_alltoall,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoall_0, single)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    const int count = std::get<2>(GetParam());

    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    UccCollArgsVec args = data_init(size, (ucc_datatype_t)dtype, count);
    UccReq    req(team, args);
    req.start();
    req.wait();
    data_validate(args);
    data_fini(args);
}

INSTANTIATE_TEST_CASE_P(
    ,
    test_alltoall_0,
    ::testing::Combine(
        ::testing::Values(1,3,16), // nprocs
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1), // dtype
        ::testing::Values(1,3,8))); // count

class test_alltoall_1 : public test_alltoall,
        public ::testing::WithParamInterface<Param_1> {};

UCC_TEST_P(test_alltoall_1, multiple)
{
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<0>(GetParam());
    const int count = std::get<1>(GetParam());

    std::vector<UccReq> reqs;
    std::vector<UccCollArgsVec> args;
    for (auto &team : UccJob::getStaticTeams()) {
        UccCollArgsVec arg = data_init(team->procs.size(),
                                       dtype, count);
        args.push_back(arg);
        reqs.push_back(UccReq(team, arg));
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);

    for (auto arg : args) {
        data_validate(arg);
        data_fini(arg);
    }
}

INSTANTIATE_TEST_CASE_P(
    ,
    test_alltoall_1,
    ::testing::Combine(
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1), // dtype
        ::testing::Values(1,3,8))); // count
