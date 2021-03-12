/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include "src/utils/ucc_math.h"

using Param_0 = std::tuple<int, int>;

template <class T>
class test_alltoallv : public UccCollArgs, public ucc::test
{
public:
    uint64_t coll_mask;
    uint64_t coll_flags;

    test_alltoallv() : coll_mask(0), coll_flags(0) {}
    UccCollArgsVec data_init(int nprocs, ucc_datatype_t dtype,
                             size_t count) {
        int buf_count;
        UccCollArgsVec args(nprocs);

        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));
            coll->coll_type = UCC_COLL_TYPE_ALLTOALLV;

            coll->mask = coll_mask;
            coll->flags = coll_flags;

            coll->src.info_v.mem_type = UCC_MEMORY_TYPE_HOST;
            coll->src.info_v.counts = (ucc_count_t*)malloc(sizeof(T) * nprocs);
            coll->src.info_v.datatype = dtype;
            coll->src.info_v.displacements = (ucc_aint_t*)malloc(sizeof(T) * nprocs);

            coll->dst.info_v.mem_type = UCC_MEMORY_TYPE_HOST;
            coll->dst.info_v.counts = (ucc_count_t*)malloc(sizeof(T) * nprocs);
            coll->dst.info_v.datatype = dtype;
            coll->dst.info_v.displacements = (ucc_aint_t*)malloc(sizeof(T) * nprocs);

            buf_count = 0;
            for (int i = 0; i < nprocs; i++) {
                int rank_count = (nprocs + r - i) * count;
                ((T*)coll->src.info_v.counts)[i] = rank_count;
                ((T*)coll->src.info_v.displacements)[i] = buf_count;
                buf_count += rank_count;
            }

            coll->src.info_v.buffer = malloc(buf_count * ucc_dt_size(dtype));
            for (int i = 0; i < nprocs; i++) {
                alltoallx_init_buf(r, i, (uint8_t*)coll->src.info_v.buffer +
                               ((T*)coll->src.info_v.displacements)[i] * ucc_dt_size(dtype),
                               ((T*)coll->src.info_v.counts)[i] * ucc_dt_size(dtype));
            }
            buf_count = 0;
            for (int i = 0; i < nprocs; i++) {
                int rank_count = (nprocs - r + i) * count;
                ((T*)coll->dst.info_v.counts)[i] = rank_count;
                ((T*)coll->dst.info_v.displacements)[i] = buf_count;
                buf_count += rank_count;
            }
            coll->dst.info_v.buffer = malloc(buf_count * ucc_dt_size(dtype));
            args[r] = coll;
        }
        return args;
    }
    void data_validate(UccCollArgsVec args)
    {
        for (int r = 0; r < args.size(); r++) {
            ucc_coll_args_t* coll = args[r];
            for (int i = 0; i < args.size(); i++) {
                size_t rank_size = ucc_dt_size(coll->dst.info_v.datatype) *
                        (size_t)((T*)coll->dst.info_v.counts)[i];
                size_t rank_offs = ucc_dt_size(coll->dst.info_v.datatype) *
                        (size_t)((T*)coll->dst.info_v.displacements)[i];
                EXPECT_EQ(0,
                          alltoallx_validate_buf(r, i,
                          (uint8_t*)coll->dst.info_v.buffer + rank_offs,
                          rank_size));
            }
        }
    }
    void data_fini(UccCollArgsVec args)
    {
        for (ucc_coll_args_t* coll : args) {
            free(coll->src.info_v.buffer);
            free(coll->src.info_v.counts);
            free(coll->src.info_v.displacements);
            free(coll->dst.info_v.buffer);
            free(coll->dst.info_v.counts);
            free(coll->dst.info_v.displacements);
            free(coll);
        }
        args.clear();
    }
};

class test_alltoallv_0 : public test_alltoallv <uint64_t>,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoallv_0, single)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    UccTeam_h team = UccJob::getStaticJob()->create_team(size);

    coll_mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                 UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;

    UccCollArgsVec args = data_init(size, (ucc_datatype_t)dtype, 1);

    UccReq    req(team, args);
    req.start();
    req.wait();

    data_validate(args);
    data_fini(args);
}


class test_alltoallv_1 : public test_alltoallv <uint32_t>,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoallv_1, single)
{
    const int size = std::get<0>(GetParam());
    const ucc_datatype_t dtype = (ucc_datatype_t)std::get<1>(GetParam());
    UccTeam_h team = UccJob::getStaticJob()->create_team(size);
    UccCollArgsVec args = data_init(size, (ucc_datatype_t)dtype, 1);

    UccReq    req(team, args);
    req.start();
    req.wait();

    data_validate(args);
    data_fini(args);
}


INSTANTIATE_TEST_CASE_P(
        64,
        test_alltoallv_0,
        ::testing::Combine(
            ::testing::Values(1,3,16), // nprocs
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype

INSTANTIATE_TEST_CASE_P(
        32,
        test_alltoallv_1,
        ::testing::Combine(
            ::testing::Values(1,3,16), // nprocs
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype


class test_alltoallv_2 : public test_alltoallv<uint64_t>,
        public ::testing::WithParamInterface<int> {};

class test_alltoallv_3 : public test_alltoallv<uint32_t>,
        public ::testing::WithParamInterface<int> {};

UCC_TEST_P(test_alltoallv_2, multiple)
{
    const ucc_datatype_t dtype = (ucc_datatype_t)(GetParam());
    std::vector<UccReq> reqs;
    std::vector<UccCollArgsVec> args;

    coll_mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                 UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;

    for (auto &team : UccJob::getStaticTeams()) {
        UccCollArgsVec arg = data_init(team->procs.size(),
                                       dtype, 1);
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


UCC_TEST_P(test_alltoallv_3, multiple)
{
    const ucc_datatype_t dtype = (ucc_datatype_t)(GetParam());
    std::vector<UccReq> reqs;
    std::vector<UccCollArgsVec> args;

    for (auto &team : UccJob::getStaticTeams()) {
        UccCollArgsVec arg = data_init(team->procs.size(),
                                       dtype, 1);
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
        64,
        test_alltoallv_2,
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1)); // dtype

INSTANTIATE_TEST_CASE_P(
        32,
        test_alltoallv_3,
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1)); // dtype
