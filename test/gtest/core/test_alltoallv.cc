/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"

// tmp
#ifdef ucc_max
#undef ucc_max
#endif

#include "src/utils/ucc_math.h"

using Param_0 = std::tuple<int, int>;

class test_alltoallv : public UccCollArgs, public ucc::test
{
private:
    void init_proc_buf(int nprocs, int rank, uint8_t *buf, size_t len)
    {
        for (auto i = 0; i < len; i++) {
            buf[i] = (uint8_t)rank;
        }
    }
    int validate_buf(int proc, uint8_t *buf, size_t len)
    {
        int err = 0;
        for (int i = 0; i < len; i ++) {
            if (buf[i] != proc) {
                err++;
            }
        }
        return err;
    }
public:
    UccCollArgsVec data_init(int nprocs, ucc_datatype_t dtype,
                             size_t count) {
        int buf_count;
        UccCollArgsVec args;

        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));
            coll->coll_type = UCC_COLL_TYPE_ALLTOALLV;
            coll->mask = UCC_COLL_ARGS_FIELD_FLAGS;
            coll->flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT | UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
            coll->src.info_v.counts = (ucc_count_t*)malloc(sizeof(ucc_count_t) * nprocs);
            coll->src.info_v.datatype = dtype;
            coll->src.info_v.displacements = (ucc_aint_t*)malloc(sizeof(ucc_aint_t) * nprocs);

            coll->dst.info_v.counts = (ucc_count_t*)malloc(sizeof(ucc_count_t) * nprocs);
            coll->dst.info_v.datatype = dtype;
            coll->dst.info_v.displacements = (ucc_aint_t*)malloc(sizeof(ucc_aint_t) * nprocs);

            buf_count = 0;
            for (int i = 0; i < nprocs; i++) {
                int rank_count = (nprocs + r - i) * count;

                coll->src.info_v.counts[i] = rank_count;
                coll->src.info_v.displacements[i] = buf_count;
                buf_count += rank_count;
            }

            coll->src.info_v.buffer = malloc(buf_count * ucc_dt_size(dtype));
            for (int i = 0; i < nprocs; i++) {
                init_proc_buf(nprocs, i, (uint8_t*)coll->src.info_v.buffer +
                              coll->src.info_v.displacements[i] * ucc_dt_size(dtype),
                              coll->src.info_v.counts[i] * ucc_dt_size(dtype));
            }

            buf_count = 0;
            for (int i = 0; i < nprocs; i++) {
                int rank_count = (nprocs - r + i) * count;
                coll->dst.info_v.counts[i] = rank_count;
                coll->dst.info_v.displacements[i] = buf_count;
                buf_count += rank_count;
            }
            coll->dst.info_v.buffer = malloc(buf_count * ucc_dt_size(dtype));
            args.push_back(coll);
        }
        return args;
    }
    void data_validate(UccCollArgsVec args)
    {
        int proc = 0;
        size_t dst_size = 0;
        for (ucc_coll_args_t* coll : args) {
            dst_size = 0;
            for (int i = 0; i < args.size(); i++) {
                dst_size += coll->dst.info_v.counts[i];
            }
            EXPECT_EQ(0, validate_buf(proc, (uint8_t*)coll->dst.info_v.buffer,
                         dst_size * ucc_dt_size(coll->dst.info_v.datatype)));
            proc++;
        }
    }
    void data_fini(UccCollArgsVec args)
    {
        for (ucc_coll_args_t* coll : args) {
            free(coll->src.info_v.counts);
            free(coll->src.info_v.displacements);
            free(coll->dst.info_v.counts);
            free(coll->dst.info_v.displacements);
            free(coll);
        }
        args.clear();
    }
};

class test_alltoallv_0 : public test_alltoallv,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoallv_0, single)
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
        ,
        test_alltoallv_0,
        ::testing::Combine(
            ::testing::Values(1,3,16), // nprocs
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype

class test_alltoallv_1 : public test_alltoallv,
        public ::testing::WithParamInterface<int> {};

UCC_TEST_P(test_alltoallv_1, multiple)
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
        ,
        test_alltoallv_1,
        ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1)); // dtype
