/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

extern "C" {
#include <core/ucc_mc.h>
}
#include "common/test_ucc.h"
#include "utils/ucc_math.h"

using Param_0 = std::tuple<int, ucc_memory_type_t, gtest_ucc_inplace_t, int>;
using Param_1 = std::tuple<ucc_memory_type_t, gtest_ucc_inplace_t, int>;

template <class T>
class test_alltoallv : public UccCollArgs, public ucc::test
{
public:
    uint64_t coll_mask;
    uint64_t coll_flags;

    test_alltoallv() : coll_mask(0), coll_flags(0) {}
    void data_init(int nprocs, ucc_datatype_t dtype, size_t count,
                   UccCollCtxVec &ctxs) {
        int buf_count;
        ctxs.resize(nprocs);

        for (auto r = 0; r < nprocs; r++) {
            ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

            ctxs[r] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[r]->args = coll;

            coll->coll_type = UCC_COLL_TYPE_ALLTOALLV;
            coll->mask = coll_mask;
            coll->flags = coll_flags;

            coll->src.info_v.mem_type = mem_type;
            coll->src.info_v.counts = (ucc_count_t*)malloc(sizeof(T) * nprocs);
            coll->src.info_v.datatype = dtype;
            coll->src.info_v.displacements = (ucc_aint_t*)malloc(sizeof(T) * nprocs);

            coll->dst.info_v.mem_type = mem_type;
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

            UCC_CHECK(ucc_mc_alloc(&ctxs[r]->init_buf,
                                   buf_count * ucc_dt_size(dtype),
                                   UCC_MEMORY_TYPE_HOST));
            for (int i = 0; i < nprocs; i++) {
                alltoallx_init_buf(r, i, (uint8_t*)ctxs[r]->init_buf +
                               ((T*)coll->src.info_v.displacements)[i] * ucc_dt_size(dtype),
                               ((T*)coll->src.info_v.counts)[i] * ucc_dt_size(dtype));
            }
            UCC_CHECK(ucc_mc_alloc(&coll->src.info_v.buffer,
                                   buf_count * ucc_dt_size(dtype),
                                   mem_type));
            UCC_CHECK(ucc_mc_memcpy(coll->src.info_v.buffer, ctxs[r]->init_buf,
                                    buf_count * ucc_dt_size(dtype), mem_type,
                                    UCC_MEMORY_TYPE_HOST));

            /* TODO: inplace support */

            buf_count = 0;
            for (int i = 0; i < nprocs; i++) {
                int rank_count = (nprocs - r + i) * count;
                ((T*)coll->dst.info_v.counts)[i] = rank_count;
                ((T*)coll->dst.info_v.displacements)[i] = buf_count;
                buf_count += rank_count;
            }
            ctxs[r]->rbuf_size = buf_count * ucc_dt_size(dtype);
            UCC_CHECK(ucc_mc_alloc(&coll->dst.info_v.buffer,
                                   buf_count * ucc_dt_size(dtype),
                                   mem_type));
        }
    }
    void data_validate(UccCollCtxVec ctxs)
    {
        std::vector<uint8_t *> dsts(ctxs.size());

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                UCC_CHECK(ucc_mc_alloc((void**)&dsts[r], ctxs[r]->rbuf_size,
                                       UCC_MEMORY_TYPE_HOST));
                UCC_CHECK(ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info_v.buffer,
                                        ctxs[r]->rbuf_size, UCC_MEMORY_TYPE_HOST,
                                        mem_type));
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *)(ctxs[r]->args->dst.info.buffer);
            }
        }
        for (int r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t* coll = ctxs[r]->args;
            for (int i = 0; i < ctxs.size(); i++) {
                size_t rank_size = ucc_dt_size(coll->dst.info_v.datatype) *
                        (size_t)((T*)coll->dst.info_v.counts)[i];
                size_t rank_offs = ucc_dt_size(coll->dst.info_v.datatype) *
                        (size_t)((T*)coll->dst.info_v.displacements)[i];
                EXPECT_EQ(0,
                          alltoallx_validate_buf(r, i,
                          (uint8_t*)dsts[r] + rank_offs,
                          rank_size));
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                ucc_mc_free((void*)dsts[r], UCC_MEMORY_TYPE_HOST);
            }
        }
    }
    void data_fini(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            UCC_CHECK(ucc_mc_free(coll->src.info_v.buffer, mem_type));
            free(coll->src.info_v.counts);
            free(coll->src.info_v.displacements);
            UCC_CHECK(ucc_mc_free(coll->dst.info_v.buffer, mem_type));
            free(coll->dst.info_v.counts);
            free(coll->dst.info_v.displacements);
            UCC_CHECK(ucc_mc_free(ctx->init_buf, UCC_MEMORY_TYPE_HOST));
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }
};

class test_alltoallv_0 : public test_alltoallv <uint64_t>,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoallv_0, single)
{
    const int            team_id = std::get<0>(GetParam());
    ucc_memory_type_t    mem_type = std::get<1>(GetParam());
    gtest_ucc_inplace_t  inplace = std::get<2>(GetParam());
    const ucc_datatype_t dtype   = (ucc_datatype_t)std::get<3>(GetParam());
    UccTeam_h            team    = UccJob::getStaticTeams()[team_id];
    int                  size    = team->procs.size();
    UccCollCtxVec        ctxs;

    coll_mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                 UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;
    set_inplace(inplace);
    set_mem_type(mem_type);

    data_init(size, (ucc_datatype_t)dtype, 1, ctxs);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();

    data_validate(ctxs);
    data_fini(ctxs);
}


class test_alltoallv_1 : public test_alltoallv <uint32_t>,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoallv_1, single)
{
    const int            team_id = std::get<0>(GetParam());
    ucc_memory_type_t    mem_type = std::get<1>(GetParam());
    gtest_ucc_inplace_t  inplace = std::get<2>(GetParam());
    const ucc_datatype_t dtype   = (ucc_datatype_t)std::get<3>(GetParam());
    UccTeam_h            team    = UccJob::getStaticTeams()[team_id];
    int                  size    = team->procs.size();
    UccCollCtxVec        ctxs;

    set_inplace(inplace);
    set_mem_type(mem_type);

    data_init(size, (ucc_datatype_t)dtype, 1, ctxs);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();

    data_validate(ctxs);
    data_fini(ctxs);
}

INSTANTIATE_TEST_CASE_P(
        64, test_alltoallv_0,
        ::testing::Combine(
            ::testing::Range(1, UccJob::nStaticTeams), // team_ids
#ifdef HAVE_CUDA
            ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
            ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
            ::testing::Values(/*TEST_INPLACE,*/ TEST_NO_INPLACE), // inplace
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype


INSTANTIATE_TEST_CASE_P(
        32, test_alltoallv_1,
        ::testing::Combine(
            ::testing::Range(1, UccJob::nStaticTeams), // team_ids
#ifdef HAVE_CUDA
            ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
            ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
            ::testing::Values(/*TEST_INPLACE,*/ TEST_NO_INPLACE), // inplace
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype

class test_alltoallv_2 : public test_alltoallv<uint64_t>,
        public ::testing::WithParamInterface<Param_1> {};

class test_alltoallv_3 : public test_alltoallv<uint32_t>,
        public ::testing::WithParamInterface<Param_1> {};


UCC_TEST_P(test_alltoallv_2, multiple)
{    
    ucc_memory_type_t           mem_type = std::get<0>(GetParam());
    gtest_ucc_inplace_t         inplace  = std::get<1>(GetParam());
    const ucc_datatype_t        dtype    = (ucc_datatype_t)std::get<2>(GetParam());
    std::vector<UccReq>         reqs;
    std::vector<UccCollCtxVec>  ctxs;

    coll_mask = UCC_COLL_ARGS_FIELD_FLAGS;
    coll_flags = UCC_COLL_ARGS_FLAG_COUNT_64BIT |
                 UCC_COLL_ARGS_FLAG_DISPLACEMENTS_64BIT;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h       team = UccJob::getStaticTeams()[tid];
        int             size = team->procs.size();
        UccCollCtxVec   ctx;

        this->set_inplace(inplace);
        this->set_mem_type(mem_type);

        data_init(size, (ucc_datatype_t)dtype, 1, ctx);
        reqs.push_back(UccReq(team, ctx));
        ctxs.push_back(ctx);
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);

    for (auto ctx : ctxs) {
        data_validate(ctx);
        data_fini(ctx);
    }
}

UCC_TEST_P(test_alltoallv_3, multiple)
{
    ucc_memory_type_t           mem_type = std::get<0>(GetParam());
    gtest_ucc_inplace_t         inplace  = std::get<1>(GetParam());
    const ucc_datatype_t        dtype    = (ucc_datatype_t)std::get<2>(GetParam());
    std::vector<UccReq>         reqs;
    std::vector<UccCollCtxVec>  ctxs;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h       team = UccJob::getStaticTeams()[tid];
        int             size = team->procs.size();
        UccCollCtxVec   ctx;

        this->set_inplace(inplace);
        this->set_mem_type(mem_type);

        data_init(size, (ucc_datatype_t)dtype, 1, ctx);
        reqs.push_back(UccReq(team, ctx));
        ctxs.push_back(ctx);
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);

    for (auto ctx : ctxs) {
        data_validate(ctx);
        data_fini(ctx);
    }
}

INSTANTIATE_TEST_CASE_P(
        64, test_alltoallv_2,
        ::testing::Combine(
#ifdef HAVE_CUDA
            ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
            ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
            ::testing::Values(/*TEST_INPLACE,*/ TEST_NO_INPLACE), // inplace
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype

INSTANTIATE_TEST_CASE_P(
        32, test_alltoallv_3,
        ::testing::Combine(
#ifdef HAVE_CUDA
            ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA), // mem type
#else
            ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
            ::testing::Values(/*TEST_INPLACE,*/ TEST_NO_INPLACE), // inplace
            ::testing::Range((int)UCC_DT_INT8, (int)UCC_DT_FLOAT64 + 1))); // dtype
