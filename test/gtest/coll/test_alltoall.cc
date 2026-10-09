/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include "components/cl/basic/cl_basic.h"
#include "components/tl/ucp/tl_ucp.h"
#include "core/ucc_team.h"
#include "utils/ucc_math.h"

#include <cstdlib>
#include <cstring>
#include <string>

namespace {

class ScopedEnvironment {
    struct SavedVariable {
        std::string name;
        bool        was_set;
        std::string value;
    };

    std::vector<SavedVariable> saved;

public:
    explicit ScopedEnvironment(const ucc_job_env_t &env)
    {
        const char *value;

        for (const auto &var : env) {
            value = std::getenv(var.first.c_str());
            saved.push_back({var.first, value != NULL, value ? value : ""});
            setenv(var.first.c_str(), var.second.c_str(), 1);
        }
    }

    ScopedEnvironment(const ScopedEnvironment &) = delete;
    ScopedEnvironment &operator=(const ScopedEnvironment &) = delete;

    ~ScopedEnvironment()
    {
        for (auto var = saved.rbegin(); var != saved.rend(); ++var) {
            if (var->was_set) {
                setenv(var->name.c_str(), var->value.c_str(), 1);
            } else {
                unsetenv(var->name.c_str());
            }
        }
    }
};

static ucc_tl_ucp_team_t *get_ucp_team(UccTeam_h team, int rank)
{
    ucc_team_t *core_team =
        reinterpret_cast<ucc_team_t *>(team->procs[rank].team);

    for (int cl = 0; cl < core_team->n_cl_teams; cl++) {
        ucc_cl_team_t *cl_team = core_team->cl_teams[cl];

        if (std::strcmp(UCC_CL_TEAM_IFACE(cl_team)->super.name, "basic")) {
            continue;
        }
        ucc_cl_basic_team_t *basic_team =
            ucc_derived_of(cl_team, ucc_cl_basic_team_t);
        for (unsigned tl = 0; tl < basic_team->n_tl_teams; tl++) {
            ucc_tl_team_t *tl_team = basic_team->tl_teams[tl];

            if (!std::strcmp(UCC_TL_TEAM_IFACE(tl_team)->super.name, "ucp")) {
                return ucc_derived_of(tl_team, ucc_tl_ucp_team_t);
            }
        }
    }

    ADD_FAILURE() << "Basic/UCP team not found for rank " << rank;
    return NULL;
}

const ucc_job_env_t node_interleaved_schedule_env = {
    {"UCC_CLS", "basic"},
    {"UCC_TLS", "ucp"},
    {"UCC_CL_BASIC_TUNE", "inf"},
    {"UCC_TL_UCP_TUNE", "alltoall:0-inf:@pairwise"},
    {"UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE", "ring_node_interleaved"}};

const ucc_job_env_t ring_schedule_env = {
    {"UCC_CLS", "basic"},
    {"UCC_TLS", "ucp"},
    {"UCC_CL_BASIC_TUNE", "inf"},
    {"UCC_TL_UCP_TUNE", "alltoall:0-inf:@pairwise"},
    {"UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE", "ring"}};

} // namespace

using Param_0 = std::tuple<int, ucc_datatype_t, ucc_memory_type_t, gtest_ucc_inplace_t, int>;
using Param_1 = std::tuple<ucc_datatype_t, ucc_memory_type_t, gtest_ucc_inplace_t, int>;

class test_alltoall : public UccCollArgs, public ucc::test
{
public:
    void data_init(int nprocs, ucc_datatype_t dtype,
                   size_t single_rank_count, UccCollCtxVec &ctxs,
                   UccTeam_h team, bool persistent)
    {
        bool  is_onesided = (NULL != team);
        void *sbuf;
        void *rbuf;
        long *work_buf;

        ctxs.resize(nprocs);
        for (auto i = 0; i < nprocs; i++) {
            ucc_coll_args_t *coll =
                (ucc_coll_args_t *)calloc(1, sizeof(ucc_coll_args_t));

            ctxs[i] =
                (gtest_ucc_coll_ctx_t *)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
            ctxs[i]->args = coll;

            coll->mask              = 0;
            coll->coll_type         = UCC_COLL_TYPE_ALLTOALL;
            coll->src.info.mem_type = mem_type;
            coll->src.info.count    = (ucc_count_t)single_rank_count * nprocs;
            coll->src.info.datatype = dtype;
            coll->dst.info.mem_type = mem_type;
            coll->dst.info.count    = (ucc_count_t)single_rank_count * nprocs;
            coll->dst.info.datatype = dtype;

            ctxs[i]->init_buf = ucc_malloc(
                ucc_dt_size(dtype) * single_rank_count * nprocs, "init buf");
            EXPECT_NE(ctxs[i]->init_buf, nullptr);
            for (int r = 0; r < nprocs; r++) {
                size_t rank_size = ucc_dt_size(dtype) * single_rank_count;
                alltoallx_init_buf(r, i,
                                   (uint8_t *)ctxs[i]->init_buf + r * rank_size,
                                   rank_size);
            }
            if (is_onesided) {
                sbuf        = team->procs[i].p->onesided_buf[0];
                rbuf        = team->procs[i].p->onesided_buf[1];
                work_buf    = (long *)team->procs[i].p->onesided_buf[2];
                coll->mask  = UCC_COLL_ARGS_FIELD_FLAGS |
                             UCC_COLL_ARGS_FIELD_GLOBAL_WORK_BUFFER;
                coll->flags = UCC_COLL_ARGS_FLAG_MEM_MAPPED_BUFFERS;
                coll->src.info.buffer    = sbuf;
                coll->src.info.mem_type  = UCC_MEMORY_TYPE_HOST;
                coll->dst.info.buffer    = rbuf;
                coll->dst.info.mem_type  = UCC_MEMORY_TYPE_HOST;
                coll->global_work_buffer = work_buf;
            } else {
                UCC_CHECK(ucc_mc_alloc(
                    &ctxs[i]->dst_mc_header,
                    ucc_dt_size(dtype) * single_rank_count * nprocs, mem_type));
                coll->dst.info.buffer = ctxs[i]->dst_mc_header->addr;
            }
            if (TEST_INPLACE == inplace) {
                coll->mask |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_IN_PLACE;
                UCC_CHECK(ucc_mc_memcpy(
                    coll->dst.info.buffer, ctxs[i]->init_buf,
                    ucc_dt_size(dtype) * single_rank_count * nprocs, mem_type,
                    UCC_MEMORY_TYPE_HOST));
            } else {
                if (!is_onesided) {
                    UCC_CHECK(ucc_mc_alloc(&ctxs[i]->src_mc_header,
                                           ucc_dt_size(dtype) *
                                               single_rank_count * nprocs,
                                           mem_type));
                    coll->src.info.buffer = ctxs[i]->src_mc_header->addr;
                }
                UCC_CHECK(ucc_mc_memcpy(
                    coll->src.info.buffer, ctxs[i]->init_buf,
                    ucc_dt_size(dtype) * single_rank_count * nprocs, mem_type,
                    UCC_MEMORY_TYPE_HOST));
            }

            if (persistent) {
                coll->mask  |= UCC_COLL_ARGS_FIELD_FLAGS;
                coll->flags |= UCC_COLL_ARGS_FLAG_PERSISTENT;
            }
        }
    }

    void data_init(int nprocs, ucc_datatype_t dtype, size_t single_rank_count,
                   UccCollCtxVec &ctxs, bool persistent = false)
    {
        data_init(nprocs, dtype, single_rank_count, ctxs, NULL, persistent);
    }

    void reset(UccCollCtxVec ctxs)
    {
        for (auto r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t *coll  = ctxs[r]->args;
            size_t single_rank_count = coll->dst.info.count / ctxs.size();
            ucc_datatype_t   dtype = coll->dst.info.datatype;
            clear_buffer(coll->dst.info.buffer,
                         single_rank_count * ucc_dt_size(dtype), mem_type, 0);
            if (TEST_INPLACE == inplace) {
                UCC_CHECK(ucc_mc_memcpy(
                    (void *)(ptrdiff_t)coll->dst.info.buffer, ctxs[r]->init_buf,
                    ucc_dt_size(dtype) * single_rank_count, mem_type,
                    UCC_MEMORY_TYPE_HOST));
            }
        }
    }

    void data_fini(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t* ctx : ctxs) {
            ucc_coll_args_t* coll = ctx->args;
            if (coll->src.info.buffer) { /* no inplace */
                UCC_CHECK(ucc_mc_free(ctx->src_mc_header));
            }
            UCC_CHECK(ucc_mc_free(ctx->dst_mc_header));
            ucc_free(ctx->init_buf);
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }

    void data_fini_onesided(UccCollCtxVec ctxs)
    {
        for (gtest_ucc_coll_ctx_t *ctx : ctxs) {
            ucc_coll_args_t *coll = ctx->args;
            ucc_free(ctx->init_buf);
            free(coll);
            free(ctx);
        }
        ctxs.clear();
    }

    bool data_validate(UccCollCtxVec ctxs)
    {
        bool                   ret = true;
        std::vector<uint8_t *> dsts(ctxs.size());

        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                size_t buf_size =
                    ucc_dt_size(ctxs[r]->args->dst.info.datatype) *
                    (size_t)ctxs[r]->args->dst.info.count;
                dsts[r] = (uint8_t *) ucc_malloc(buf_size, "dsts buf");
                EXPECT_NE(dsts[r], nullptr);
                UCC_CHECK(ucc_mc_memcpy(dsts[r], ctxs[r]->args->dst.info.buffer,
                                        buf_size, UCC_MEMORY_TYPE_HOST, mem_type));
            }
        } else {
            for (int r = 0; r < ctxs.size(); r++) {
                dsts[r] = (uint8_t *)(ctxs[r]->args->dst.info.buffer);
            }
        }
        for (int r = 0; r < ctxs.size(); r++) {
            ucc_coll_args_t* coll = ctxs[r]->args;

            for (int i = 0; i < ctxs.size(); i++) {
                size_t rank_size = ucc_dt_size(coll->dst.info.datatype) *
                                   (size_t)(coll->dst.info.count / ctxs.size());
                if (0 != alltoallx_validate_buf(i, r, (uint8_t*)dsts[r] +
                                                rank_size * i, rank_size)) {
                    ret = false;
                    break;
                }
            }
        }
        if (UCC_MEMORY_TYPE_HOST != mem_type) {
            for (int r = 0; r < ctxs.size(); r++) {
                ucc_free(dsts[r]);
            }
        }
        return ret;
    }

    void check_topology_schedule(UccJob::ucc_job_ctx_mode_t ctx_mode,
                                 int nprocs, bool expected_enabled,
                                 ucc_tl_ucp_alltoall_pairwise_schedule_t
                                     expected_schedule,
                                 const std::vector<ucc_rank_t> &expected_order)
    {
        UccJob        job(nprocs, ctx_mode, ucc_job_env_t(),
                          UCC_THREAD_MULTIPLE);
        UccTeam_h     team = job.create_team(nprocs);
        UccCollCtxVec ctxs;

        for (int rank = 0; rank < nprocs; rank++) {
            ucc_tl_ucp_team_t *tl_team = get_ucp_team(team, rank);

            ASSERT_NE(nullptr, tl_team);
            EXPECT_EQ(expected_enabled,
                      !!tl_team->alltoall_node_interleaved.available);
            EXPECT_EQ(expected_schedule,
                      tl_team->cfg.alltoall_pairwise_schedule);
            if (expected_enabled) {
                ASSERT_EQ((size_t)nprocs, expected_order.size());
                ASSERT_NE(nullptr,
                          tl_team->alltoall_node_interleaved.rank_order);
                ASSERT_NE(nullptr,
                          tl_team->alltoall_node_interleaved.rank_labels);
                for (ucc_rank_t label = 0; label < (ucc_rank_t)nprocs;
                     label++) {
                    EXPECT_EQ(expected_order[label],
                              tl_team->alltoall_node_interleaved
                                  .rank_order[label]);
                    EXPECT_EQ(label,
                              tl_team->alltoall_node_interleaved.rank_labels[
                                  expected_order[label]]);
                }
            } else {
                EXPECT_EQ(nullptr,
                          tl_team->alltoall_node_interleaved.rank_order);
                EXPECT_EQ(nullptr,
                          tl_team->alltoall_node_interleaved.rank_labels);
            }
        }

        set_inplace(TEST_NO_INPLACE);
        set_mem_type(UCC_MEMORY_TYPE_HOST);
        data_init(nprocs, UCC_DT_INT32, 3, ctxs, false);
        UccReq req(team, ctxs);
        EXPECT_EQ(UCC_OK, req.status);
        if (req.status != UCC_OK) {
            data_fini(ctxs);
            return;
        }
        req.start();
        ucc_status_t status = req.wait();
        EXPECT_EQ(UCC_OK, status);
        if (status == UCC_OK) {
            EXPECT_TRUE(data_validate(ctxs));
        }
        data_fini(ctxs);
    }
};

UCC_TEST_F(test_alltoall, topology_schedule_uniform_topology)
{
    ScopedEnvironment env(node_interleaved_schedule_env);

    check_topology_schedule(
        UccJob::UCC_JOB_CTX_GLOBAL, 8, true,
        UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE_RING_NODE_INTERLEAVED,
        {0, 4, 1, 5, 2, 6, 3, 7});
}

UCC_TEST_F(test_alltoall, topology_schedule_uneven_ppn_fallback)
{
    ScopedEnvironment env(node_interleaved_schedule_env);

    check_topology_schedule(
        UccJob::UCC_JOB_CTX_GLOBAL, 11, false,
        UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE_RING_NODE_INTERLEAVED, {});
}

UCC_TEST_F(test_alltoall, topology_schedule_no_context_topology_fallback)
{
    ScopedEnvironment env(node_interleaved_schedule_env);

    check_topology_schedule(
        UccJob::UCC_JOB_CTX_LOCAL, 8, false,
        UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE_RING_NODE_INTERLEAVED, {});
}

UCC_TEST_F(test_alltoall, topology_schedule_forced_ring_skips_topology)
{
    ScopedEnvironment env(ring_schedule_env);

    check_topology_schedule(UccJob::UCC_JOB_CTX_GLOBAL, 8, false,
                            UCC_TL_UCP_ALLTOALL_PAIRWISE_SCHEDULE_RING, {});
}

class test_alltoall_0 : public test_alltoall,
        public ::testing::WithParamInterface<Param_0> {};

UCC_TEST_P(test_alltoall_0, single)
{
    const int            team_id  = std::get<0>(GetParam());
    const ucc_datatype_t dtype    = std::get<1>(GetParam());
    ucc_memory_type_t    mem_type = std::get<2>(GetParam());
    gtest_ucc_inplace_t  inplace  = std::get<3>(GetParam());
    const int            count    = std::get<4>(GetParam());
    UccTeam_h            team     = UccJob::getStaticTeams()[team_id];
    int                  size     = team->procs.size();
    UccCollCtxVec        ctxs;

    this->set_inplace(inplace);
    SET_MEM_TYPE(mem_type);

    data_init(size, dtype, count, ctxs, false);
    UccReq    req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini(ctxs);
}

UCC_TEST_P(test_alltoall_0, single_onesided)
{
    const int            team_id        = std::get<0>(GetParam());
    const ucc_datatype_t dtype          = std::get<1>(GetParam());
    ucc_memory_type_t    mem_type       = std::get<2>(GetParam());
    gtest_ucc_inplace_t  inplace        = std::get<3>(GetParam());
    const int            count          = std::get<4>(GetParam());
    UccTeam_h            reference_team = UccJob::getStaticTeams()[team_id];
    int                  size           = reference_team->procs.size();
    ucc_job_env_t        env       = {{"UCC_TL_UCP_TUNE", "alltoall:0-inf:@1"}};
    bool                 is_contig = true;
    UccJob               job(size, UccJob::UCC_JOB_CTX_GLOBAL_ONESIDED, env);
    UccTeam_h            team;
    std::vector<int>     reference_ranks;
    UccCollCtxVec        ctxs;

    for (auto i = 0; i < reference_team->n_procs; i++) {
        int rank = reference_team->procs[i].p->job_rank;
        reference_ranks.push_back(rank);
        if (is_contig && i > 0 &&
            (rank - reference_ranks[i - 1] > 1 ||
             reference_ranks[i - 1] - rank > 1)) {
            is_contig = false;
        }
    }
    team = job.create_team(reference_ranks, true, is_contig, true);
    this->set_inplace(inplace);
    SET_MEM_TYPE(mem_type);
    data_init(size, dtype, count, ctxs, team, false);
    UccReq req(team, ctxs);
    req.start();
    req.wait();
    EXPECT_EQ(true, data_validate(ctxs));
    data_fini_onesided(ctxs);
}

UCC_TEST_P(test_alltoall_0, single_persistent)
{
    const int            team_id  = std::get<0>(GetParam());
    const ucc_datatype_t dtype    = std::get<1>(GetParam());
    ucc_memory_type_t    mem_type = std::get<2>(GetParam());
    gtest_ucc_inplace_t  inplace  = std::get<3>(GetParam());
    const int            count    = std::get<4>(GetParam());
    UccTeam_h            team     = UccJob::getStaticTeams()[team_id];
    int                  size     = team->procs.size();
    const int            n_calls  = 3;
    UccCollCtxVec        ctxs;

    this->set_inplace(inplace);
    SET_MEM_TYPE(mem_type);

    data_init(size, dtype, count, ctxs, true);
    UccReq req(team, ctxs);

    for (auto i = 0; i < n_calls; i++) {
        req.start();
        req.wait();
        EXPECT_EQ(true, data_validate(ctxs));
        reset(ctxs);
    }
    data_fini(ctxs);
}

INSTANTIATE_TEST_CASE_P(
    , test_alltoall_0,
    ::testing::Combine(
        ::testing::Range(1, UccJob::nStaticTeams), // team_ids
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA,
                          UCC_MEMORY_TYPE_CUDA_MANAGED),
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(/*TEST_INPLACE,*/ TEST_NO_INPLACE),
        ::testing::Values(1,3)));

class test_alltoall_1 : public test_alltoall,
        public ::testing::WithParamInterface<Param_1> {};

UCC_TEST_P(test_alltoall_1, multiple)
{
    const ucc_datatype_t       dtype    = std::get<0>(GetParam());
    ucc_memory_type_t          mem_type = std::get<1>(GetParam());
    gtest_ucc_inplace_t        inplace  = std::get<2>(GetParam());
    const int                  count    = std::get<3>(GetParam());
    std::vector<UccReq>        reqs;
    std::vector<UccCollCtxVec> ctxs;

    for (int tid = 0; tid < UccJob::nStaticTeams; tid++) {
        UccTeam_h       team = UccJob::getStaticTeams()[tid];
        int             size = team->procs.size();
        UccCollCtxVec   ctx;

        this->set_inplace(inplace);
        SET_MEM_TYPE(mem_type);

        data_init(size, dtype, count, ctx, false);
        reqs.push_back(UccReq(team, ctx));
        ctxs.push_back(ctx);
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);

    for (auto ctx : ctxs) {
        EXPECT_EQ(true, data_validate(ctx));
        data_fini(ctx);
    }
}

INSTANTIATE_TEST_CASE_P(
    , test_alltoall_1,
    ::testing::Combine(
        PREDEFINED_DTYPES,
#ifdef HAVE_CUDA
        ::testing::Values(UCC_MEMORY_TYPE_HOST, UCC_MEMORY_TYPE_CUDA,
                          UCC_MEMORY_TYPE_CUDA_MANAGED),
#else
        ::testing::Values(UCC_MEMORY_TYPE_HOST),
#endif
        ::testing::Values(/*TEST_INPLACE,*/ TEST_NO_INPLACE),
        ::testing::Values(1,3,8192))); // count
