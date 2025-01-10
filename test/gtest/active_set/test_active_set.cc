/**
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"
#include <unordered_set>

typedef std::tuple<uint64_t, uint64_t, int64_t, uint64_t, size_t,
                   ucc_memory_type_t> op_t;
using Param = std::tuple<std::vector<op_t>, ucc_job_env_t, int>;

#define OP_T(_root, _aset_start, _aset_stride, _aset_size, _msg_size, _mt) ({  \
            op_t _op(_root, _aset_start, _aset_stride, _aset_size,             \
                     _msg_size, UCC_MEMORY_TYPE_ ## _mt);                      \
            _op;                                                               \
        })

class test_active_set : public ucc::test,
                       public ::testing::WithParamInterface<Param>
{
public:
    std::vector<UccCollCtxVec> ctxs;
    void data_init(std::vector<op_t> &ops, UccTeam_h team) {
        ucc_rank_t tsize = team->procs.size();
        ctxs.resize(ops.size());
        for (int i = 0; i < ops.size(); i++) {
            uint64_t aset_root = std::get<0>(ops[i]);
            uint64_t aset_start = std::get<1>(ops[i]);
            int64_t aset_stride = std::get<2>(ops[i]);
            uint64_t aset_size = std::get<3>(ops[i]);
            size_t msglen = std::get<4>(ops[i]);
            ucc_memory_type_t mt = std::get<5>(ops[i]);

            // aset contains ranks of the active_set in terms of the original
            // team
            uint64_t to_add = aset_start;
            std::unordered_set<uint64_t> aset{to_add};

            while(aset.size() != aset_size) {
                to_add = (to_add + aset_stride) % tsize;
                // the following assertion makes sure the active set
                // start/stride/size combo doesnt select the same rank twice
                EXPECT_EQ(aset.find(to_add), aset.end());
                aset.insert(to_add);
            }

            ctxs[i].resize(tsize);
            for (int j = 0; j < tsize; j++) {
                if (aset.find(j) == aset.end()) {
                    ctxs[i][j] = NULL;
                    continue;
                }
                ctxs[i][j] = (gtest_ucc_coll_ctx_t*)
                                calloc(1, sizeof(gtest_ucc_coll_ctx_t));
                ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

                ctxs[i][j]->args = coll;

                coll->mask =
                    UCC_COLL_ARGS_FIELD_ACTIVE_SET | UCC_COLL_ARGS_FIELD_TAG;
                coll->coll_type = UCC_COLL_TYPE_BCAST;
                coll->src.info.mem_type = mt;
                coll->src.info.count   = (ucc_count_t)msglen;
                coll->src.info.datatype = UCC_DT_INT8;
                coll->root = aset_root;
                coll->active_set.size = aset_size;
                coll->active_set.start = aset_start;
                coll->active_set.stride = aset_stride;
                coll->tag = i;

                ctxs[i][j]->rbuf_size = msglen;
                UCC_CHECK(ucc_mc_alloc(&ctxs[i][j]->src_mc_header,
                                        ctxs[i][j]->rbuf_size, mt));
                coll->src.info.buffer = ctxs[i][j]->src_mc_header->addr;

                for (int k = 0; k < ctxs[i][j]->rbuf_size; k++) {
                    ((uint8_t *)coll->src.info.buffer)[k] = (uint8_t) 0;
                }

                if (j == aset_root) {
                    ctxs[i][j]->init_buf = ucc_malloc(ctxs[i][j]->rbuf_size,
                                                      "init buf");
                    EXPECT_NE(ctxs[i][j]->init_buf, nullptr);
                    uint8_t *sbuf = (uint8_t*)ctxs[i][j]->init_buf;
                    for (int k = 0; k < ctxs[i][j]->rbuf_size; k++) {
                        sbuf[k] = (uint8_t) aset_root;
                    }
                    UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer,
                                            ctxs[i][j]->init_buf,
                                            ctxs[i][j]->rbuf_size, mt,
                                            UCC_MEMORY_TYPE_HOST));
                }

            }
        }
    }

    void data_fini(UccTeam_h team)
    {
        for (int i = 0; i < ctxs.size(); i++) {
            for (int j = 0; j < ctxs[i].size(); j++ ) {
                gtest_ucc_coll_ctx_t *ctx = ctxs[i][j];
                if (!ctx) {
                    continue;
                }
                ucc_coll_args_t* coll = ctx->args;
                UCC_CHECK(ucc_mc_free(ctx->src_mc_header));
                if (j == coll->root) {
                    ucc_free(ctx->init_buf);
                }
                free(coll);
                free(ctx);
            }
            ctxs[i].clear();
        }
        ctxs.clear();
    }

    bool data_validate_one(UccCollCtxVec ctx, UccTeam_h team)
    {
        bool     ret  = true;
        int      root = 0;
        ucc_memory_type_t mem_type;
        uint8_t *rst;

        for (int i = 0; i < ctx.size(); i++) {
            if (!ctx[i]) {
                continue;
            }

            root = ctx[i]->args->root;
            mem_type = ctx[i]->args->src.info.mem_type;

            rst = (uint8_t*) ucc_malloc(ctx[i]->rbuf_size, "dsts buf");
            EXPECT_NE(rst, nullptr);

            UCC_CHECK(ucc_mc_memcpy(rst, ctx[i]->args->src.info.buffer,
                                    ctx[i]->rbuf_size,
                                    UCC_MEMORY_TYPE_HOST, mem_type));

            for (int j = 0; j < ctx[root]->rbuf_size; j++) {
                if ((uint8_t) root != rst[j]) {
                    ret = false;
                    break;
                }
            }

            ucc_free(rst);
        }
        
        return ret;
    }
    bool data_validate(UccTeam_h team) {
        for (auto &c : ctxs) {
            if (true != data_validate_one(c, team)) {
                return false;
            }
        }
        return true;
    }
};

UCC_TEST_P(test_active_set, single)
{
    auto                ops     = std::get<0>(GetParam());
    const ucc_job_env_t env     = std::get<1>(GetParam());
    const int           n_procs = std::get<2>(GetParam());

    UccJob    job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h team = job.create_team(n_procs);

    data_init(ops, team);
    for (auto &c : ctxs) {
        UccReq req(team, c);
        req.start();
        req.wait();
    }
    EXPECT_EQ(true, data_validate(team));
    data_fini(team);
}

UCC_TEST_P(test_active_set, multiple)
{
    auto                ops     = std::get<0>(GetParam());
    const ucc_job_env_t env     = std::get<1>(GetParam());
    const int           n_procs = std::get<2>(GetParam());

    UccJob    job(n_procs, UccJob::UCC_JOB_CTX_GLOBAL, env);
    UccTeam_h team = job.create_team(n_procs);

    std::vector<UccReq> reqs;

    data_init(ops, team);
    for (auto &c : ctxs) {
        reqs.push_back(UccReq(team, c));
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);
    EXPECT_EQ(true, data_validate(team));
    data_fini(team);
}

ucc_job_env_t knomial_env = {{"UCC_TL_UCP_TUNE", "bcast:@knomial:0-inf:inf"},
                              {"UCC_CLS", "basic"}};
extern ucc_job_env_t dbt_env; // test_bcast.cc

INSTANTIATE_TEST_CASE_P
(
    , test_active_set,
        ::testing::Combine
        (
            ::testing::Values
            (
                std::vector<op_t>
                ({
                    // root, start, stride, size, msglen, mt
                    OP_T(0, 0, 1, 16, 8, HOST), // subset == full set
                    OP_T(0, 0, 2, 8, 6, HOST), // even ranks in full set
                    OP_T(3, 3, 8, 2, 65530, HOST), // pt2pt
                    OP_T(2, 1, 1, 2, 65531, HOST), // pt2pt
                    OP_T(0, 0, 1, 2, 8, HOST), // pt2pt
                    OP_T(3, 3, 4, 2, 1024, HOST), // pt2pt
                    OP_T(7, 3, 4, 2, 1023, HOST), // pt2pt
                    OP_T(3, 11, -8, 2, 65530, HOST), // pt2pt
                    OP_T(11, 11, -8, 2, 65531, HOST), // pt2pt
                    OP_T(5, 7, -2, 2, 123456, HOST), // pt2pt
                    OP_T(7, 7, -2, 2, 123455, HOST), // pt2pt
                    OP_T(0, 0, 1, 4, 1337, HOST),
                    OP_T(2, 0, 2, 4, 64, HOST),
                    OP_T(6, 0, 3, 3, 1335, HOST),
                    OP_T(4, 7, -1, 6, 18, HOST)

                })
            ),
            ::testing::Values(knomial_env, dbt_env), // env
            ::testing::Values(16) // n_procs
        )
);
