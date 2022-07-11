/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#include "common/test_ucc.h"

typedef std::tuple<std::pair<int, int>, size_t, ucc_memory_type_t> op_t;
using param = std::vector<op_t>;

#define OP_T(_src, _dst, _size, _mt) ({                                 \
            op_t _op(std::pair<int, int>(_src, _dst), _size, UCC_MEMORY_TYPE_ ## _mt); \
            _op;                                                        \
        })

class test_active_set : public ucc::test
{};

class test_active_set_2 : public test_active_set,
                       public ::testing::WithParamInterface<param>
{
public:
    std::vector<UccCollCtxVec> ctxs;
    void data_init(std::vector<op_t> &ops, UccTeam_h team) {
        ucc_rank_t tsize = team->procs.size();
        ctxs.resize(ops.size());
        for (int i = 0; i < ops.size(); i++) {
            ucc_rank_t src = std::get<0>(ops[i]).first;
            ucc_rank_t dst = std::get<0>(ops[i]).second;
            size_t     msglen = std::get<1>(ops[i]);
            ucc_memory_type_t mt = std::get<2>(ops[i]);
            ctxs[i].resize(tsize);
            for (int j = 0; j < tsize; j++) {
                if (j != src && j != dst) {
                    ctxs[i][j] = NULL;
                    continue;
                }
                ctxs[i][j] = (gtest_ucc_coll_ctx_t*)calloc(1, sizeof(gtest_ucc_coll_ctx_t));
                ucc_coll_args_t *coll = (ucc_coll_args_t*)
                    calloc(1, sizeof(ucc_coll_args_t));

                ctxs[i][j]->args = coll;

                coll->mask = UCC_COLL_ARGS_FIELD_ACTIVE_SET;
                coll->coll_type = UCC_COLL_TYPE_BCAST;
                coll->src.info.mem_type = mt;
                coll->src.info.count   = (ucc_count_t)msglen;
                coll->src.info.datatype = UCC_DT_INT8;
                coll->root = src;
                coll->active_set.size = 2;
                coll->active_set.start = src;
                coll->active_set.stride = (int)dst - (int)src;

                ctxs[i][j]->rbuf_size = msglen;
                UCC_CHECK(ucc_mc_alloc(&ctxs[i][j]->src_mc_header, ctxs[i][j]->rbuf_size,
                                       mt));
                coll->src.info.buffer = ctxs[i][j]->src_mc_header->addr;
                if (j == src) {
                    ctxs[i][j]->init_buf = ucc_malloc(ctxs[i][j]->rbuf_size, "init buf");
                    EXPECT_NE(ctxs[i][j]->init_buf, nullptr);
                    uint8_t *sbuf = (uint8_t*)ctxs[i][j]->init_buf;
                    for (int k = 0; k < ctxs[i][j]->rbuf_size; k++) {
                        sbuf[k] = (uint8_t)(src + k * dst);
                    }
                    UCC_CHECK(ucc_mc_memcpy(coll->src.info.buffer, ctxs[i][j]->init_buf,
                                            ctxs[i][j]->rbuf_size, mt,
                                            UCC_MEMORY_TYPE_HOST));
                }

            }
        }
    }

    ~test_active_set_2()
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

    bool data_validate_one(UccCollCtxVec ctxs)
    {
        bool     ret  = true;
        int      root = 0;
        ucc_rank_t src = 0, dst = 0;
        ucc_memory_type_t mem_type;
        uint8_t *rst;

        for (int i = 0; i < ctxs.size(); i++) {
            if (!ctxs[i]) {
                continue;
            }
            root = ctxs[i]->args->root;
            mem_type = ctxs[i]->args->src.info.mem_type;
            if (root == i) {
                src = i;
            } else {
                dst = i;
            }
        }

        rst = (uint8_t*) ucc_malloc(ctxs[root]->rbuf_size, "dsts buf");
        EXPECT_NE(rst, nullptr);
        UCC_CHECK(ucc_mc_memcpy(rst, ctxs[dst]->args->src.info.buffer,
                                ctxs[root]->rbuf_size,
                                UCC_MEMORY_TYPE_HOST, mem_type));

        for (int i = 0; i < ctxs[root]->rbuf_size; i++) {
            if ((uint8_t)(src + i * dst) != rst[i]) {
                ret = false;
                break;
            }
        }

        ucc_free(rst);
        return ret;
    }
    bool data_validate() {
        for (auto &c : ctxs) {
            if (true != data_validate_one(c)) {
                return false;
            }
        }
        return true;
    }
};

UCC_TEST_P(test_active_set_2, single)
{
    auto ops = GetParam();

    UccTeam_h team = UccJob::getStaticTeams().back();

    data_init(ops, team);
    for (auto &c : ctxs) {
        UccReq req(team, c);
        req.start();
        req.wait();
    }
    EXPECT_EQ(true, data_validate());
}

UCC_TEST_P(test_active_set_2, multiple)
{
    auto ops = GetParam();
    UccTeam_h team = UccJob::getStaticTeams().back();
    std::vector<UccReq> reqs;

    data_init(ops, team);
    for (auto &c : ctxs) {
        reqs.push_back(UccReq(team, c));
    }
    UccReq::startall(reqs);
    UccReq::waitall(reqs);
    EXPECT_EQ(true, data_validate());
}

INSTANTIATE_TEST_CASE_P(
    , test_active_set_2,
        ::testing::Values(
            std::vector<op_t>({
                    OP_T(0, 1, 8, HOST),
                    OP_T(3, 7, 1024, HOST),
                    OP_T(11, 3, 65530, HOST),
                    OP_T(7, 5, 123456, HOST)
                })));
