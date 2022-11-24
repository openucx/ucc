/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

extern "C" {
#include "utils/ucc_parser.h"
#include "utils/ucc_datastruct.h"
}
#include <common/test.h>
#include <common/test_ucc.h>

class test_parse_mrange : public ucc::test {
public:
    ucc_mrange_uint_t *p;
    test_parse_mrange() {
        p = (ucc_mrange_uint_t *) ucc_malloc(sizeof(ucc_mrange_uint_t));
    }
    ~test_parse_mrange() {
        ucc_free(p);
    }
};

UCC_TEST_F(test_parse_mrange, check_valid) {
    std::string str = "0-4K:host:8,auto";
    size_t      msgsize1 = 1024, msgsize2 = 8192;

    EXPECT_EQ(1, ucc_config_sscanf_uint_ranged(str.c_str(), p, NULL));
    EXPECT_EQ(8, ucc_mrange_uint_get(p, msgsize1, UCC_MEMORY_TYPE_HOST));
    EXPECT_EQ(UCC_UUNITS_AUTO, ucc_mrange_uint_get(p, msgsize2,
                                                   UCC_MEMORY_TYPE_HOST));
    ucc_mrange_uint_destroy(p);
}

UCC_TEST_F(test_parse_mrange, check_invalid) {
    std::string       str = "0-4K:host:8:8";

    EXPECT_EQ(0, ucc_config_sscanf_uint_ranged(str.c_str(), p, NULL));
    ucc_mrange_uint_destroy(p);

    str = "0-4K:host:a";
    EXPECT_EQ(0, ucc_config_sscanf_uint_ranged(str.c_str(), p, NULL));
    ucc_mrange_uint_destroy(p);

    str = "0-4K:gpu:8";
    EXPECT_EQ(0, ucc_config_sscanf_uint_ranged(str.c_str(), p, NULL));
    ucc_mrange_uint_destroy(p);

    str = "0-f:host:8";
    EXPECT_EQ(0, ucc_config_sscanf_uint_ranged(str.c_str(), p, NULL));
    ucc_mrange_uint_destroy(p);
}

UCC_TEST_F(test_parse_mrange, check_range_multiple) {
    std::string str = "0-4K:host:8,4k-inf:host:10,0-4k:cuda:7,auto";
    size_t      msgsize1 = 1024, msgsize2 = 8192;

    EXPECT_EQ(1, ucc_config_sscanf_uint_ranged(str.c_str(), p, NULL));
    EXPECT_EQ(8, ucc_mrange_uint_get(p, msgsize1, UCC_MEMORY_TYPE_HOST));
    EXPECT_EQ(10, ucc_mrange_uint_get(p, msgsize2, UCC_MEMORY_TYPE_HOST));
    EXPECT_EQ(7, ucc_mrange_uint_get(p, msgsize1, UCC_MEMORY_TYPE_CUDA));
    EXPECT_EQ(UCC_UUNITS_AUTO, ucc_mrange_uint_get(p, msgsize2,
                                                   UCC_MEMORY_TYPE_CUDA));
    ucc_mrange_uint_destroy(p);
}

