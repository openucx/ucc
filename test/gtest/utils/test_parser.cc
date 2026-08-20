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
#include <sstream>
#include <vector>

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
    std::string str      =
        "0-4K:host:8,4k-inf:host:10,0-4k:cuda:7,0-4k:cuda_managed:6,auto";
    size_t      msgsize1 = 1024, msgsize2 = 8192;

    EXPECT_EQ(1, ucc_config_sscanf_uint_ranged(str.c_str(), p, NULL));
    EXPECT_EQ(8, ucc_mrange_uint_get(p, msgsize1, UCC_MEMORY_TYPE_HOST));
    EXPECT_EQ(10, ucc_mrange_uint_get(p, msgsize2, UCC_MEMORY_TYPE_HOST));
    EXPECT_EQ(7, ucc_mrange_uint_get(p, msgsize1, UCC_MEMORY_TYPE_CUDA));
    EXPECT_EQ(UCC_UUNITS_AUTO, ucc_mrange_uint_get(p, msgsize2,
                                                   UCC_MEMORY_TYPE_CUDA));
    EXPECT_EQ(6, ucc_mrange_uint_get(p, msgsize1,
                                     UCC_MEMORY_TYPE_CUDA_MANAGED));
    EXPECT_EQ(UCC_UUNITS_AUTO,
              ucc_mrange_uint_get(p, msgsize2, UCC_MEMORY_TYPE_CUDA_MANAGED));
    ucc_mrange_uint_destroy(p);
}

class test_parse_kn_radix : public ucc::test {
  protected:
    static void expect_schedule(
        const ucc_kn_radix_schedule_t *schedule,
        const std::vector<ucc_kn_radix_t> &expected)
    {
        ASSERT_EQ(expected.size(), schedule->n_radices);
        for (size_t i = 0; i < expected.size(); i++) {
            EXPECT_EQ(expected[i], schedule->radices[i]);
        }
    }
};

UCC_TEST_F(test_parse_kn_radix, valid_ranged_schedules)
{
    ucc_mrange_kn_radix_t p;

    ASSERT_TRUE(ucc_config_sscanf_kn_radix(
        "0-4K:host:8x6,4K-inf:cuda:4,auto", &p, NULL));
    expect_schedule(
        ucc_mrange_kn_radix_get(&p, 1024, UCC_MEMORY_TYPE_HOST), {8, 6});
    expect_schedule(
        ucc_mrange_kn_radix_get(&p, 8192, UCC_MEMORY_TYPE_CUDA), {4});
    expect_schedule(
        ucc_mrange_kn_radix_get(&p, 8192, UCC_MEMORY_TYPE_HOST), {});
    ucc_mrange_kn_radix_destroy(&p);
}

UCC_TEST_F(test_parse_kn_radix, schedule_before_memory_type)
{
    ucc_mrange_kn_radix_t p;
    char                  value[128];

    ASSERT_TRUE(ucc_config_sscanf_kn_radix("8x6:cuda", &p, NULL));
    expect_schedule(
        ucc_mrange_kn_radix_get(&p, 1024, UCC_MEMORY_TYPE_CUDA), {8, 6});
    expect_schedule(
        ucc_mrange_kn_radix_get(&p, 1024, UCC_MEMORY_TYPE_HOST), {});
    EXPECT_TRUE(ucc_config_sprintf_kn_radix(value, sizeof(value), &p, NULL));
    EXPECT_STREQ("8x6:Cuda,auto", value);
    ucc_mrange_kn_radix_destroy(&p);
}

UCC_TEST_F(test_parse_kn_radix, invalid_schedules)
{
    const char *invalid[] = {"", "8xx6", "8x", "1x8", "0", "65536"};
    ucc_mrange_kn_radix_t p;

    for (auto value : invalid) {
        EXPECT_FALSE(ucc_config_sscanf_kn_radix(value, &p, NULL));
    }
}

UCC_TEST_F(test_parse_kn_radix, overlong_schedule)
{
    ucc_mrange_kn_radix_t p;
    std::ostringstream    value;

    for (unsigned i = 0; i <= UCC_KN_MAX_RADIX_PHASES; i++) {
        value << (i ? "x2" : "2");
    }
    EXPECT_FALSE(ucc_config_sscanf_kn_radix(value.str().c_str(), &p, NULL));
}

UCC_TEST_F(test_parse_kn_radix, clone)
{
    ucc_mrange_kn_radix_t src, dst;

    ASSERT_TRUE(ucc_config_sscanf_kn_radix("8x6:cuda,4", &src, NULL));
    ASSERT_EQ(UCC_OK, ucc_mrange_kn_radix_copy(&dst, &src));
    expect_schedule(
        ucc_mrange_kn_radix_get(&dst, 1024, UCC_MEMORY_TYPE_CUDA), {8, 6});
    expect_schedule(
        ucc_mrange_kn_radix_get(&dst, 1024, UCC_MEMORY_TYPE_HOST), {4});
    ucc_mrange_kn_radix_destroy(&dst);
    ucc_mrange_kn_radix_destroy(&src);
}
