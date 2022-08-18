/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */
extern "C" {
#include "utils/ucc_string.h"
#include "utils/ucc_malloc.h"
}
#include <common/test.h>
#include <string>
#include <vector>

using Param = std::vector<std::string>;
class test_string : public ucc::test,
                    public ::testing::WithParamInterface<Param> {
  public:
    std::string str;
    void        make_str(Param &p);
};

void test_string::make_str(Param &p)
{
    unsigned i;
    str = "";
    for (i = 1; i < p.size() - 1; i++) {
        str += p[i] + p[0];
    }
    str += p[i];
}

UCC_TEST_P(test_string, split)
{
    auto        p     = GetParam();
    const char *delim = p[0].c_str();
    make_str(p);
    char **split = ucc_str_split(str.c_str(), delim);
    EXPECT_NE(nullptr, split);
    EXPECT_EQ(ucc_str_split_count(split), p.size() - 1);
    for (auto i = 0; i < ucc_str_split_count(split); i++) {
        EXPECT_EQ(p[i + 1], split[i]);
    }
    ucc_str_split_free(split);
}

INSTANTIATE_TEST_CASE_P(
    , test_string,
    ::testing::Values(std::vector<std::string>({",", "aaa", "bbb", "ccc"}),
                      std::vector<std::string>({":", "a", "b", "c", "d", "e"}),
                      std::vector<std::string>({" ", "a", "bb"}),
                      std::vector<std::string>({"...", "aaaaaa", "b",
                                                "ccccc"})));

UCC_TEST_F(test_string, split_no_delim)
{
    std::string delim = ":";
    std::string s     = "string with no delimiter";
    char **     split = ucc_str_split(s.c_str(), delim.c_str());
    EXPECT_NE(nullptr, split);
    EXPECT_EQ(ucc_str_split_count(split), 1);
    EXPECT_EQ(s, split[0]);
    ucc_str_split_free(split);
}

UCC_TEST_F(test_string, find_last)
{
    const char* str1 = "aaa bbb aaa ccc aaa ddd";
    const char* str2 = "/tmp/build/ucc_tl_ucp/install/lib/ucc_tl_ucp.so";
    const char *s;

    s = ucc_strstr_last(str1, "aaa");
    EXPECT_EQ(16, (ptrdiff_t)s - (ptrdiff_t)str1);

    s = ucc_strstr_last(str2, "ucc_tl_ucp");
    EXPECT_EQ(strlen(str2) - strlen("ucc_tl_ucp.so"),
              (ptrdiff_t)s - (ptrdiff_t)str2);

    s = ucc_strstr_last(str1, "fff");
    EXPECT_EQ(NULL, s);

    s = ucc_strstr_last(str2, "/tmp");
    EXPECT_EQ(str2, s);
}

UCC_TEST_F(test_string, concat)
{
    char *rst;

    EXPECT_EQ(UCC_OK, ucc_str_concat("aaa", "bbb", &rst));
    EXPECT_EQ("aaabbb", std::string(rst));
    ucc_free(rst);

    EXPECT_EQ(UCC_OK, ucc_str_concat("aaabbbccc", "d", &rst));
    EXPECT_EQ("aaabbbcccd", std::string(rst));
    ucc_free(rst);

    EXPECT_EQ(UCC_OK, ucc_str_concat("aaa", "", &rst));
    EXPECT_EQ("aaa", std::string(rst));
    ucc_free(rst);

    EXPECT_EQ(UCC_OK, ucc_str_concat("", "aaa", &rst));
    EXPECT_EQ("aaa", std::string(rst));
    ucc_free(rst);
}
