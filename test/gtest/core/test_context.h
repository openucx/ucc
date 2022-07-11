/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_CONTEXT_H
#define UCC_TEST_CONTEXT_H
#include <common/test_ucc.h>
class test_context_config : public ucc::test
{
public:
    test_context_config();
    ~test_context_config();
    ucc_lib_config_h lib_config;
    ucc_lib_params_t lib_params;
    ucc_lib_h        lib_h;
};

class test_context : public test_context_config
{
public:
    ucc_context_config_h ctx_config;
    test_context();
    ~test_context();
};

class test_context_get_attr : public test_context {
  public:
    ucc_context_h ctx_h;
    test_context_get_attr();
    ~test_context_get_attr();
};

#endif
