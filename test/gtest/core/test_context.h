/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_TEST_CONTEXT_H
#define UCC_TEST_CONTEXT_H
#include <common/test.h>
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


#endif
