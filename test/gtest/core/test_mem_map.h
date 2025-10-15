/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef TEST_MEM_MAP_H
#define TEST_MEM_MAP_H

#include "../common/test_ucc.h"
#include "test_context.h"
#include <vector>
#include <memory>

class test_mem_map : public test_context_config
{
protected:
    ucc_context_h        ctx_h;
    ucc_context_params_t ctx_params;
    ucc_context_config_h ctx_config;

public:
    test_mem_map();
    ~test_mem_map();

    void SetUp() override;
    void TearDown() override;
};

class test_mem_map_export : public test_mem_map
{
protected:
    void *               test_buffer;
    size_t               buffer_size;
    ucc_mem_map_params_t map_params;
    ucc_mem_map_t        segment;

public:
    test_mem_map_export();
    ~test_mem_map_export();

    void SetUp() override;
    void TearDown() override;
};

class test_mem_map_import : public test_mem_map
{
protected:
    void *               test_buffer;
    size_t               buffer_size;
    ucc_mem_map_mem_h    memh;
    size_t               memh_size;

public:
    test_mem_map_import();
    ~test_mem_map_import();

    void SetUp() override;
    void TearDown() override;
};

#endif /* TEST_MEM_MAP_H */
