/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 *
 * Minimal MC component used as a loadable .so in tests.
 * The symbol name "ucc_mc_mock" matches the filename libucc_mc_mock.so,
 * which is what ucc_component_load_one() looks up via dlsym().
 */

#include "components/mc/base/ucc_mc_base.h"
#include <stdlib.h>
#include <string.h>

static ucc_status_t mock_so_init(const ucc_mc_params_t *params)
{
    (void)params;
    return UCC_OK;
}

static ucc_status_t mock_so_get_attr(ucc_mc_attr_t *attr)
{
    if (attr->field_mask & UCC_MC_ATTR_FIELD_THREAD_MODE) {
        attr->thread_mode = UCC_THREAD_SINGLE;
    }
    return UCC_OK;
}

static ucc_status_t mock_so_finalize(void)
{
    return UCC_OK;
}

static ucc_status_t mock_so_mem_query(const void *ptr, ucc_mem_attr_t *mem_attr)
{
    (void)ptr;
    (void)mem_attr;
    return UCC_ERR_NOT_FOUND;
}

static ucc_status_t mock_so_mem_alloc(ucc_mc_buffer_header_t **h_ptr,
                                      size_t size, ucc_memory_type_t mt)
{
    ucc_mc_buffer_header_t *h =
        (ucc_mc_buffer_header_t *)malloc(sizeof(*h) + size);
    if (!h) {
        return UCC_ERR_NO_MEMORY;
    }
    h->mt        = mt;
    h->from_pool = 0;
    h->addr      = (char *)h + sizeof(*h);
    *h_ptr       = h;
    return UCC_OK;
}

static ucc_status_t mock_so_mem_free(ucc_mc_buffer_header_t *h_ptr)
{
    free(h_ptr);
    return UCC_OK;
}

static ucc_config_field_t mock_config_table[] = {{NULL}};

ucc_mc_base_t ucc_mc_mock = {
    .super        = {.name = "mock"},
    .ee_type      = UCC_EE_CPU_THREAD,
    .type         = UCC_MEMORY_TYPE_HOST,
    .config_table = {.name   = "MOCK",
                     .prefix = "MC_MOCK_",
                     .table  = mock_config_table,
                     .size   = sizeof(ucc_mc_config_t)},
    .init         = mock_so_init,
    .get_attr     = mock_so_get_attr,
    .finalize     = mock_so_finalize,
    .ops          = {.mem_query = mock_so_mem_query,
                     .mem_alloc = mock_so_mem_alloc,
                     .mem_free  = mock_so_mem_free},
};
