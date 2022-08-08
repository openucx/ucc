/**
 * Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_H_
#define UCC_MC_H_

#include "ucc/api/ucc.h"
#include "components/mc/base/ucc_mc_base.h"
#include "core/ucc_dt.h"
#include "utils/ucc_math.h"

ucc_status_t ucc_mc_init(const ucc_mc_params_t *mc_params);

ucc_status_t ucc_mc_finalize();

ucc_status_t ucc_mc_available(ucc_memory_type_t mem_type);

/**
 * Query for memory attributes.
 * @param [in]        ptr       Memory pointer to query.
 * @param [in,out]    mem_attr  Memory attributes.
 */
ucc_status_t ucc_mc_get_mem_attr(const void *ptr, ucc_mem_attr_t *mem_attr);

ucc_status_t ucc_mc_alloc(ucc_mc_buffer_header_t **h_ptr, size_t len,
                          ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_free(ucc_mc_buffer_header_t *h_ptr);

ucc_status_t ucc_mc_flush(ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_memcpy(void *dst, const void *src, size_t len,
                           ucc_memory_type_t dst_mem,
                           ucc_memory_type_t src_mem);


#endif
