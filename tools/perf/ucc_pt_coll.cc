/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_coll.h"

ucc_status_t ucc_pt_alloc(ucc_mc_buffer_header_t **h_ptr, size_t len,
                          ucc_memory_type_t mem_type)
{
    ucc_status_t status;

    status = ucc_mc_alloc(h_ptr, len, mem_type);
    if (status != UCC_OK) {
        return status;
    }

    status = ucc_mc_memset((*h_ptr)->addr, 0, len, mem_type);
    if (status != UCC_OK) {
        ucc_mc_free(*h_ptr);
        return status;
    }
    return UCC_OK;
}

ucc_status_t ucc_pt_free(ucc_mc_buffer_header_t *h_ptr)
{
    return ucc_mc_free(h_ptr);
}

bool ucc_pt_coll::has_reduction()
{
    return has_reduction_;
}

bool ucc_pt_coll::has_inplace()
{
    return has_inplace_;
}

bool ucc_pt_coll::has_range()
{
    return has_range_;
}

bool ucc_pt_coll::has_bw()
{
    return has_bw_;
}
