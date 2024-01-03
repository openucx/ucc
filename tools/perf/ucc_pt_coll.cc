/**
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_coll.h"
#include "ucc_pt_cuda.h"
#include "utils/ucc_malloc.h"

ucc_status_t ucc_pt_alloc(ucc_mc_buffer_header_t **h_ptr, size_t len,
                          ucc_memory_type_t mem_type)
{
    ucc_status_t status;
    int cuda_st;

    switch (mem_type) {
    case UCC_MEMORY_TYPE_CUDA:
        *h_ptr = new ucc_mc_buffer_header_t;
        (*h_ptr)->mt = UCC_MEMORY_TYPE_CUDA;
        cuda_st = ucc_pt_cudaMalloc(&((*h_ptr)->addr), len);
        if (cuda_st != 0) {
            return UCC_ERR_NO_MEMORY;
        }
        cuda_st = ucc_pt_cudaMemset((*h_ptr)->addr, 0, len);
        if (cuda_st != 0) {
            ucc_pt_cudaFree((*h_ptr)->addr);
            delete *h_ptr;
            return UCC_ERR_NO_MEMORY;
        }
        return UCC_OK;
    case UCC_MEMORY_TYPE_CUDA_MANAGED:
        *h_ptr = new ucc_mc_buffer_header_t;
        (*h_ptr)->mt = UCC_MEMORY_TYPE_CUDA_MANAGED;
        cuda_st = ucc_pt_cudaMallocManaged(&((*h_ptr)->addr), len);
        if (cuda_st != 0) {
            return UCC_ERR_NO_MEMORY;
        }
        cuda_st = ucc_pt_cudaMemset((*h_ptr)->addr, 0, len);
        if (cuda_st != 0) {
            ucc_pt_cudaFree((*h_ptr)->addr);
            delete *h_ptr;
            return UCC_ERR_NO_MEMORY;
        }
        return UCC_OK;
    case UCC_MEMORY_TYPE_HOST:
        *h_ptr = new ucc_mc_buffer_header_t;
        (*h_ptr)->mt = UCC_MEMORY_TYPE_HOST;
        (*h_ptr)->addr = ucc_malloc(len, "perftest data");
        if (!((*h_ptr)->addr)) {
            return UCC_ERR_NO_MEMORY;
        }
        memset((*h_ptr)->addr, 0, len);
        return UCC_OK;
    default:
        break;
    }

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
    switch (h_ptr->mt) {
    case UCC_MEMORY_TYPE_CUDA:
    case UCC_MEMORY_TYPE_CUDA_MANAGED:
        ucc_pt_cudaFree(h_ptr->addr);
        delete h_ptr;
        return UCC_OK;
    case UCC_MEMORY_TYPE_HOST:
        ucc_free(h_ptr->addr);
        delete h_ptr;
        return UCC_OK;
    default:
        break;
    }

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
