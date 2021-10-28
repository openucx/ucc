/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CUDA_REDUCE_OPS_H_
#define UCC_MC_CUDA_REDUCE_OPS_H_

#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif

#if __CUDA_ARCH__ >= 530
#define DO_OP_MAX_HALF(_v1, _v2) (__hgt(_v1, _v2) ? _v1 : _v2)
#define DO_OP_MIN_HALF(_v1, _v2) (__hlt(_v1, _v2) ? _v1 : _v2)
#define DO_OP_SUM_HALF(_v1, _v2) (__hadd(_v1, _v2))
#define DO_OP_PROD_HALF(_v1, _v2) (__hmul(_v1, _v2))
#else
#define DO_OP_MAX_HALF(_v1, _v2) ((__half2float(_v1) > __half2float(_v2)) ? _v1 : _v2)
#define DO_OP_MIN_HALF(_v1, _v2) ((__half2float(_v1) < __half2float(_v2)) ? _v1 : _v2)
#define DO_OP_SUM_HALF(_v1, _v2) (__float2half(__half2float(_v1) + __half2float(_v2)))
#define DO_OP_PROD_HALF(_v1, _v2) (__float2half(__half2float(_v1) * __half2float(_v2)))
#endif

#if CUDART_VERSION >= 11000
#if __CUDA_ARCH__ >= 800
#define DO_OP_MAX_BFLOAT16(_v1, _v2) (__hgt(_v1, _v2) ? _v1 : _v2)
#define DO_OP_MIN_BFLOAT16(_v1, _v2) (__hlt(_v1, _v2) ? _v1 : _v2)
#define DO_OP_SUM_BFLOAT16(_v1, _v2) (__hadd(_v1, _v2))
#define DO_OP_PROD_BFLOAT16(_v1, _v2) (__hmul(_v1, _v2))
#else
#define DO_OP_MAX_BFLOAT16(_v1, _v2)                                           \
    ((__bfloat162float(_v1) > __bfloat162float(_v2)) ? _v1 : _v2)
#define DO_OP_MIN_BFLOAT16(_v1, _v2)                                           \
    ((__bfloat162float(_v1) < __bfloat162float(_v2)) ? _v1 : _v2)
#define DO_OP_SUM_BFLOAT16(_v1, _v2)                                           \
    (__float2bfloat16(__bfloat162float(_v1) + __bfloat162float(_v2)))
#define DO_OP_PROD_BFLOAT16(_v1, _v2)                                          \
    (__float2bfloat16(__bfloat162float(_v1) * __bfloat162float(_v2)))
#endif
#endif

#endif
