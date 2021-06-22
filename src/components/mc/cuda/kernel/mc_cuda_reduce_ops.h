/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CUDA_REDUCE_OPS_H_
#define UCC_MC_CUDA_REDUCE_OPS_H_

#include <cuda_fp16.h>

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

#endif
