/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_ROCM_REDUCE_OPS_H_
#define UCC_MC_ROCM_REDUCE_OPS_H_

#include <hip_fp16.h>
#include <hip_bfloat16.h>

#define DO_OP_MAX_HALF(_v1, _v2) ((__half2float(_v1) > __half2float(_v2)) ? _v1 : _v2)
#define DO_OP_MIN_HALF(_v1, _v2) ((__half2float(_v1) < __half2float(_v2)) ? _v1 : _v2)
#define DO_OP_SUM_HALF(_v1, _v2) (__float2half(__half2float(_v1) + __half2float(_v2)))
#define DO_OP_PROD_HALF(_v1, _v2) (__float2half(__half2float(_v1) * __half2float(_v2)))

#define DO_OP_MAX_BFLOAT16(_v1, _v2)					\
    ((static_cast<float>(_v1) > static_cast<float>(_v2)) ? _v1 : _v2)
#define DO_OP_MIN_BFLOAT16(_v1, _v2)					\
    ((static_cast<float>(_v1) < static_cast<float>(_v2)) ? _v1 : _v2)
#define DO_OP_SUM_BFLOAT16(_v1, _v2)					\
    (hip_bfloat16(static_cast<float>(_v1) + static_cast<float>(_v2)))
#define DO_OP_PROD_BFLOAT16(_v1, _v2)					\
    (hip_bfloat16(static_cast<float>(_v1) * static_cast<float>(_v2)))

#endif
