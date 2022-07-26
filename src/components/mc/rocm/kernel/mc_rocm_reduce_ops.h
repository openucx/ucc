/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define DO_OP_SUM_FLOAT_COMPLEX(_v1, _v2)  (hipCaddf(_v1, _v2))
#define DO_OP_PROD_FLOAT_COMPLEX(_v1, _v2) (hipCmulf(_v1, _v2))
#define DO_OP_PROD_SCALAR_FLOAT_COMPLEX(_v1, _v2)                              \
    make_hipFloatComplex(hipCrealf(_v1) * _v2, hipCimagf(_v1) * _v2)

#define DO_OP_SUM_DOUBLE_COMPLEX(_v1, _v2)  (hipCadd(_v1, _v2))
#define DO_OP_PROD_DOUBLE_COMPLEX(_v1, _v2) (hipCmul(_v1, _v2))
#define DO_OP_PROD_SCALAR_DOUBLE_COMPLEX(_v1, _v2)                             \
    make_hipDoubleComplex(hipCreal(_v1) * _v2, hipCimag(_v1) * _v2)

#endif
