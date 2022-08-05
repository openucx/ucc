/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CUDA_REDUCE_OPS_H_
#define UCC_EC_CUDA_REDUCE_OPS_H_

#include <cuda_fp16.h>
#if CUDART_VERSION >= 11000
#include <cuda_bf16.h>
#endif
#include <cuComplex.h>

__device__
cuDoubleComplex operator+ (const cuDoubleComplex & first,
                           const cuDoubleComplex & second) {
    return cuCadd(first, second);
}

__device__
cuDoubleComplex operator* (const cuDoubleComplex & first,
                           const cuDoubleComplex & second) {
    return cuCmul(first, second);
}

__device__
cuDoubleComplex operator* (const cuDoubleComplex & first,
                           const double & second) {
    return make_cuDoubleComplex(cuCreal(first) * second,
                                cuCimag(first) * second);
}

__device__
cuFloatComplex operator+ (const cuFloatComplex & first,
                          const cuFloatComplex & second) {
    return cuCaddf(first, second);
}

__device__
cuFloatComplex operator* (const cuFloatComplex & first,
                          const cuFloatComplex & second) {
    return cuCmulf(first, second);
}

__device__
cuFloatComplex operator* (const cuFloatComplex & first,
                          const float & second) {
    return make_cuFloatComplex(cuCrealf(first) * second,
                               cuCimagf(first) * second);
}

#endif
