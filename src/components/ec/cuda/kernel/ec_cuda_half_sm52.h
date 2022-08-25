/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms of ucc.
 *
 * This file is copy-pasted from cuda_fp16.hpp in the CUDA toolkit and modified.
 * See the original cuda_fp16.hpp for terms of cuda_fp16.hpp.
 */

/**
 * We copy-pasted and modify cuda_fp16.hpp because half operators are only available
 * for SM>=5.3 but we need to support earlier architectures. On earlier architectures,
 * we emulate the operators by converting half to float, do the operation, then convert
 * the result back to half
 */

#pragma once

#include <cuda_fp16.h>

/* Global-space operator functions are only available to nvcc compilation */
#if defined(__CUDACC__)

/* Arithmetic FP16 operations in cuda_fp16.hpp only supported on arch >= 5.3,
 * however, we support early architectures*/
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530)
#if !defined(__CUDA_NO_HALF_OPERATORS__)

/* Some basic arithmetic operations expected of a builtin */
__device__ __forceinline__ __half operator+(const __half &lh, const __half &rh)
{
    return __float2half(__half2float(lh) + __half2float(rh));
}
__device__ __forceinline__ __half operator-(const __half &lh, const __half &rh)
{
    return __float2half(__half2float(lh) - __half2float(rh));
}
__device__ __forceinline__ __half operator*(const __half &lh, const __half &rh)
{
    return __float2half(__half2float(lh) * __half2float(rh));
}
__device__ __forceinline__ __half operator/(const __half &lh, const __half &rh)
{
    return __float2half(__half2float(lh) / __half2float(rh));
}

__device__ __forceinline__ __half &operator+=(__half &lh, const __half &rh)
{
    lh = __float2half(__half2float(lh) + __half2float(rh));
    return lh;
}
__device__ __forceinline__ __half &operator-=(__half &lh, const __half &rh)
{
    lh = __float2half(__half2float(lh) - __half2float(rh));
    return lh;
}
__device__ __forceinline__ __half &operator*=(__half &lh, const __half &rh)
{
    lh = __float2half(__half2float(lh) * __half2float(rh));
    return lh;
}
__device__ __forceinline__ __half &operator/=(__half &lh, const __half &rh)
{
    lh = __float2half(__half2float(lh) / __half2float(rh));
    return lh;
}

/* Note for increment and decrement we use the raw value 0x3C00U equating to half(1.0F), to avoid the extra conversion */
__device__ __forceinline__ __half &operator++(__half &h)
{
    __half_raw one;
    one.x = 0x3C00U;
    h += one;
    return h;
}
__device__ __forceinline__ __half &operator--(__half &h)
{
    __half_raw one;
    one.x = 0x3C00U;
    h -= one;
    return h;
}
__device__ __forceinline__ __half operator++(__half &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __half ret = h;
    __half_raw   one;
    one.x = 0x3C00U;
    h += one;
    return ret;
}
__device__ __forceinline__ __half operator--(__half &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __half ret = h;
    __half_raw   one;
    one.x = 0x3C00U;
    h -= one;
    return ret;
}

/* Unary plus and inverse operators */
__device__ __forceinline__ __half operator+(const __half &h)
{
    return h;
}
__device__ __forceinline__ __half operator-(const __half &h)
{
    return __float2half(-__half2float(h));
}

/* Some basic comparison operations to make it look like a builtin */
__device__ __forceinline__ bool operator==(const __half &lh, const __half &rh)
{
    return __half2float(lh) ==  __half2float(rh);
}
__device__ __forceinline__ bool operator!=(const __half &lh, const __half &rh)
{
    return __half2float(lh) !=  __half2float(rh);
}
__device__ __forceinline__ bool operator>(const __half &lh, const __half &rh)
{
    return __half2float(lh) >  __half2float(rh);
}
__device__ __forceinline__ bool operator<(const __half &lh, const __half &rh)
{
    return __half2float(lh) <  __half2float(rh);
}
__device__ __forceinline__ bool operator>=(const __half &lh, const __half &rh)
{
    return __half2float(lh) >=  __half2float(rh);
}
__device__ __forceinline__ bool operator<=(const __half &lh, const __half &rh)
{
    return __half2float(lh) <=  __half2float(rh);
}

#endif /* !defined(__CUDA_NO_HALF_OPERATORS__) */
#endif /* defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 530) */
#endif /* defined(__CUDACC__) */
