/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_NVLS_CUH_
#define UCC_TL_CUDA_NVLS_CUH_

#include <cuda.h>
#include <stdint.h>

#define MULTIMEM_ST(val, ptr)                                                  \
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),  \
                 "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)                \
                 : "memory");

#define MULTIMEM_LD(val, ptr)                                                  \
    asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"            \
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                   \
        : "l"(ptr)                                                             \
        : "memory");

#define MULTIMEM_ST_BF16(val, ptr)                                               \
    asm volatile("multimem.st.global.v4.bf16x2 [%0], {%1,%2,%3,%4};" ::"l"(ptr), \
                 "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)                  \
                 : "memory");

#define MULTIMEM_LD_BF16(val, ptr)                                             \
    asm("multimem.ld_reduce.global.add.v4.bf16x2 {%0,%1,%2,%3}, [%4];"         \
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                   \
        : "l"(ptr)                                                             \
        : "memory");

#ifdef __cplusplus
// Traits wrapping NVLS LD/ST variants on 32-bit lanes
struct NvlsFp32Ops {
    __device__ static inline void ld(uint4 &v, const uint32_t *ptr) {
        MULTIMEM_LD(v, ptr);
    }
    __device__ static inline void st(const uint4 &v, uint32_t *ptr) {
        MULTIMEM_ST(v, ptr);
    }
};

struct NvlsBf16Ops {
    __device__ static inline void ld(uint4 &v, const uint32_t *ptr) {
        MULTIMEM_LD_BF16(v, ptr);
    }
    __device__ static inline void st(const uint4 &v, uint32_t *ptr) {
        MULTIMEM_ST_BF16(v, ptr);
    }
};
#endif // __cplusplus

#endif // UCC_TL_CUDA_NVLS_CUH_
