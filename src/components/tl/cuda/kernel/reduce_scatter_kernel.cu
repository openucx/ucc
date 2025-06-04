/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UINT32_MAX
#define __STDC_LIMIT_MACROS
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#include "utils/arch/cuda_def.h"
#include "../tl_cuda.h"

#ifdef __cplusplus
}
#endif

#define LOCAL_ST(val, ptr)                                                     \
    asm volatile("st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),          \
                 "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)                \
                 : "memory");

#define MULTIMEM_ST(val, ptr)                                                  \
    asm volatile("multimem.st.global.v4.f32 [%0], {%1,%2,%3,%4};" ::"l"(ptr),  \
                 "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w)                \
                 : "memory");

#define MULTIMEM_LD(val, ptr)                                                  \
    asm("multimem.ld_reduce.global.add.v4.f32 {%0,%1,%2,%3}, [%4];"            \
        : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)                   \
        : "l"(ptr)                                                             \
        : "memory");


// __global__ void reduce_scatter_kernel(float* src_addr, float* dst_addr, size_t src_float_count, uint32_t rank, uint32_t tsize) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         size_t idx = 0;
//         uintptr_t p = reinterpret_cast<uintptr_t>(src_addr + idx);
//         if (p % 16 != 0) {
//             printf("Misaligned! src_addr = %p\n", src_addr + idx);
//             return;
//         }

//         uint4 val;
//         MULTIMEM_LD(val, src_addr + idx);
//         reinterpret_cast<float4*>(dst_addr)[0] = *reinterpret_cast<float4*>(&val);
//     }
// }

__global__ void reduce_scatter_kernel(float* src_addr, float* dst_addr, size_t src_float_count, uint32_t rank, uint32_t tsize) {

    size_t chunk_start  = ((int64_t)src_float_count * (int64_t)rank) / (int64_t)tsize;
    size_t chunk_end    = ((int64_t)src_float_count * (int64_t)(rank + 1)) / (int64_t)tsize;
    
    size_t thread_offset = (threadIdx.x + blockIdx.x * blockDim.x) * 4;
    size_t stride = blockDim.x * gridDim.x * 4;

    for (size_t idx = chunk_start + thread_offset; idx < chunk_end; idx += stride) {
        // assert(((uintptr_t)(src_addr + idx) % 16) == 0);
        // assert(((uintptr_t)(dst_addr + (idx - chunk_start)) % 16) == 0);

        uint4 val;
        MULTIMEM_LD(val, src_addr + idx);
        float* dst = dst_addr + (idx - chunk_start);
        reinterpret_cast<float4*>(dst)[0] = *reinterpret_cast<float4*>(&val);
    }

    return;
}


#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t
post_reduce_scatter_kernel(cudaStream_t stream, CUdeviceptr src_addr, CUdeviceptr dst_addr, size_t src_size_bytes, uint32_t rank, uint32_t tsize) {
    int block_size = 1024;
    int nblocks = 10;
    
    reduce_scatter_kernel<<<nblocks, block_size, 0, stream>>>((float*) src_addr, (float*) dst_addr, src_size_bytes / sizeof(float), rank, tsize);
    // cudaError_t err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     ucc_error("Kernel launch failed: %s\n", cudaGetErrorString(err));
    //     return UCC_ERR_NO_RESOURCE;
    // }
    
    // CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
