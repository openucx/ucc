/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_cuda.h"
#include <iostream>
#include <dlfcn.h>
#include <stdlib.h>

ucc_pt_cuda_iface_t ucc_pt_cuda_iface = {
    .available = 0,
};

#define LOAD_CUDA_SYM(_sym, _pt_sym) ({                                    \
            void *h = dlsym(handle, _sym);                                 \
            if (dlerror() != NULL)  {                                      \
                return;                                                    \
            }                                                              \
            ucc_pt_cuda_iface. _pt_sym =                                   \
                reinterpret_cast<decltype(ucc_pt_cuda_iface. _pt_sym)>(h); \
        })


void ucc_pt_cuda_init(void)
{
    void *handle;

    handle = dlopen ("libcudart.so", RTLD_LAZY);
    if (!handle) {
        return;
    }

    LOAD_CUDA_SYM("cudaGetDeviceCount", getDeviceCount);
    LOAD_CUDA_SYM("cudaSetDevice", setDevice);
    LOAD_CUDA_SYM("cudaGetErrorString", getErrorString);
    LOAD_CUDA_SYM("cudaStreamCreateWithFlags", streamCreateWithFlags);
    LOAD_CUDA_SYM("cudaStreamDestroy", streamDestroy);
    LOAD_CUDA_SYM("cudaMalloc", cudaMalloc);
    LOAD_CUDA_SYM("cudaFree", cudaFree);
    LOAD_CUDA_SYM("cudaMemset", cudaMemset);
    LOAD_CUDA_SYM("cudaMallocManaged", cudaMallocManaged);
    LOAD_CUDA_SYM("cudaGetDeviceProperties", cudaGetDeviceProperties);
    LOAD_CUDA_SYM("cudaDeviceGetPCIBusId", cudaDeviceGetPCIBusId);

    ucc_pt_cuda_iface.available = 1;
}
