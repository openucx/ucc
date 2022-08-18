/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_CUDA_H
#define UCC_PT_CUDA_H
#include <iostream>

#define cudaSuccess 0
#define cudaStreamNonBlocking 0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */
typedef struct CUStream_st *cudaStream_t;

#define STR(x) #x
#define CUDA_CHECK(_call)                                       \
    do {                                                        \
        int _status = (_call);                                  \
        if (cudaSuccess != _status) {                           \
            std::cerr << "UCC perftest error: " <<              \
                ucc_pt_cuda_iface.getErrorString(_status)       \
                      << " in " << STR(_call) << "\n";          \
            return _status;                                     \
        }                                                       \
    } while (0)

typedef struct ucc_pt_cuda_iface {
    int available;
    int (*getDeviceCount)(int* count);
    int (*setDevice)(int device);
    int (*streamCreateWithFlags)(cudaStream_t *stream, unsigned int flags);
    int (*streamDestroy)(cudaStream_t stream);
    char* (*getErrorString)(int err);
} ucc_pt_cuda_iface_t;

extern ucc_pt_cuda_iface_t ucc_pt_cuda_iface;

void ucc_pt_cuda_init(void);

static inline int ucc_pt_cudaGetDeviceCount(int *count)
{
    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }
    CUDA_CHECK(ucc_pt_cuda_iface.getDeviceCount(count));
    return 0;
}

static inline int ucc_pt_cudaSetDevice(int device)
{
    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }
    CUDA_CHECK(ucc_pt_cuda_iface.setDevice(device));
    return 0;
}

static inline int ucc_pt_cudaStreamCreateWithFlags(cudaStream_t *stream,
                                                   unsigned int flags)
{
    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }
    CUDA_CHECK(ucc_pt_cuda_iface.streamCreateWithFlags(stream, flags));
    return 0;
}

static inline int ucc_pt_cudaStreamDestroy(cudaStream_t stream)
{
    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }
    CUDA_CHECK(ucc_pt_cuda_iface.streamDestroy(stream));
    return 0;
}

#endif
