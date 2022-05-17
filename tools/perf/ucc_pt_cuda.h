/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_CUDA_H
#define UCC_PT_CUDA_H
#include <iostream>

typedef struct ucc_pt_cuda_iface {
    int available;
    int (*getDeviceCount)(int* count);
    int (*setDevice)(int device);
    char* (*getErrorString)(int err);
} ucc_pt_cuda_iface_t;

extern ucc_pt_cuda_iface_t ucc_pt_cuda_iface;
void ucc_pt_cuda_init(void);

#define cudaSuccess 0

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

#endif
