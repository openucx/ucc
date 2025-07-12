/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_CUDA_H
#define UCC_PT_CUDA_H
#include <iostream>
#include <unistd.h>

#define cudaSuccess 0
#define cudaStreamNonBlocking 0x01  /**< Stream does not synchronize with stream 0 (the NULL stream) */
#define cudaMemAttachGlobal   0x01  /**< Memory can be accessed by any stream on any device*/
typedef struct CUStream_st *cudaStream_t;
typedef struct cudaDeviceProp {
    char name[256];
    char padding[2048]; /* take extra space to avoid future changes, real size is 1080 - 256 = 824 on x86 */
} cudaDeviceProp;

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
    int (*getDeviceInfo)(std::string &info);
    int (*streamCreateWithFlags)(cudaStream_t *stream, unsigned int flags);
    int (*streamDestroy)(cudaStream_t stream);
    char* (*getErrorString)(int err);
    int (*cudaMalloc)(void **devptr, size_t size);
    int (*cudaMallocManaged)(void **ptr, size_t size, unsigned int flags);
    int (*cudaFree)(void *devptr);
    int (*cudaMemset)(void *devptr, int value, size_t count);
    int (*cudaGetDeviceProperties)(void *prop, int device);
    int (*cudaDeviceGetPCIBusId)(char *pciBusId, int len, int device);
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

static inline int ucc_pt_cudaMalloc(void **devptr, size_t size)
{
    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }
    CUDA_CHECK(ucc_pt_cuda_iface.cudaMalloc(devptr, size));
    return 0;
}

static inline int ucc_pt_cudaMallocManaged(void **ptr, size_t size)
{
    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }
    CUDA_CHECK(ucc_pt_cuda_iface.cudaMallocManaged(ptr, size,
               cudaMemAttachGlobal));
    return 0;
}

static inline int ucc_pt_cudaFree(void *devptr)
{
    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }
    CUDA_CHECK(ucc_pt_cuda_iface.cudaFree(devptr));
    return 0;
}

static inline int ucc_pt_cudaMemset(void *devptr, int value, size_t count)
{
    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }
    CUDA_CHECK(ucc_pt_cuda_iface.cudaMemset(devptr, value, count));
    return 0;
}

static inline int ucc_pt_cudaGetDeviceInfo(int device, std::string &info)
{
    char pciBusId[256];
    char hostname[256];
    cudaDeviceProp prop;

    if (!ucc_pt_cuda_iface.available) {
        return 1;
    }

    if (gethostname(hostname, sizeof(hostname)) == 0) {
        info.append(hostname);
        info.append(" - ");
    }

    CUDA_CHECK(ucc_pt_cuda_iface.cudaGetDeviceProperties(&prop, device));
    info.append(prop.name);
    info.append(" ");
    CUDA_CHECK(ucc_pt_cuda_iface.cudaDeviceGetPCIBusId(pciBusId,
                                                       sizeof(pciBusId), device));
    info.append(pciBusId);

    return 0;
}

#endif
