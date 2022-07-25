/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_PT_ROCM_H
#define UCC_PT_ROCM_H
#include <iostream>

typedef struct ucc_pt_rocm_iface {
    int available;
    int (*getDeviceCount)(int* count);
    int (*setDevice)(int device);
    char* (*getErrorString)(int err);
} ucc_pt_rocm_iface_t;

extern ucc_pt_rocm_iface_t ucc_pt_rocm_iface;
void ucc_pt_rocm_init(void);

#define hipSuccess 0

#define STR(x) #x
#define HIP_CHECK(_call)                                        \
    do {                                                        \
        int _status = (_call);                                  \
        if (hipSuccess != _status) {                            \
            std::cerr << "UCC perftest error: " <<              \
                ucc_pt_rocm_iface.getErrorString(_status)       \
                      << " in " << STR(_call) << "\n";          \
            return _status;                                     \
        }                                                       \
    } while (0)

static inline int ucc_pt_rocmGetDeviceCount(int *count)
{
    if (!ucc_pt_rocm_iface.available) {
        return 1;
    }
    HIP_CHECK(ucc_pt_rocm_iface.getDeviceCount(count));
    return 0;
}

static inline int ucc_pt_rocmSetDevice(int device)
{
    if (!ucc_pt_rocm_iface.available) {
        return 1;
    }
    HIP_CHECK(ucc_pt_rocm_iface.setDevice(device));
    return 0;
}

#endif
