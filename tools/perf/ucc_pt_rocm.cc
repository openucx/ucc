/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucc_pt_rocm.h"
#include <iostream>
#include <dlfcn.h>
#include <stdlib.h>

ucc_pt_rocm_iface_t ucc_pt_rocm_iface = {
    .available = 0,
};

#define LOAD_ROCM_SYM(_sym, _pt_sym) ({                                    \
            void *h = dlsym(handle, _sym);                                 \
            if ((error = dlerror()) != NULL)  {                            \
                return;                                                    \
            }                                                              \
            ucc_pt_rocm_iface. _pt_sym =                                   \
                reinterpret_cast<decltype(ucc_pt_rocm_iface. _pt_sym)>(h); \
        })

void ucc_pt_rocm_init(void)
{
    char *error;
    void *handle;

    handle = dlopen ("libamdhip64.so", RTLD_LAZY);
    if (!handle) {
        return;
    }
    LOAD_ROCM_SYM("hipGetDeviceCount", getDeviceCount);
    LOAD_ROCM_SYM("hipSetDevice", setDevice);
    LOAD_ROCM_SYM("hipGetErrorString", getErrorString);
    ucc_pt_rocm_iface.available = 1;
}
