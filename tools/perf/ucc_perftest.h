/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include <ucc/api/ucc.h>
#include <iostream>
#include "config.h"

#define STR(x) #x
#define UCCCHECK_GOTO(_call, _label, _status)                                  \
    do {                                                                       \
        _status = (_call);                                                     \
        if (UCC_OK != _status) {                                               \
            std::cerr << "UCC perftest error: " << ucc_status_string(_status)  \
                      << " in " << STR(_call) << "\n";                         \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#ifdef HAVE_CUDA
#include <cuda_runtime_api.h>
#define CUDA_CHECK_GOTO(_call, _label, _status)                                \
    do {                                                                       \
        _status = (_call);                                                     \
        if (cudaSuccess != _status) {                                          \
            std::cerr << "UCC perftest error: " << cudaGetErrorString(_status) \
                      << " in " << STR(_call) << "\n";                         \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#endif