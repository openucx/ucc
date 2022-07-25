/**
* Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
* Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/

#ifndef UCC_ROCM_DEF_H
#define UCC_ROCM_DEF_H

#include "config.h"

#if HAVE_ROCM

#include "utils/ucc_log.h"
#include <hip/hip_runtime_api.h>

#define ROCMCHECK(cmd) do {                                                    \
        hipError_t e = cmd;                                                    \
        if(e != hipSuccess) {                                                  \
            ucc_error("ROCm failed with ret:%d(%s)", e,                        \
                      hipGetErrorString(e));                                   \
            return UCC_ERR_NO_MESSAGE;                                         \
        }                                                                      \
} while(0)

#define ROCM_FUNC(_func)                                                       \
    ({                                                                         \
        ucc_status_t _status = UCC_OK;                                         \
        do {                                                                   \
            hipError_t _result = (_func);                                      \
            if (hipSuccess != _result) {                                       \
                ucc_error("%s() failed: %d(%s)",                               \
                          #_func, _result, hipGetErrorString(_result));        \
                _status = UCC_ERR_INVALID_PARAM;                               \
            }                                                                  \
        } while (0);                                                           \
        _status;                                                               \
    })


#endif  /* HAVE_ROCM */

#endif /* UCC_ROCM_DEF_H */
