/**
* Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
*
* See file LICENSE for terms.
*/

#ifndef UCC_CUDA_DEF_H
#define UCC_CUDA_DEF_H

#include "config.h"

#if HAVE_CUDA

#include "utils/ucc_log.h"
#include <cuda_runtime.h>
#include <cuda.h>

static inline ucc_status_t cuda_error_to_ucc_status(cudaError_t cuda_status)
{
    ucc_status_t ucc_status;

    switch(cuda_status) {
    case cudaSuccess:
        ucc_status = UCC_OK;
        break;
    case cudaErrorNotReady:
        ucc_status = UCC_INPROGRESS;
        break;
    case cudaErrorInvalidValue:
        ucc_status = UCC_ERR_INVALID_PARAM;
        break;
    default:
        ucc_status = UCC_ERR_NO_MESSAGE;
    }
    return ucc_status;
}

#define CUDA_FUNC(_func)                                                       \
    ({                                                                         \
        ucc_status_t _status;                                                  \
        do {                                                                   \
            cudaError_t _result = (_func);                                     \
            if (ucc_unlikely(cudaSuccess != _result)) {                        \
                ucc_error("%s() failed: %d(%s)",                               \
                          #_func, _result, cudaGetErrorString(_result));       \
            }                                                                  \
            _status = cuda_error_to_ucc_status(_result);                       \
        } while (0);                                                           \
        _status;                                                               \
    })

#define CUDADRV_FUNC(_func)                                                    \
    ({                                                                         \
        ucc_status_t _status = UCC_OK;                                         \
        do {                                                                   \
            CUresult _result = (_func);                                        \
            const char *cu_err_str;                                            \
            if (ucc_unlikely(CUDA_SUCCESS != _result)) {                       \
                cuGetErrorString(_result, &cu_err_str);                        \
                ucc_error("%s() failed: %d(%s)",                               \
                          #_func, _result, cu_err_str);                        \
                _status = UCC_ERR_NO_MESSAGE;                                  \
            }                                                                  \
        } while (0);                                                           \
        _status;                                                               \
    })

#define CUDA_CHECK(_cmd)                                                       \
    /* coverity[dead_error_line] */                                            \
    do {                                                                       \
        ucc_status_t _cuda_status = CUDA_FUNC(_cmd);                           \
        if (ucc_unlikely(_cuda_status != UCC_OK)) {                            \
            return _cuda_status;                                               \
        }                                                                      \
    } while(0)

#define CUDA_CHECK_GOTO(_cmd, _label, _cuda_status)                            \
    do {                                                                       \
        _cuda_status = CUDA_FUNC(_cmd);                                        \
        if (ucc_unlikely(_cuda_status != UCC_OK)) {                            \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#endif

#endif
