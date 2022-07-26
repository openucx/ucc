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

#define CUDA_ERROR_TO_UCC_STATUS(_cuda_st)                                     \
    ({                                                                         \
        ucc_status_t _ucc_st;                                                  \
        switch(_cuda_st) {                                                     \
        case cudaSuccess:                                                      \
            _ucc_st = UCC_OK;                                                  \
            break;                                                             \
        case cudaErrorNotReady:                                                \
            _ucc_st = UCC_INPROGRESS;                                          \
            break;                                                             \
        case cudaErrorInvalidValue:                                            \
            _ucc_st = UCC_ERR_INVALID_PARAM;                                   \
            break;                                                             \
        default:                                                               \
            _ucc_st = UCC_ERR_NO_MESSAGE;                                      \
        }                                                                      \
        _ucc_st;                                                               \
    })

#define CUDA_FUNC(_func)                                                       \
    ({                                                                         \
        ucc_status_t _status = UCC_OK;                                         \
        do {                                                                   \
            cudaError_t _result = (_func);                                     \
            if (ucc_unlikely(cudaSuccess != _result)) {                        \
                ucc_error("%s() failed: %d(%s)",                               \
                          #_func, _result, cudaGetErrorString(_result));       \
                _status = CUDA_ERROR_TO_UCC_STATUS(_result);                   \
            }                                                                  \
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
