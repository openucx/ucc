/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_TL_CUDA_COMMON_H_
#define UCC_TL_CUDA_COMMON_H_

#include <cuda_runtime.h>

#define CUDACHECK_GOTO(_cmd, _label, _status, _lib)                            \
    do {                                                                       \
        cudaError_t e = _cmd;                                                  \
        if (ucc_unlikely(cudaSuccess != e)) {                                  \
            tl_error(_lib, "CUDA error %d %s", e, cudaGetErrorName(e));        \
            _status = UCC_ERR_NO_MESSAGE;                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#define CUDACHECK_NORET(_cmd)                                                  \
    do {                                                                       \
        cudaError_t e = _cmd;                                                  \
        if (ucc_unlikely(cudaSuccess != e)) {                                  \
            ucc_error("CUDA error %d %s", e, cudaGetErrorName(e));             \
        }                                                                      \
    } while (0)

#define NVMLCHECK_GOTO(_cmd, _label, _status, _lib)                            \
    do {                                                                       \
        nvmlReturn_t e = _cmd;                                                 \
        if (ucc_unlikely(NVML_SUCCESS != e)) {                                 \
            tl_error(_lib, "NVML error %d %s", e, nvmlErrorString(e));         \
            _status = UCC_ERR_NO_MESSAGE;                                      \
            goto _label;                                                       \
        }                                                                      \
    } while (0)

#endif
