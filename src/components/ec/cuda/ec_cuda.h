/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_CUDA_H_
#define UCC_EC_CUDA_H_

#include "components/ec/base/ucc_ec_base.h"
#include "components/ec/ucc_ec_log.h"
#include "utils/ucc_mpool.h"
#include <cuda_runtime.h>

typedef enum ucc_ec_cuda_strm_task_mode {
    UCC_EC_CUDA_TASK_KERNEL,
    UCC_EC_CUDA_TASK_MEM_OPS,
    UCC_EC_CUDA_TASK_AUTO,
    UCC_EC_CUDA_TASK_LAST,
} ucc_ec_cuda_strm_task_mode_t;

typedef enum ucc_ec_cuda_task_stream_type {
    UCC_EC_CUDA_USER_STREAM,
    UCC_EC_CUDA_INTERNAL_STREAM,
    UCC_EC_CUDA_TASK_STREAM_LAST
} ucc_ec_cuda_task_stream_type_t;

typedef enum ucc_ec_task_status {
    UCC_EC_CUDA_TASK_COMPLETED,
    UCC_EC_CUDA_TASK_POSTED,
    UCC_EC_CUDA_TASK_STARTED,
    UCC_EC_CUDA_TASK_COMPLETED_ACK
} ucc_ec_task_status_t;

static inline ucc_status_t cuda_error_to_ucc_status(cudaError_t cu_err)
{
    switch(cu_err) {
    case cudaSuccess:
        return UCC_OK;
    case cudaErrorNotReady:
        return UCC_INPROGRESS;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}

typedef ucc_status_t (*ucc_ec_cuda_task_post_fn) (uint32_t *dev_status,
                                                  int blocking_wait,
                                                  cudaStream_t stream);

typedef struct ucc_ec_cuda_config {
    ucc_ec_config_t                super;
    ucc_ec_cuda_strm_task_mode_t   strm_task_mode;
    ucc_ec_cuda_task_stream_type_t task_strm_type;
    int                            stream_blocking_wait;
} ucc_ec_cuda_config_t;

typedef struct ucc_ec_cuda {
    ucc_ec_base_t                  super;
    cudaStream_t                   stream;
    ucc_mpool_t                    events;
    ucc_mpool_t                    strm_reqs;
    ucc_thread_mode_t              thread_mode;
    ucc_ec_cuda_strm_task_mode_t   strm_task_mode;
    ucc_ec_cuda_task_stream_type_t task_strm_type;
    ucc_ec_cuda_task_post_fn       post_strm_task;
    ucc_spinlock_t                 init_spinlock;
} ucc_ec_cuda_t;

typedef struct ucc_ec_cuda_event {
    cudaEvent_t    event;
} ucc_ec_cuda_event_t;

typedef struct ucc_ec_cuda_stream_request {
    uint32_t            status;
    uint32_t           *dev_status;
    cudaStream_t        stream;
} ucc_ec_cuda_stream_request_t;

extern ucc_ec_cuda_t ucc_ec_cuda;
#define CUDACHECK(cmd) do {                                                    \
        cudaError_t e = cmd;                                                   \
        if(e != cudaSuccess) {                                                 \
            ec_error(&ucc_ec_cuda.super, "cuda failed with ret:%d(%s)", e,     \
                     cudaGetErrorString(e));                                   \
            return UCC_ERR_NO_MESSAGE;                                         \
        }                                                                      \
} while(0)

#define CUDA_FUNC(_func)                                                       \
    ({                                                                         \
        ucc_status_t _status = UCC_OK;                                         \
        do {                                                                   \
            cudaError_t _result = (_func);                                     \
            if (cudaSuccess != _result) {                                      \
                ec_error(&ucc_ec_cuda.super, "%s() failed: %s",                \
                       #_func, cudaGetErrorString(_result));                   \
                _status = UCC_ERR_INVALID_PARAM;                               \
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
            if (CUDA_SUCCESS != _result) {                                     \
                cuGetErrorString(_result, &cu_err_str);                        \
                ec_error(&ucc_ec_cuda.super, "%s() failed: %s",                \
                        #_func, cu_err_str);                                   \
                _status = UCC_ERR_INVALID_PARAM;                               \
            }                                                                  \
        } while (0);                                                           \
        _status;                                                               \
    })

#define EC_CUDA_CONFIG                                                         \
    (ucc_derived_of(ucc_ec_cuda.super.config, ucc_ec_cuda_config_t))

#define UCC_EC_CUDA_INIT_STREAM() do {                                         \
    if (ucc_ec_cuda.stream == NULL) {                                          \
        cudaError_t cuda_st = cudaSuccess;                                     \
        ucc_spin_lock(&ucc_ec_cuda.init_spinlock);                             \
        if (ucc_ec_cuda.stream == NULL) {                                      \
            cuda_st = cudaStreamCreateWithFlags(&ucc_ec_cuda.stream,           \
                                                cudaStreamNonBlocking);        \
        }                                                                      \
        ucc_spin_unlock(&ucc_ec_cuda.init_spinlock);                           \
        if(cuda_st != cudaSuccess) {                                           \
            ec_error(&ucc_ec_cuda.super, "cuda failed with ret:%d(%s)",        \
                     cuda_st, cudaGetErrorString(cuda_st));                    \
            return UCC_ERR_NO_MESSAGE;                                         \
        }                                                                      \
    }                                                                          \
} while(0)

#endif
