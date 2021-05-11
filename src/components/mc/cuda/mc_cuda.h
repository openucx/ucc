/**
 * Copyright (C) Mellanox Technologies Ltd. 2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_CUDA_H_
#define UCC_MC_CUDA_H_

#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc_log.h"
#include "utils/ucc_mpool.h"
#include <cuda_runtime.h>

typedef enum ucc_mc_cuda_strm_task_mode {
    UCC_MC_CUDA_TASK_KERNEL,
    UCC_MC_CUDA_TASK_MEM_OPS,
    UCC_MC_CUDA_TASK_AUTO,
    UCC_MC_CUDA_TASK_LAST,
} ucc_mc_cuda_strm_task_mode_t;

typedef enum ucc_mc_cuda_task_stream_type {
    UCC_MC_CUDA_USER_STREAM,
    UCC_MC_CUDA_INTERNAL_STREAM,
    UCC_MC_CUDA_TASK_STREAM_LAST
} ucc_mc_cuda_task_stream_type_t;

typedef enum ucc_mc_task_status {
    UCC_MC_CUDA_TASK_COMPLETED,
    UCC_MC_CUDA_TASK_POSTED,
    UCC_MC_CUDA_TASK_STARTED,
    UCC_MC_CUDA_TASK_COMPLETED_ACK
} ucc_mc_task_status_t;

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

typedef ucc_status_t (*ucc_mc_cuda_task_post_fn) (uint32_t *dev_status,
                                                  int blocking_wait,
                                                  cudaStream_t stream);

typedef struct ucc_mc_cuda_config {
    ucc_mc_config_t                super;
    unsigned long                  reduce_num_blocks;
    int                            reduce_num_threads;
    ucc_mc_cuda_strm_task_mode_t   strm_task_mode;
    ucc_mc_cuda_task_stream_type_t task_strm_type;
    int                            stream_blocking_wait;
    size_t                         cuda_elem_size;
    int                            cuda_max_elems;

} ucc_mc_cuda_config_t;

typedef struct ucc_mc_cuda {
    ucc_mc_base_t                  super;
    cudaStream_t                   stream;
    ucc_mpool_t                    events;
    ucc_mpool_t                    strm_reqs;
    ucc_mpool_t                    mpool; // need more indicative name to diffirentiate from events and strm_reqs?
    int                            mpool_init_flag;
    ucc_spinlock_t                 mpool_init_spinlock;
    ucc_mc_cuda_strm_task_mode_t   strm_task_mode;
    ucc_mc_cuda_task_stream_type_t task_strm_type;
    ucc_mc_cuda_task_post_fn       post_strm_task;
} ucc_mc_cuda_t;

typedef struct ucc_cuda_mc_event {
    cudaEvent_t    event;
} ucc_mc_cuda_event_t;

typedef struct ucc_mc_cuda_stream_request {
    uint32_t            status;
    uint32_t           *dev_status;
    cudaStream_t        stream;
} ucc_mc_cuda_stream_request_t;

ucc_status_t ucc_mc_cuda_reduce(const void *src1, const void *src2,
                                void *dst, size_t count, ucc_datatype_t dt,
                                ucc_reduction_op_t op);

ucc_status_t ucc_mc_cuda_reduce_multi(const void *src1, const void *src2,
                                      void *dst, size_t size, size_t count,
                                      size_t stride, ucc_datatype_t dt,
                                      ucc_reduction_op_t op);

extern ucc_mc_cuda_t ucc_mc_cuda;
#define CUDACHECK(cmd) do {                                                    \
        cudaError_t e = cmd;                                                   \
        if(e != cudaSuccess) {                                                 \
            mc_error(&ucc_mc_cuda.super, "cuda failed with ret:%d(%s)", e,     \
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
                mc_error(&ucc_mc_cuda.super, "%s() failed: %s",                \
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
                mc_error(&ucc_mc_cuda.super, "%s() failed: %s",                \
                        #_func, cu_err_str);                                   \
                _status = UCC_ERR_INVALID_PARAM;                               \
            }                                                                  \
        } while (0);                                                           \
        _status;                                                               \
    })

#define MC_CUDA_CONFIG                                                         \
    (ucc_derived_of(ucc_mc_cuda.super.config, ucc_mc_cuda_config_t))

#endif
