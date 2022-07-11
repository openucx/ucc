/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_ROCM_H_
#define UCC_MC_ROCM_H_

#include "components/mc/base/ucc_mc_base.h"
#include "components/mc/ucc_mc_log.h"
#include "utils/ucc_mpool.h"
#include "utils/arch/rocm_def.h"
#include <hip/hip_runtime_api.h>
#include <hip/hip_complex.h>

static inline ucc_status_t hip_error_to_ucc_status(hipError_t hip_err)
{
    switch(hip_err) {
    case hipSuccess:
        return UCC_OK;
    case hipErrorNotReady:
        return UCC_INPROGRESS;
    default:
        break;
    }
    return UCC_ERR_NO_MESSAGE;
}

typedef ucc_status_t (*ucc_mc_rocm_task_post_fn) (uint32_t *dev_status,
                                                  int blocking_wait,
                                                  hipStream_t stream);

typedef struct ucc_mc_rocm_config {
    ucc_mc_config_t                super;
    unsigned long                  reduce_num_blocks;
    int                            reduce_num_threads;
    size_t                         mpool_elem_size;
    int                            mpool_max_elems;
} ucc_mc_rocm_config_t;

typedef struct ucc_mc_rocm {
    ucc_mc_base_t                  super;
    int                            stream_initialized;
    hipStream_t                    stream;
    ucc_mpool_t                    events;
    ucc_mpool_t                    strm_reqs;
    ucc_mpool_t                    mpool;
    int                            mpool_init_flag;
    ucc_spinlock_t                 init_spinlock;
    ucc_thread_mode_t              thread_mode;
} ucc_mc_rocm_t;


#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_mc_rocm_reduce(const void *src1, const void *src2,
                                void *dst, size_t count, ucc_datatype_t dt,
                                ucc_reduction_op_t op);

ucc_status_t ucc_mc_rocm_reduce_multi(const void *src1, const void *src2,
                                      void *dst, size_t n_vectors,
                                      size_t count, size_t stride,
                                      ucc_datatype_t dt,
                                      ucc_reduction_op_t op);

ucc_status_t
ucc_mc_rocm_reduce_multi_alpha(const void *src1, const void *src2, void *dst,
                               size_t n_vectors, size_t count, size_t stride,
                               ucc_datatype_t dt, ucc_reduction_op_t reduce_op,
                               ucc_reduction_op_t vector_op, double alpha);

#ifdef __cplusplus
}    
#endif
    
extern ucc_mc_rocm_t ucc_mc_rocm;

#define MC_ROCM_CONFIG                                                         \
    (ucc_derived_of(ucc_mc_rocm.super.config, ucc_mc_rocm_config_t))

#define UCC_MC_ROCM_INIT_STREAM() do {                                         \
    if (!ucc_mc_rocm.stream_initialized) {                                     \
        hipError_t hip_st = hipSuccess;                                        \
        ucc_spin_lock(&ucc_mc_rocm.init_spinlock);                             \
        if (!ucc_mc_rocm.stream_initialized) {                                 \
            hip_st = hipStreamCreateWithFlags(&ucc_mc_rocm.stream,             \
                                                hipStreamNonBlocking);         \
            ucc_mc_rocm.stream_initialized = 1;                                \
        }								       \
        ucc_spin_unlock(&ucc_mc_rocm.init_spinlock);                           \
        if(hip_st != hipSuccess) {                                             \
            mc_error(&ucc_mc_rocm.super, "rocm failed with ret:%d(%s)",        \
                     hip_st, hipGetErrorString(hip_st));                       \
            return hip_error_to_ucc_status(hip_st);	                       \
        }                                                                      \
    }                                                                          \
} while(0)

#endif
