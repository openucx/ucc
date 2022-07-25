/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (C) Advanced Micro Devices, Inc. 2022. ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ec_rocm.h"
#include "utils/ucc_math.h"
#include <inttypes.h>

#define align_pow2(_n, _p) ((_n) & ((_p) - 1))

__global__ void executor_start(ucc_ec_rocm_executor_state_t *state,
                               int *cidx)
{
    *cidx  = 0;
    *state = UCC_EC_ROCM_EXECUTOR_STARTED;
}

__global__ void executor_shutdown_ack(ucc_ec_rocm_executor_state_t *state)
{
    *state = UCC_EC_ROCM_EXECUTOR_SHUTDOWN_ACK;
}

template <typename T>
__device__ void executor_copy(T* __restrict__ d, T* __restrict__ s,
                              size_t count)
{
    size_t start = threadIdx.x;
    const size_t step  = blockDim.x;

    for (size_t i = start; i < count; i+=step) {
        d[i] = s[i];
    }
}

template <typename T>
__device__ void executor_copy_aligned(T* __restrict__ d, T* __restrict__ s,
                                      size_t count)
{
    size_t idx = threadIdx.x;
    const size_t step  = blockDim.x;
    const int n = count / sizeof(T);
    const int num_iter = n / step + ((idx < n % step) ? 1 : 0);
    char1 *s1 = (char1*)s;
    char1 *d1 = (char1*)d;

#pragma unroll 4
    for(int i = 0; i < num_iter; i++) {
        d[i * step + idx] = s[i * step + idx];
    }

    if (idx < count % sizeof(T)) {
        d1[count - idx - 1] = s1[count - idx - 1];
    }
}

__device__ inline void add_float4(float4 &d, const float4 &x, const float4 &y)
{
    d.x = x.x + y.x;
    d.y = x.y + y.y;
    d.z = x.z + y.z;
    d.w = x.w + y.w;
}

__device__ void executor_reduce_float(const float *s1, const float *s2,
                                      float *d, size_t count)
{
    const float4 *s14      = (const float4*)s1;
    const float4 *s24      = (const float4*)s2;
    float4       *d4       = (float4*)d;
    const size_t  idx      = threadIdx.x;
    const size_t  step     = blockDim.x;
    const int     n        = count / 4;
    const int     num_iter = n / step + ((idx < n % step) ? 1 : 0);

    for(int i = 0; i < num_iter; i++) {
        add_float4(d4[i * step + idx], s14[i * step + idx],
                   s24[i * step + idx]);
    }
    if (idx < count % 4) {
        d[count - idx - 1] = s1[count - idx - 1] + s2[count - idx - 1];
    }
}

template <typename T>
__device__ void executor_reduce(const T* __restrict__ s1,
                                const T* __restrict__ s2,
                                T* __restrict__ d, size_t count)
{
    const size_t step  = blockDim.x;
    const size_t start = threadIdx.x;

    for (size_t i = start; i < count; i+=step) {
        d[i] = s1[i] + s2[i];
    }
}

template <typename T>
__device__ void executor_reduce_multi(const T* __restrict__ s1,
                                      const T* __restrict__ s2,
                                      T* __restrict__ d, size_t count,
                                      size_t size, size_t stride)
{
    const size_t step  = blockDim.x;
    const size_t start = threadIdx.x;
    const size_t ld    = stride / sizeof(T);

    for (size_t i = start; i < count; i+=step) {
        d[i] = s1[i] + s2[i];
        for (size_t j = 1; j < size; j++) {
            d[i] = d[i] + s2[i + j*ld];
        }
    }
}

__global__ void executor_kernel(volatile ucc_ec_rocm_executor_t *eee,
                                int q_size)
{
    const uint32_t  worker_id   = blockIdx.x;
    const uint32_t  num_workers = gridDim.x;
    bool            is_master   = (threadIdx.x == 0) ? true: false;
    int cidx_local, pidx_local;
    volatile int *pidx, *cidx;
    ucc_ee_executor_task_t *tasks;
    __shared__ ucc_ee_executor_task_args_t args;
    __shared__ bool worker_done;

    if (is_master) {
        cidx_local = worker_id;
        pidx       = eee->dev_pidx;
        cidx       = eee->dev_cidx;
        tasks      = eee->dev_tasks;
    }

    worker_done = false;
    __syncthreads();
    while (1) {
        if (is_master) {
            while ((*cidx % num_workers) != worker_id);
            do {
                pidx_local = *pidx;
            } while (*cidx == pidx_local);
            (*cidx)++;
            worker_done = (pidx_local == -1);
            if (!worker_done) {
                args = tasks[cidx_local].args;
            }
        }
        __syncthreads();
        if (worker_done) {
            return;
        }
        switch (args.task_type) {
            bool aligned;
            case UCC_EE_EXECUTOR_TASK_TYPE_COPY:
                aligned = !(align_pow2((intptr_t)args.bufs[0], 16) ||
                            align_pow2((intptr_t)args.bufs[1], 16));
                if (aligned) {
                    executor_copy_aligned<uint4>((uint4*)args.bufs[0],
                                                 (uint4*)args.bufs[1],
                                                 args.count);

                } else {
                    executor_copy((char*)args.bufs[0],
                                  (char*)args.bufs[1],
                                   args.count);
                }
                break;
            case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE:
                aligned = !(align_pow2((intptr_t)args.bufs[0], 16) ||
                            align_pow2((intptr_t)args.bufs[1], 16) ||
                            align_pow2((intptr_t)args.bufs[2], 16));
                switch (args.dt)
                {
                case UCC_DT_FLOAT32:
                    if (aligned) {
                        executor_reduce_float((float*)args.bufs[1],
                                              (float*)args.bufs[2],
                                              (float*)args.bufs[0],
                                              args.count);
                    } else {
                        executor_reduce<float>((float*)args.bufs[1],
                                               (float*)args.bufs[2],
                                               (float*)args.bufs[0],
                                               args.count);
                    }
                    break;
                case UCC_DT_FLOAT64:
                    executor_reduce<double>((double*)args.bufs[1],
                                            (double*)args.bufs[2],
                                            (double*)args.bufs[0],
                                             args.count);
                    break;
                case UCC_DT_INT32:
                    executor_reduce<int32_t>((int32_t*)args.bufs[1],
                                             (int32_t*)args.bufs[2],
                                             (int32_t*)args.bufs[0],
                                              args.count);
                    break;

                default:
                    break;
                }
                break;
            case UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI:
                switch(args.dt) {
                case UCC_DT_FLOAT32:
                    executor_reduce_multi<float>((float*)args.bufs[1],
                                                 (float*)args.bufs[2],
                                                 (float*)args.bufs[0],
                                                 args.count, args.size,
                                                 args.stride);
                    break;
                case UCC_DT_FLOAT64:
                    executor_reduce_multi<double>((double*)args.bufs[1],
                                                  (double*)args.bufs[2],
                                                  (double*)args.bufs[0],
                                                  args.count, args.size,
                                                  args.stride);
                    break;
                case UCC_DT_INT32:
                    executor_reduce_multi<int32_t>((int32_t*)args.bufs[1],
                                                   (int32_t*)args.bufs[2],
                                                   (int32_t*)args.bufs[0],
                                                   args.count, args.size,
                                                   args.stride);
                    break;
                }
                break;
            default: break;
        }
        __syncthreads();
        __threadfence_system();
        if (is_master) {
            tasks[cidx_local].status = UCC_OK;
            cidx_local = (cidx_local + num_workers) %q_size;
        }
    }
}

#ifdef __cplusplus
extern "C" {
#endif

ucc_status_t ucc_ec_rocm_persistent_kernel_start(ucc_ec_rocm_executor_t *eee)
{
    hipStream_t stream  = (hipStream_t)eee->super.ee_context;
    int          nb     = EC_ROCM_CONFIG->exec_num_workers;
    int          nt     = EC_ROCM_CONFIG->exec_num_threads;
    int          q_size = EC_ROCM_CONFIG->exec_max_tasks;

    executor_start<<<1, 1, 0, stream>>>(eee->dev_state, eee->dev_cidx);
    executor_kernel<<<nb, nt, 0, stream>>>(eee, q_size);
    executor_shutdown_ack<<<1, 1, 0, stream>>>(eee->dev_state);
    ROCMCHECK(hipGetLastError());

    return UCC_OK;
}


#ifdef __cplusplus
}
#endif
