/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "../ec_cuda.h"
#include "utils/ucc_math.h"
#include <inttypes.h>

#ifdef __cplusplus
}
#endif

#define align_pow2(_n, _p) ((_n) & ((_p) - 1))

__global__ void executor_start(ucc_ec_cuda_executor_state_t *state,
                               int *cidx)
{
    *cidx  = 0;
    *state = UCC_EC_CUDA_EXECUTOR_STARTED;
}

__global__ void executor_shutdown_ack(ucc_ec_cuda_executor_state_t *state)
{
    *state = UCC_EC_CUDA_EXECUTOR_SHUTDOWN_ACK;
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

#pragma unroll
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

__device__ void executor_reduce_float_sum_aligned_2(const float *s1,
                                                    const float *s2, float *d,
                                                    size_t count)
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
__device__ void executor_reduce_sum(const T **__restrict__ srcs,
                                    uint16_t n_srcs, T *__restrict__ d,
                                    size_t   count)
{
    const size_t step  = blockDim.x;
    const size_t start = threadIdx.x;

    for (size_t i = start; i < count; i+=step) {
        d[i] = srcs[0][i] + srcs[1][i];
        for (size_t j = 2; j < n_srcs; j++) {
            d[i] = d[i] + srcs[j][i];
        }
    }
}

template <typename T>
__device__ void executor_reduce_strided_sum(const T* __restrict__ s1,
                                            const T* __restrict__ s2,
                                            T* __restrict__ d, size_t count,
                                            size_t n_src2, size_t stride)
{
    const size_t step  = blockDim.x;
    const size_t start = threadIdx.x;
    const size_t ld    = stride / sizeof(T);

    for (size_t i = start; i < count; i+=step) {
        d[i] = s1[i] + s2[i];
        for (size_t j = 1; j < n_src2; j++) {
            d[i] = d[i] + s2[i + j*ld];
        }
     }
 }

__device__ void executor_copy_task(ucc_eee_task_copy_t *task)
{
    bool aligned = !(align_pow2((intptr_t)task->dst, 16) ||
                     align_pow2((intptr_t)task->src, 16));
    if (aligned) {
        executor_copy_aligned<uint4>((uint4 *)task->dst,
                                     (uint4 *)task->src,
                                     task->len);

    } else {
        executor_copy((char *)task->dst, (char *)task->src,
                      task->len);
    }
}

__device__ void executor_reduce_strided_task(ucc_eee_task_reduce_strided_t *task)
{
    bool aligned = !(align_pow2((intptr_t)task->dst, 16) ||
                     align_pow2((intptr_t)task->src1, 16) ||
                     align_pow2((intptr_t)task->src2, 16));

    switch (task->dt) {
    case UCC_DT_FLOAT32:
        if (task->n_src2 == 1 && aligned) {
            executor_reduce_float_sum_aligned_2((float *)task->src1,
                       (float *)task->src2, (float *)task->dst, task->count);
        } else {
            executor_reduce_strided_sum<float>((float *)task->src1,
                                               (float *)task->src2, (float *)task->dst,
                                               task->count, task->n_src2, task->stride);
        }
        break;
    case UCC_DT_FLOAT64:
        executor_reduce_strided_sum<double>((double *)task->src1,
                                            (double *)task->src2, (double *)task->dst,
                                            task->count, task->n_src2, task->stride);
        break;
    case UCC_DT_INT32:
        executor_reduce_strided_sum<int32_t>((int32_t *)task->src1,
                                             (int32_t *)task->src2, (int32_t *)task->dst,
                                             task->count, task->n_src2, task->stride);
        break;
    default:
        break;
    }
}

__device__ void executor_reduce_task(ucc_eee_task_reduce_t *task)
{
    bool aligned = !(align_pow2((intptr_t)task->dst, 16) ||
                     align_pow2((intptr_t)task->srcs[0], 16) ||
                     align_pow2((intptr_t)task->srcs[1], 16));
    switch (task->dt) {
    case UCC_DT_FLOAT32:
        if (task->n_srcs == 2 && aligned) {
            executor_reduce_float_sum_aligned_2((float *)task->srcs[0],
                      (float *)task->srcs[1], (float *)task->dst, task->count);
        } else {
            executor_reduce_sum<float>((const float **)task->srcs, task->n_srcs,
                                       (float *)task->dst, task->count);
        }
        break;
    case UCC_DT_FLOAT64:
        executor_reduce_sum<double>((const double **)task->srcs, task->n_srcs,
                                    (double *)task->dst, task->count);
        break;
    case UCC_DT_INT32:
        executor_reduce_sum<int32_t>((const int32_t **)task->srcs, task->n_srcs,
                                     (int32_t *)task->dst, task->count);
        break;
    default:
        break;
    }
}

__device__ void executor_copy_multi(ucc_eee_task_copy_multi_t *task)
{
    const size_t     step     = blockDim.x;
    size_t           min_size = task->counts[0];
    size_t           idx      = threadIdx.x;
    __shared__ int4 *dsts[UCC_EE_EXECUTOR_NUM_COPY_BUFS];
    __shared__ int4 *srcs[UCC_EE_EXECUTOR_NUM_COPY_BUFS];
    bool             aligned;

    for (int i = 0; i < task->num_vectors; i++) {
        dsts[i] = (int4*)task->dst[i];
        srcs[i] = (int4*)task->src[i];
        aligned = !(align_pow2((intptr_t)srcs[i], 16) ||
                    align_pow2((intptr_t)dsts[i], 16));
        if (!aligned) {
            break;
        }
        if (task->counts[i] < min_size) {
            min_size = task->counts[i];
        }
    }

    if (!aligned || min_size < 16) {
        for (int i = 0; i < task->num_vectors; i++) {
            executor_copy((char*)task->dst[i],
                          (char*)task->src[i],
                          task->counts[i]);
        }
        return;
    }

    const int n        = min_size / sizeof(uint4);
    const int num_iter = n / step + ((threadIdx.x < n % step) ? 1 : 0);

    for (size_t i = 0; i < num_iter; i++) {
#pragma unroll
        for (int j = 0; j < task->num_vectors; j++) {
            dsts[j][idx] = srcs[j][idx];
        }
        idx += step;
    }

    const size_t left = min_size + min_size % sizeof(uint4);

    for (int i = 0; i < task->num_vectors; i++) {
        executor_copy((char*)task->dst[i] + left,
                      (char*)task->src[i] + left,
                      task->counts[i] - left);
    }
}


__global__ void executor_kernel(volatile ucc_ec_cuda_executor_t *eee,
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
        case UCC_EE_EXECUTOR_TASK_COPY:
            executor_copy_task(&args.copy);
            break;
        case UCC_EE_EXECUTOR_TASK_REDUCE:
            executor_reduce_task(&args.reduce);
            break;
        case UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED:
            executor_reduce_strided_task(&args.reduce_strided);
            break;
        case UCC_EE_EXECUTOR_TASK_COPY_MULTI:
            executor_copy_multi(&args.copy_multi);
            break;
        default:
            break;
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

ucc_status_t ucc_ec_cuda_persistent_kernel_start(ucc_ec_cuda_executor_t *eee)
{
    cudaStream_t stream = (cudaStream_t)eee->super.ee_context;
    int          nb     = EC_CUDA_CONFIG->exec_num_workers;
    int          nt     = EC_CUDA_CONFIG->exec_num_threads;
    int          q_size = EC_CUDA_CONFIG->exec_max_tasks;

    executor_start<<<1, 1, 0, stream>>>(eee->dev_state, eee->dev_cidx);
    executor_kernel<<<nb, nt, 0, stream>>>(eee, q_size);
    executor_shutdown_ack<<<1, 1, 0, stream>>>(eee->dev_state);
    CUDA_CHECK(cudaGetLastError());

    return UCC_OK;
}

__global__ void kernel_copy_multi(ucc_eee_task_copy_multi_t args)
{
    int     blocks_per_buf = gridDim.x / args.num_vectors;
    int     buf_id         = blockIdx.x / blocks_per_buf;
    char1  *src            = (char1*)args.src[buf_id];
    char1  *dst            = (char1*)args.dst[buf_id];
    size_t  cnt            = args.counts[buf_id];
    size_t  start          = threadIdx.x + (blockIdx.x % blocks_per_buf) * blockDim.x;
    size_t  step           = blockDim.x * blocks_per_buf;

    for (size_t i = start; i < cnt; i += step) {
        dst[i] = src[i];
    }
}

__global__ void kernel_copy_multi_aligned(ucc_eee_task_copy_multi_t args)
{
    int    blocks_per_buf = gridDim.x / args.num_vectors;
    int    buf_id         = blockIdx.x / blocks_per_buf;
    int    idx            = threadIdx.x + (blockIdx.x % blocks_per_buf) * blockDim.x;
    int    step           = blockDim.x * blocks_per_buf;
    size_t n              = args.counts[buf_id] / sizeof(uint4);
    size_t num_iter       = n / step + ((idx < n % step) ? 1 : 0);
    uint4 *src            = (uint4*)args.src[buf_id];
    uint4 *dst            = (uint4*)args.dst[buf_id];

    for(size_t i = 0; i < num_iter; i++) {
        dst[i * step + idx] = src[i * step + idx];
    }

    if (idx < (args.counts[buf_id] % sizeof(uint4))) {
        ((char*)args.dst[buf_id])[args.counts[buf_id] - idx - 1] =
            ((char*)args.src[buf_id])[args.counts[buf_id] - idx - 1];
    }
}

ucc_status_t ucc_ec_cuda_copy_multi_kernel(const ucc_ee_executor_task_args_t *args,
                                           cudaStream_t stream)
{
    int nt = 1024;
    int nb = args->copy_multi.num_vectors * 4;
    int aligned = 1;

    for (int i = 0; i < args->copy_multi.num_vectors; i++) {
        if (align_pow2((intptr_t)args->copy_multi.src[i], 16) ||
            align_pow2((intptr_t)args->copy_multi.dst[i], 16)) {
            aligned = 0;
            break;
        }
    }

    if (aligned) {
        kernel_copy_multi_aligned<<<nb, nt, 0, stream>>>(args->copy_multi);
    } else {
        kernel_copy_multi<<<nb, nt, 0, stream>>>(args->copy_multi);
    }
    CUDA_CHECK(cudaGetLastError());
    return UCC_OK;
}

#ifdef __cplusplus
}
#endif
