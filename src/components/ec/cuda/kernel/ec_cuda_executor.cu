/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifdef __cplusplus
extern "C" {
#endif

#include "../ec_cuda.h"
#include "../ec_cuda_executor.h"
#include "utils/ucc_math.h"
#include <inttypes.h>

#ifdef __cplusplus
}
#endif

#include "ec_cuda_reduce_ops.h"

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

__device__ void executor_copy(char *__restrict__ d, char *__restrict__ s,
                              size_t count)
{
    size_t start = threadIdx.x;
    const size_t step  = blockDim.x;

    for (size_t i = start; i < count; i+=step) {
        d[i] = s[i];
    }
}

__device__ void executor_copy_aligned(uint4 *__restrict__ d,
                                      uint4 *__restrict__ s, size_t count)
{
    size_t idx = threadIdx.x;
    const size_t step  = blockDim.x;
    const int    n        = count / sizeof(uint4);
    const int num_iter = n / step + ((idx < n % step) ? 1 : 0);
    char1 *s1 = (char1*)s;
    char1 *d1 = (char1*)d;

#pragma unroll
    for(int i = 0; i < num_iter; i++) {
        d[i * step + idx] = s[i * step + idx];
    }

    if (idx < count % sizeof(uint4)) {
        d1[count - idx - 1] = s1[count - idx - 1];
    }
}

__device__ void executor_copy_task(ucc_ee_executor_task_args_t *task_args)
{
    ucc_eee_task_copy_t task_copy = task_args->copy;
    bool                aligned   = !(align_pow2((intptr_t)task_copy.dst, 16) ||
                     align_pow2((intptr_t)task_copy.src, 16));
    if (aligned) {
        executor_copy_aligned((uint4 *)task_copy.dst, (uint4 *)task_copy.src,
                              task_copy.len);

    } else {
        executor_copy((char *)task_copy.dst, (char *)task_copy.src,
                      task_copy.len);
    }
}
__device__ void *executor_copy_ptr = (void *)executor_copy_task;

__device__ void executor_copy_multi(ucc_ee_executor_task_args_t *task_args)
{
    ucc_eee_task_copy_multi_t task_copy_multi = task_args->copy_multi;
    const size_t     step     = blockDim.x;
    size_t                    min_size        = task_copy_multi.counts[0];
    size_t           idx      = threadIdx.x;
    __shared__ int4 *dsts[UCC_EE_EXECUTOR_NUM_COPY_BUFS];
    __shared__ int4 *srcs[UCC_EE_EXECUTOR_NUM_COPY_BUFS];
    bool             aligned;

    for (int i = 0; i < task_copy_multi.num_vectors; i++) {
        dsts[i] = (int4 *)task_copy_multi.dst[i];
        srcs[i] = (int4 *)task_copy_multi.src[i];
        aligned = !(align_pow2((intptr_t)srcs[i], 16) ||
                    align_pow2((intptr_t)dsts[i], 16));
        if (!aligned) {
            break;
        }
        if (task_copy_multi.counts[i] < min_size) {
            min_size = task_copy_multi.counts[i];
        }
    }

    if (!aligned || min_size < 16) {
        for (int i = 0; i < task_copy_multi.num_vectors; i++) {
            executor_copy((char *)task_copy_multi.dst[i],
                          (char *)task_copy_multi.src[i],
                          task_copy_multi.counts[i]);
        }
        return;
    }

    const int n        = min_size / sizeof(uint4);
    const int num_iter = n / step + ((threadIdx.x < n % step) ? 1 : 0);

    for (size_t i = 0; i < num_iter; i++) {
#pragma unroll
        for (int j = 0; j < task_copy_multi.num_vectors; j++) {
            dsts[j][idx] = srcs[j][idx];
        }
        idx += step;
    }

    const size_t left = min_size + min_size % sizeof(uint4);

    for (int i = 0; i < task_copy_multi.num_vectors; i++) {
        executor_copy((char *)task_copy_multi.dst[i] + left,
                      (char *)task_copy_multi.src[i] + left,
                      task_copy_multi.counts[i] - left);
    }
}
__device__ void *executor_copy_multi_ptr = (void *)executor_copy_multi;

#define DECLARE_REDUCE_EXEC(DT, OP)                                            \
    static __device__ void EXECUTOR_REDUCE_##DT##_##OP(                        \
        ucc_ee_executor_task_args_t *task_args)                                \
    {                                                                          \
        ucc_eee_task_reduce_t task_reduce = task_args->reduce;                 \
        const DT **           srcs        = (const DT **)task_reduce.srcs;     \
        const uint16_t        n_srcs      = task_reduce.n_srcs;                \
        const size_t          count       = task_reduce.count;                 \
        DT *                  d           = (DT *)task_reduce.dst;             \
                                                                               \
        const size_t step  = blockDim.x;                                       \
        const size_t start = threadIdx.x;                                      \
                                                                               \
        for (size_t i = start; i < count; i += step) {                         \
            d[i] = DO_OP_##OP(srcs[0][i], srcs[1][i]);                         \
            for (size_t j = 2; j < n_srcs; j++) {                              \
                d[i] = DO_OP_##OP(d[i], srcs[j][i]);                           \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    __device__ void *EXECUTOR_REDUCE_##DT##_##OP##_ptr =                       \
        (void *)EXECUTOR_REDUCE_##DT##_##OP;

#define DECLARE_REDUCE_STRIDED_EXEC(DT, OP)                                    \
    static __device__ void EXECUTOR_REDUCE_STRIDED_##DT##_##OP(                \
        ucc_ee_executor_task_args_t *task_args)                                \
    {                                                                          \
        ucc_eee_task_reduce_strided_t task_reduce_strided =                    \
            task_args->reduce_strided;                                         \
        const DT *   s1     = (const DT *)task_reduce_strided.src1;            \
        const DT *   s2     = (const DT *)task_reduce_strided.src2;            \
        const size_t count  = task_reduce_strided.count;                       \
        const size_t n_src2 = task_reduce_strided.n_src2;                      \
        const size_t stride = task_reduce_strided.stride;                      \
        DT *         d      = (DT *)task_reduce_strided.dst;                   \
                                                                               \
        const size_t step  = blockDim.x;                                       \
        const size_t start = threadIdx.x;                                      \
        const size_t ld    = stride / sizeof(DT);                              \
                                                                               \
        for (size_t i = start; i < count; i += step) {                         \
            d[i] = DO_OP_##OP(s1[i], s2[i]);                                   \
            for (size_t j = 1; j < n_src2; j++) {                              \
                d[i] = DO_OP_##OP(d[i], s2[i + j * ld]);                       \
            }                                                                  \
        }                                                                      \
    }                                                                          \
    __device__ void *EXECUTOR_REDUCE_STRIDED_##DT##_##OP##_ptr =               \
        (void *)EXECUTOR_REDUCE_STRIDED_##DT##_##OP;

#define DECLARE_REDUCE_MIXED_EXEC(DT, OP)                                      \
    DECLARE_REDUCE_EXEC(DT, OP)                                                \
    DECLARE_REDUCE_STRIDED_EXEC(DT, OP)

#define DECLARE_REDUCE_MIXED_EXEC_COMPLEX_OPS(DT)                              \
    DECLARE_REDUCE_MIXED_EXEC(DT, SUM)                                         \
    DECLARE_REDUCE_MIXED_EXEC(DT, PROD)

#define DECLARE_REDUCE_MIXED_EXEC_REAL_OPS(DT)                                 \
    DECLARE_REDUCE_MIXED_EXEC_COMPLEX_OPS(DT)                                  \
    DECLARE_REDUCE_MIXED_EXEC(DT, MAX)                                         \
    DECLARE_REDUCE_MIXED_EXEC(DT, MIN)

#define DECLARE_REDUCE_MIXED_EXEC_INT_OPS(DT)                                  \
    DECLARE_REDUCE_MIXED_EXEC_REAL_OPS(DT)                                     \
    DECLARE_REDUCE_MIXED_EXEC(DT, LAND)                                        \
    DECLARE_REDUCE_MIXED_EXEC(DT, LOR)                                         \
    DECLARE_REDUCE_MIXED_EXEC(DT, LXOR)                                        \
    DECLARE_REDUCE_MIXED_EXEC(DT, BAND)                                        \
    DECLARE_REDUCE_MIXED_EXEC(DT, BOR)                                         \
    DECLARE_REDUCE_MIXED_EXEC(DT, BXOR)

DECLARE_REDUCE_MIXED_EXEC_REAL_OPS(float)
DECLARE_REDUCE_MIXED_EXEC_REAL_OPS(double)
DECLARE_REDUCE_MIXED_EXEC_COMPLEX_OPS(cuFloatComplex)
DECLARE_REDUCE_MIXED_EXEC_COMPLEX_OPS(cuDoubleComplex)
DECLARE_REDUCE_MIXED_EXEC_INT_OPS(int8_t)
DECLARE_REDUCE_MIXED_EXEC_INT_OPS(int16_t)
DECLARE_REDUCE_MIXED_EXEC_INT_OPS(int32_t)
DECLARE_REDUCE_MIXED_EXEC_INT_OPS(int64_t)
DECLARE_REDUCE_MIXED_EXEC_INT_OPS(uint8_t)
DECLARE_REDUCE_MIXED_EXEC_INT_OPS(uint16_t)
DECLARE_REDUCE_MIXED_EXEC_INT_OPS(uint32_t)
DECLARE_REDUCE_MIXED_EXEC_INT_OPS(uint64_t)

//Optimized function with vectorized operation for aligned data
#define VECTORIZE(OP, d, s1, s2)                                               \
    d.x = DO_OP_##OP(s1.x, s2.x);                                              \
    d.y = DO_OP_##OP(s1.y, s2.y);                                              \
    d.z = DO_OP_##OP(s1.z, s2.z);                                              \
    d.w = DO_OP_##OP(s1.w, s2.w);

#define EXECUTOR_REDUCE_float_ALIGNED_2(OP, s1, s2, d, count)                  \
    const float4 *s14      = (const float4 *)s1;                               \
    const float4 *s24      = (const float4 *)s2;                               \
    const size_t  idx      = threadIdx.x;                                      \
    const size_t  step     = blockDim.x;                                       \
    const int     n        = count / 4;                                        \
    const int     num_iter = n / step + ((idx < n % step) ? 1 : 0);            \
    float4 *      d4       = (float4 *)d;                                      \
                                                                               \
    for (int i = 0; i < num_iter; i++) {                                       \
        VECTORIZE(OP, d4[i * step + idx], s14[i * step + idx],                 \
                  s24[i * step + idx]);                                        \
    }                                                                          \
    if (idx < count % 4) {                                                     \
        d[count - idx - 1] =                                                   \
            DO_OP_##OP(s1[count - idx - 1], s2[count - idx - 1]);              \
    }

#define DECLARE_REDUCE_float_CHECK_ALIGN_EXEC(OP)                              \
    static __device__ void EXECUTOR_REDUCE_float_##OP##_CHECK_ALIGN(           \
        ucc_ee_executor_task_args_t *task_args)                                \
    {                                                                          \
        ucc_eee_task_reduce_t task_reduce = task_args->reduce;                 \
        const float **        srcs        = (const float **)task_reduce.srcs;  \
        const uint16_t        n_srcs      = task_reduce.n_srcs;                \
        const size_t          count       = task_reduce.count;                 \
        float *               d           = (float *)task_reduce.dst;          \
                                                                               \
        bool aligned = !(align_pow2((intptr_t)d, 16) ||                        \
                         align_pow2((intptr_t)srcs[0], 16) ||                  \
                         align_pow2((intptr_t)srcs[1], 16));                   \
                                                                               \
        if (n_srcs == 2 && aligned) {                                          \
            EXECUTOR_REDUCE_float_ALIGNED_2(OP, srcs[0], srcs[1], d, count);   \
        } else {                                                               \
            EXECUTOR_REDUCE_float_##OP(task_args);                             \
        }                                                                      \
    }                                                                          \
    __device__ void *EXECUTOR_REDUCE_float_##OP##_CHECK_ALIGN_ptr =            \
        (void *)EXECUTOR_REDUCE_float_##OP##_CHECK_ALIGN;

#define DECLARE_REDUCE_STRIDED_float_CHECK_ALIGN_EXEC(OP)                      \
    static __device__ void EXECUTOR_REDUCE_STRIDED_float_##OP##_CHECK_ALIGN(   \
        ucc_ee_executor_task_args_t *task_args)                                \
    {                                                                          \
        ucc_eee_task_reduce_strided_t task_reduce_strided =                    \
            task_args->reduce_strided;                                         \
        const float *s1     = (float *)task_reduce_strided.src1;               \
        const float *s2     = (float *)task_reduce_strided.src2;               \
        const size_t count  = task_reduce_strided.count;                       \
        const size_t n_src2 = task_reduce_strided.n_src2;                      \
        float *      d      = (float *)task_reduce_strided.dst;                \
                                                                               \
        bool aligned =                                                         \
            !(align_pow2((intptr_t)d, 16) || align_pow2((intptr_t)s1, 16) ||   \
              align_pow2((intptr_t)s2, 16));                                   \
                                                                               \
        if (n_src2 == 1 && aligned) {                                          \
            EXECUTOR_REDUCE_float_ALIGNED_2(OP, s1, s2, d, count);             \
        } else {                                                               \
            EXECUTOR_REDUCE_STRIDED_float_##OP(task_args);                     \
        }                                                                      \
    }                                                                          \
    __device__ void *EXECUTOR_REDUCE_STRIDED_float_##OP##_CHECK_ALIGN_ptr =    \
        (void *)EXECUTOR_REDUCE_STRIDED_float_##OP##_CHECK_ALIGN;

#define DECLARE_REDUCE_float_CHECK_ALIGN_MIXED(OP)                             \
    DECLARE_REDUCE_float_CHECK_ALIGN_EXEC(OP)                                  \
        DECLARE_REDUCE_STRIDED_float_CHECK_ALIGN_EXEC(OP)

DECLARE_REDUCE_float_CHECK_ALIGN_MIXED(SUM)
    DECLARE_REDUCE_float_CHECK_ALIGN_MIXED(PROD)
        DECLARE_REDUCE_float_CHECK_ALIGN_MIXED(MIN)
            DECLARE_REDUCE_float_CHECK_ALIGN_MIXED(MAX)

/*AVG EXECUTOR
the AVG executor is used only if the flag UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA
is true. Otherwise we use the SUM executor
*/
#define PARSE_REDUCE_TASK(task_args)                                           \
    ucc_eee_task_reduce_t task = task_args->reduce;

#define PARSE_REDUCE_STRIDED_TASK(task_args)                                   \
    ucc_eee_task_reduce_strided_t task = task_args->reduce_strided;

#define DECLARE_REDUCE_AVG_EXEC(TYPE, DT, alpha_DT, SUFFIX)                    \
    static __device__ void EXECUTOR_##TYPE##_##DT##_AVG##SUFFIX(               \
        ucc_ee_executor_task_args_t *task_args)                                \
    {                                                                          \
        PARSE_##TYPE##_TASK(task_args) const alpha_DT alpha =                  \
            (alpha_DT)task.alpha;                                              \
        const size_t count = task.count;                                       \
        DT *         d     = (DT *)task.dst;                                   \
                                                                               \
        const size_t step  = blockDim.x;                                       \
        const size_t start = threadIdx.x;                                      \
                                                                               \
        EXECUTOR_##TYPE##_##DT##_SUM##SUFFIX(task_args);                       \
                                                                               \
        for (int i = start; i < count; i += step) {                            \
            d[i] = d[i] * alpha;                                               \
        }                                                                      \
    }                                                                          \
    __device__ void *EXECUTOR_##TYPE##_##DT##_AVG##SUFFIX##_ptr =              \
        (void *)EXECUTOR_##TYPE##_##DT##_AVG##SUFFIX;

#define DECLARE_REDUCE_MIXED_AVG_EXEC(DT, alpha_DT, SUFFIX)                    \
    DECLARE_REDUCE_AVG_EXEC(REDUCE, DT, alpha_DT, SUFFIX)                      \
    DECLARE_REDUCE_AVG_EXEC(REDUCE_STRIDED, DT, alpha_DT, SUFFIX)

                DECLARE_REDUCE_MIXED_AVG_EXEC(float, float, _CHECK_ALIGN)
                    DECLARE_REDUCE_MIXED_AVG_EXEC(double, double, )
                        DECLARE_REDUCE_MIXED_AVG_EXEC(cuFloatComplex, float, )
                            DECLARE_REDUCE_MIXED_AVG_EXEC(cuDoubleComplex,
                                                          double, )

//Fill array of function pointers
#define float_TO_UCC_DT           UCC_DT_FLOAT32
#define double_TO_UCC_DT          UCC_DT_FLOAT64
#define uint8_t_TO_UCC_DT         UCC_DT_UINT8
#define uint16_t_TO_UCC_DT        UCC_DT_UINT16
#define uint32_t_TO_UCC_DT        UCC_DT_UINT32
#define uint64_t_TO_UCC_DT        UCC_DT_UINT64
#define int8_t_TO_UCC_DT          UCC_DT_INT8
#define int16_t_TO_UCC_DT         UCC_DT_INT16
#define int32_t_TO_UCC_DT         UCC_DT_INT32
#define int64_t_TO_UCC_DT         UCC_DT_INT64
#define cuFloatComplex_TO_UCC_DT  UCC_DT_FLOAT32_COMPLEX
#define cuDoubleComplex_TO_UCC_DT UCC_DT_FLOAT64_COMPLEX

#define FILL_FUNC_PTR_REDUCE(TYPE, DT, OP, SUFFIX)                             \
    CUDA_CHECK(cudaMemcpyFromSymbol(                                           \
        &(ucc_ec_cuda.exec_ptr                                                 \
              .TYPE[UCC_DT_INDEX(DT##_TO_UCC_DT)][UCC_OP_##OP]),               \
        EXECUTOR_##TYPE##_##DT##_##OP##SUFFIX##_ptr, sizeof(void *), 0,        \
        cudaMemcpyDeviceToHost));

#define FILL_FUNC_PTR_REDUCE_MIXED(DT, OP, SUFFIX)                             \
    FILL_FUNC_PTR_REDUCE(REDUCE, DT, OP, SUFFIX)                               \
    FILL_FUNC_PTR_REDUCE(REDUCE_STRIDED, DT, OP, SUFFIX)

#define FILL_FUNC_PTR_REDUCE_MIXED_COMPLEX_OPS(DT, SUFFIX)                     \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, SUM, SUFFIX)                                \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, PROD, SUFFIX)                               \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, AVG, SUFFIX)

#define FILL_FUNC_PTR_REDUCE_MIXED_REAL_OPS(DT, SUFFIX)                        \
    FILL_FUNC_PTR_REDUCE_MIXED_COMPLEX_OPS(DT, SUFFIX)                         \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, MAX, SUFFIX)                                \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, MIN, SUFFIX)

#define FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(DT, SUFFIX)                         \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, SUM, SUFFIX)                                \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, PROD, SUFFIX)                               \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, MAX, SUFFIX)                                \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, MIN, SUFFIX)                                \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, LAND, SUFFIX)                               \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, LOR, SUFFIX)                                \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, LXOR, SUFFIX)                               \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, BAND, SUFFIX)                               \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, BOR, SUFFIX)                                \
    FILL_FUNC_PTR_REDUCE_MIXED(DT, BXOR, SUFFIX)

ucc_status_t ucc_ec_cuda_executor_init_exec_ptr()
{
    CUDA_CHECK(cudaMemcpyFromSymbol(&(ucc_ec_cuda.exec_ptr.COPY),
                                    executor_copy_ptr, sizeof(void *), 0,
                                    cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpyFromSymbol(&(ucc_ec_cuda.exec_ptr.COPY_MULTI),
                                    executor_copy_multi_ptr, sizeof(void *), 0,
                                    cudaMemcpyDeviceToHost));

    FILL_FUNC_PTR_REDUCE_MIXED_REAL_OPS(float, _CHECK_ALIGN)
    FILL_FUNC_PTR_REDUCE_MIXED_REAL_OPS(double, )
    FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(uint8_t, )
    FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(uint16_t, )
    FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(uint32_t, )
    FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(uint64_t, )
    FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(int8_t, )
    FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(int16_t, )
    FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(int32_t, )
    FILL_FUNC_PTR_REDUCE_MIXED_INT_OPS(int64_t, )
    FILL_FUNC_PTR_REDUCE_MIXED_COMPLEX_OPS(cuFloatComplex, )
    FILL_FUNC_PTR_REDUCE_MIXED_COMPLEX_OPS(cuDoubleComplex, )

    return UCC_OK;
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

        auto exec_function =
            (void (*)(ucc_ee_executor_task_args_t *))args.exec_ptr;
        exec_function(&args);

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
