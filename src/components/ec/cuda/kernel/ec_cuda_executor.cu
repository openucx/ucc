/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UINT32_MAX
#define __STDC_LIMIT_MACROS
#include "utils/arch/cuda_def.h"
#include <stdint.h>
#endif

extern "C" {
#include "../ec_cuda.h"
}
#include "ec_cuda_reduce_ops.h"
#include <cooperative_groups.h>
using namespace cooperative_groups;

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

template<int UNROLL>
__device__ void executor_copy_task(ucc_eee_task_copy_t &task)
{
    size_t      count     = task.len;
    const char *s1        = reinterpret_cast<const char*>(task.src);
    char       *d1        = reinterpret_cast<char *>(task.dst);

    if (!(align_pow2((intptr_t)s1, sizeof(vectype)) ||
          align_pow2((intptr_t)d1, sizeof(vectype)))) {
        int            warp      = threadIdx.x / WARP_SIZE;
        int            num_warps = blockDim.x / WARP_SIZE;
        int            idx       = threadIdx.x % WARP_SIZE;
        const vectype *s4        = reinterpret_cast<const vectype*>(s1);
        vectype       *d4        = reinterpret_cast<vectype*>(d1);
        size_t         num_lines = (count / (WARP_SIZE * UNROLL * sizeof(vectype))) *
                                   (WARP_SIZE * UNROLL);
        vectype        tmp[UNROLL];

        for (size_t line = warp * WARP_SIZE * UNROLL + idx;
            line < num_lines; line += num_warps * WARP_SIZE * UNROLL) {
            #pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                tmp[i] = s4[line + WARP_SIZE * i];
            }

            #pragma unroll
            for (int i = 0; i < UNROLL; i++) {
                d4[line + WARP_SIZE * i] = tmp[i];
            }
        }

        count = count - num_lines * sizeof(vectype);
        if (count == 0) {
            return;
        }

        s4 = s4 + num_lines;
        d4 = d4 + num_lines;
        num_lines = count / sizeof(vectype);
        for (int line = threadIdx.x; line < num_lines; line += blockDim.x) {
            d4[line] =  s4[line];
        }

        count = count - num_lines * sizeof(vectype);
        if (count == 0) {
            return;
        }

        s1 = reinterpret_cast<const char*>(s4 + num_lines);
        d1 = reinterpret_cast<char*>(d4 + num_lines);
    }

    for (size_t line = threadIdx.x; line < count; line += blockDim.x) {
        d1[line] = s1[line];
    }
}

#define LAUNCH_REDUCE_A(NAME, _Type, _AlphaType, _task, _unroll, ...)          \
    do {                                                                       \
        if (_task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {                 \
            ucc_reduce_cuda_##NAME<_Type, _AlphaType, true, false, _unroll,    \
                                   ucc_eee_task_reduce_t>(_task->reduce,       \
                                                          _task->flags);       \
        } else {                                                               \
            ucc_reduce_cuda_##NAME<_Type, _AlphaType, true, true, _unroll,     \
                                   ucc_eee_task_reduce_strided_t>(             \
                _task->reduce_strided, _task->flags);                          \
        }                                                                      \
        return UCC_OK;                                                         \
    } while (0)

#define LAUNCH_REDUCE(NAME, _Type, _task, _unroll, ...)                        \
    LAUNCH_REDUCE_A(NAME, _Type, _Type, _task, _unroll)

__device__ ucc_status_t executor_reduce(ucc_ee_executor_task_args_t *task)
{
    ucc_reduction_op_t op;
    ucc_datatype_t     dt;
    size_t             count;

    if (task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE) {
        dt      = task->reduce.dt;
        count   = task->reduce.count;
        op      = task->reduce.op;
    } else {
        ucc_assert_system(task->task_type == UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED);
        dt      = task->reduce_strided.dt;
        count   = task->reduce_strided.count;
        op      = task->reduce_strided.op;
    }
    if (count == 0) {
        return UCC_OK;
    }

    switch (dt) {
    case UCC_DT_INT8:
        DT_REDUCE_INT(int8_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    case UCC_DT_INT16:
        DT_REDUCE_INT(int16_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    case UCC_DT_INT32:
        DT_REDUCE_INT(int32_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_FOUR);
        break;
    case UCC_DT_INT64:
        DT_REDUCE_INT(int64_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_FOUR);
        break;
    case UCC_DT_UINT8:
        DT_REDUCE_INT(uint8_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    case UCC_DT_UINT16:
        DT_REDUCE_INT(uint16_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    case UCC_DT_UINT32:
        DT_REDUCE_INT(uint32_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    case UCC_DT_UINT64:
        DT_REDUCE_INT(uint64_t, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    case UCC_DT_FLOAT16:
        DT_REDUCE_FLOAT(__half, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    case UCC_DT_FLOAT32:
#if SIZEOF_FLOAT == 4
        DT_REDUCE_FLOAT(float, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_FOUR);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64:
#if SIZEOF_DOUBLE == 8
        DT_REDUCE_FLOAT(double, task, op, REDUCE_LOOP_UNROLL_TRIGGERED_FOUR);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT32_COMPLEX:
#if SIZEOF_CUFLOATCOMPLEX == 8
        DT_REDUCE_FLOAT_COMPLEX(cuFloatComplex, float, task, op,
                                REDUCE_LOOP_UNROLL_TRIGGERED_FOUR);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_FLOAT64_COMPLEX:
#if SIZEOF_CUDOUBLECOMPLEX == 16
        DT_REDUCE_FLOAT_COMPLEX(cuDoubleComplex, double, task, op,
                                REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
#else
        return UCC_ERR_NOT_SUPPORTED;
#endif
    case UCC_DT_BFLOAT16:
        ucc_assert_system(2 == sizeof(__nv_bfloat16));
        DT_REDUCE_FLOAT(__nv_bfloat16, task, op,
                        REDUCE_LOOP_UNROLL_TRIGGERED_TWO);
        break;
    default:
        return UCC_ERR_NOT_SUPPORTED;
    }
    return UCC_OK;
}

template<bool useCoopLaunch>
__global__ void executor_kernel(volatile ucc_ec_cuda_executor_t *eee,
                                int q_size)
{
    const uint32_t  worker_id   = blockIdx.x;
    const uint32_t  num_workers = gridDim.x;
    bool            is_master   = (threadIdx.x == 0) ? true: false;
    grid_group      grid        = this_grid();
    int cidx_local, pidx_local;
    volatile int *pidx, *cidx;
    ucc_ee_executor_task_args_t *tasks;
    __shared__ ucc_ee_executor_task_args_t args;
    __shared__ bool worker_done;

    if (is_master) {
        if (useCoopLaunch) {
            *eee->dev_cidx  = 0;
            *eee->dev_state = UCC_EC_CUDA_EXECUTOR_STARTED;
        }
        cidx_local = worker_id;
        pidx       = eee->dev_pidx;
        cidx       = eee->dev_cidx;
        tasks      = eee->dev_tasks;
    }

    worker_done = false;
    if (useCoopLaunch) {
        grid.sync();
    } else {
        __syncthreads();
    }
    while (1) {
        if (is_master) {
            while ((*cidx % num_workers) != worker_id);
            do {
                pidx_local = *pidx;
            } while (*cidx == pidx_local);
            (*cidx)++;
            worker_done = (pidx_local == -1);
            if (!worker_done) {
                args = tasks[cidx_local];
            }
        }
        __syncthreads();
        if (worker_done) {
            if (useCoopLaunch) {
                grid.sync();
                if (is_master && (worker_id == 0)) {
                    *eee->dev_state = UCC_EC_CUDA_EXECUTOR_SHUTDOWN_ACK;
                }
            }
            return;
        }
        switch (args.task_type) {
        case UCC_EE_EXECUTOR_TASK_COPY:
            executor_copy_task<COPY_LOOP_UNROLL>(args.copy);
            break;
        case UCC_EE_EXECUTOR_TASK_REDUCE:
        case UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED:
            executor_reduce(&args);
            break;
        default:
            break;
        }
        __syncthreads();
        __threadfence_system();
        if (is_master) {
            tasks[cidx_local].task_type = UCC_EE_EXECUTOR_TASK_LAST;
            cidx_local = (cidx_local + num_workers) % q_size;
        }
    }
}


extern "C" {
ucc_status_t ucc_ec_cuda_executor_kernel_calc_max_threads(int *max)
{
    int use_coop_launch = EC_CUDA_CONFIG->use_cooperative_launch;
    cudaFuncAttributes attr;

    CUDA_CHECK(cudaFuncGetAttributes(
        &attr,
        (use_coop_launch ? executor_kernel<true> : executor_kernel<false>)));
    *max = (attr.maxThreadsPerBlock / WARP_SIZE) * WARP_SIZE;
    return UCC_OK;
}

ucc_status_t ucc_ec_cuda_persistent_kernel_start(
    ucc_ec_cuda_executor_t *eee, unsigned num_threads, unsigned num_blocks)
{
    cudaStream_t stream        = (cudaStream_t)eee->super.ee_context;
    int          q_size        = EC_CUDA_CONFIG->exec_max_tasks;
    int          useCoopLaunch = EC_CUDA_CONFIG->use_cooperative_launch;

    if (useCoopLaunch) {
        void *kernelArgs[] = {&eee, &q_size};
        dim3  dimBlock(num_threads, 1, 1);
        dim3  dimGrid(num_blocks, 1, 1);
        cudaLaunchCooperativeKernel(
            (void *)executor_kernel<true>,
            dimGrid,
            dimBlock,
            kernelArgs,
            0,
            stream);
    } else {
        executor_start<<<1, 1, 0, stream>>>(eee->dev_state, eee->dev_cidx);
        executor_kernel<false>
            <<<num_blocks, num_threads, 0, stream>>>(eee, q_size);
        executor_shutdown_ack<<<1, 1, 0, stream>>>(eee->dev_state);
    }
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

}
