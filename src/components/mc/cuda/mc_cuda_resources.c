/*
 * SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
 * Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "mc_cuda_resources.h"
#include "components/mc/ucc_mc_log.h"
#include "utils/ucc_malloc.h"

static ucc_status_t ucc_mc_cuda_chunk_alloc(ucc_mpool_t *mp, //NOLINT
                                            size_t *size_p,
                                            void **chunk_p)
{
    *chunk_p = ucc_malloc(*size_p, "mc cuda");
    if (!*chunk_p) {
        return UCC_ERR_NO_MEMORY;
    }

    return UCC_OK;
}

static void ucc_mc_cuda_chunk_init(ucc_mpool_t *mp, //NOLINT
                                   void *obj, void *chunk) //NOLINT
{
    ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *)obj;
    cudaError_t st;

    st = cudaMalloc(&h->addr, ucc_mc_cuda_config->mpool_elem_size);
    if (st != cudaSuccess) {
        // h->addr will be 0 so ucc_mc_cuda_mem_alloc_pool function will
        // return UCC_ERR_NO_MEMORY. As such mc_error message is suffice.
        cudaGetLastError();
    }
    h->from_pool = 1;
    h->mt        = UCC_MEMORY_TYPE_CUDA;
}

static void ucc_mc_cuda_chunk_release(ucc_mpool_t *mp, void *chunk) //NOLINT: mp is unused
{
    ucc_free(chunk);
}

static void ucc_mc_cuda_chunk_cleanup(ucc_mpool_t *mp, void *obj) //NOLINT: mp is unused
{
    ucc_mc_buffer_header_t *h = (ucc_mc_buffer_header_t *)obj;
    cudaError_t             st;

    st = cudaFree(h->addr);
    if (st != cudaSuccess) {
        cudaGetLastError();
    }
}

static ucc_mpool_ops_t ucc_mc_ops = {.chunk_alloc   = ucc_mc_cuda_chunk_alloc,
                                     .chunk_release = ucc_mc_cuda_chunk_release,
                                     .obj_init      = ucc_mc_cuda_chunk_init,
                                     .obj_cleanup   = ucc_mc_cuda_chunk_cleanup};

ucc_status_t ucc_mc_cuda_resources_init(ucc_mc_base_t *mc,
                                        ucc_mc_cuda_resources_t *resources)
{
    ucc_status_t status;

    CUDADRV_CHECK(cuCtxGetCurrent(&resources->cu_ctx));
    status = ucc_mpool_init(&resources->scratch_mpool, 0,
                            sizeof(ucc_mc_buffer_header_t), 0,
                            UCC_CACHE_LINE_SIZE, 1,
                            ucc_mc_cuda_config->mpool_max_elems, &ucc_mc_ops,
                            UCC_THREAD_MULTIPLE, "mc cuda mpool buffers");
    if (status != UCC_OK) {
        mc_error(mc, "failed to create scratch buffers mpool");
        return status;
    }

    status = CUDA_FUNC(cudaStreamCreateWithFlags(&resources->stream,
                                                 cudaStreamNonBlocking));
    if (status != UCC_OK) {
        mc_error(mc, "failed to create CUDA stream");
        goto free_scratch_mpool;
    }

    return UCC_OK;

free_scratch_mpool:
    ucc_mpool_cleanup(&resources->scratch_mpool, 0);
    return status;
}

void ucc_mc_cuda_resources_cleanup(ucc_mc_cuda_resources_t *resources)
{
    CUcontext tmp_context;
#if CUDA_VERSION >= 12000
    CUresult status;
    unsigned long long int cu_ctx_id;

    status = cuCtxGetId(resources->cu_ctx, &cu_ctx_id);
    if (ucc_unlikely(status != CUDA_SUCCESS)) {
        // ctx is not available, can be due to cudaDeviceReset
        return;
    }
#endif

    cuCtxPushCurrent(resources->cu_ctx);
    ucc_mpool_cleanup(&resources->scratch_mpool, 1);
    CUDA_FUNC(cudaStreamDestroy(resources->stream));
    cuCtxPopCurrent(&tmp_context);
}
