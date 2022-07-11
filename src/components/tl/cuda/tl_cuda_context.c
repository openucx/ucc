/**
 * Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_cuda.h"
#include "utils/arch/cpu.h"
#include "utils/arch/cuda_def.h"
#include <tl_cuda_topo.h>
#include <cuda_runtime.h>
#include <cuda.h>

static ucc_mpool_ops_t ucc_tl_cuda_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = NULL,
    .obj_cleanup   = NULL,
};

UCC_CLASS_INIT_FUNC(ucc_tl_cuda_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_cuda_context_config_t *tl_cuda_config =
        ucc_derived_of(config, ucc_tl_cuda_context_config_t);
    ucc_status_t status;
    int num_devices;
    cudaError_t cuda_st;
    CUcontext cu_ctx;
    CUresult cu_st;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_cuda_config->super,
                              params->context);
    memcpy(&self->cfg, tl_cuda_config, sizeof(*tl_cuda_config));

    cuda_st = cudaGetDeviceCount(&num_devices);
    if (cuda_st != cudaSuccess) {
        tl_info(self->super.super.lib, "failed to get number of GPU devices"
                "%d %s", cuda_st, cudaGetErrorName(cuda_st));
        return UCC_ERR_NO_MESSAGE;
    } else if (num_devices == 0) {
        tl_info(self->super.super.lib, "no GPU devices found");
        return UCC_ERR_NO_RESOURCE;
    }

    cu_st = cuCtxGetCurrent(&cu_ctx);
    if (cu_ctx == NULL || cu_st != CUDA_SUCCESS) {
        tl_info(self->super.super.lib,
                "cannot create CUDA TL context without active CUDA context");
        return UCC_ERR_NO_RESOURCE;
    }

    status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_cuda_task_t), 0,
                            UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                            &ucc_tl_cuda_req_mpool_ops, params->thread_mode,
                            "tl_cuda_req_mp");
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_cuda_req mpool");
        return status;
    }

    CUDA_CHECK_GOTO(cudaGetDevice(&self->device), free_mpool, status);
    status = ucc_tl_cuda_topo_create(self->super.super.lib, &self->topo);
    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_cuda_topo");
        goto free_mpool;
    }
    status = ucc_tl_cuda_topo_get_pci_id(self->super.super.lib, self->device,
                                         &self->device_id);
    if (status != UCC_OK) {
        tl_error(self->super.super.lib, "failed to get pci id");
        goto free_mpool;
    }

    self->ipc_cache = kh_init(tl_cuda_ep_hash);
    tl_info(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;

free_mpool:
    ucc_mpool_cleanup(&self->req_mp, 1);
    return status;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_cuda_context_t)
{
    tl_info(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_tl_cuda_topo_destroy(self->topo);
    ucc_mpool_cleanup(&self->req_mp, 1);
}

UCC_CLASS_DEFINE(ucc_tl_cuda_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_cuda_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    if (attr->attr.mask & UCC_CONTEXT_ATTR_FIELD_CTX_ADDR_LEN) {
        attr->attr.ctx_addr_len = 0;
    }
    attr->topo_required = 1;
    return UCC_OK;
}
