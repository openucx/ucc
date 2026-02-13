/**
 * Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * Copyright (c) Facebook, Inc. and its affiliates. 2021.
 *
 * See file LICENSE for terms.
 */

#include "tl_nccl.h"
#include "components/mc/ucc_mc.h"
#include "components/ec/ucc_ec.h"
#include "core/ucc_ee.h"
#include "utils/arch/cpu.h"

static ucc_status_t ucc_tl_nccl_nb_progress(ucc_tl_nccl_task_t *task) {
#if NCCL_USE_NON_BLOCKING
    ucc_tl_nccl_team_t *team = TASK_TEAM(task);
    ncclResult_t nccl_status, st;

    if (task->nccl_progress_st == UCC_INPROGRESS) {
        st = ncclCommGetAsyncError(team->nccl_comm, &nccl_status);
        if (st != ncclSuccess ||
            (nccl_status != ncclSuccess && nccl_status != ncclInProgress)) {
            tl_error(UCC_TL_TEAM_LIB(team), "NCCL error %d %s",
                     st != ncclSuccess ? st : nccl_status,
                     ncclGetErrorString(st != ncclSuccess ? st : nccl_status));
            return UCC_ERR_NO_MESSAGE;
        }
        if (nccl_status == ncclInProgress) {
            return UCC_INPROGRESS;
        }
    }
#endif
    return UCC_OK;
}

void ucc_tl_nccl_event_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t status;

    status = ucc_tl_nccl_nb_progress(task);
    if (status != UCC_OK) {
        coll_task->status = status;
        return;
    }

    ucc_assert(task->completed != NULL);
    status = ucc_ec_event_test(task->completed, UCC_EE_CUDA_STREAM);
    coll_task->status = status;
#ifdef HAVE_PROFILING_TL_NCCL
    if (coll_task->status == UCC_OK) {
        UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_coll_done", 0);
    }
#endif
}

void ucc_tl_nccl_driver_collective_progress(ucc_coll_task_t *coll_task)
{
    ucc_tl_nccl_task_t *task = ucc_derived_of(coll_task, ucc_tl_nccl_task_t);
    ucc_status_t status;

    status = ucc_tl_nccl_nb_progress(task);
    if (status != UCC_OK) {
        coll_task->status = status;
        return;
    }

    coll_task->status = task->host_status;
#ifdef HAVE_PROFILING_TL_NCCL
    if (coll_task->status == UCC_OK) {
        UCC_TL_NCCL_PROFILE_REQUEST_EVENT(coll_task, "nccl_coll_done", 0);
    }
#endif
}

static void ucc_tl_nccl_req_mpool_obj_cleanup(ucc_mpool_t *mp, void *obj)
{
    ucc_coll_task_destruct(obj);
}

static void ucc_tl_nccl_req_mpool_obj_init(ucc_mpool_t *mp, void *obj,
                                           void *chunk)
{
    ucc_tl_nccl_task_t *req = (ucc_tl_nccl_task_t*) obj;

    ucc_coll_task_construct(&req->super);
    req->super.progress   = ucc_tl_nccl_event_collective_progress;
    req->nccl_progress_st = UCC_OK;
}


static ucc_mpool_ops_t ucc_tl_nccl_req_mpool_ops = {
    .chunk_alloc   = ucc_mpool_hugetlb_malloc,
    .chunk_release = ucc_mpool_hugetlb_free,
    .obj_init      = ucc_tl_nccl_req_mpool_obj_init,
    .obj_cleanup   = ucc_tl_nccl_req_mpool_obj_cleanup
};

static ucc_status_t ucc_tl_nccl_req_mapped_mpool_chunk_malloc(ucc_mpool_t *mp,
                                                              size_t *size_p,
                                                              void ** chunk_p)
{
    cudaError_t cu_st;

    cu_st = cudaHostAlloc((void**)chunk_p, *size_p, cudaHostAllocMapped);
    if (cu_st != cudaSuccess) {
        return UCC_ERR_NO_MEMORY;
    }

    return UCC_OK;
}

static void ucc_tl_nccl_req_mapped_mpool_chunk_free(ucc_mpool_t *mp,
                                                    void *chunk)
{
    cudaFreeHost(chunk);
}

static void ucc_tl_nccl_req_mapped_mpool_obj_init(ucc_mpool_t *mp, void *obj,
                                                  void *chunk)
{
    ucc_tl_nccl_task_t *req = (ucc_tl_nccl_task_t*) obj;
    cudaError_t st;
    st = cudaHostGetDevicePointer((void **)(&req->dev_status),
                                  (void *)&req->host_status, 0);
    if (st != cudaSuccess) {
        req->super.status = UCC_ERR_NO_MESSAGE;
    }
    ucc_coll_task_construct(&req->super);
    req->super.progress   = ucc_tl_nccl_driver_collective_progress;
    req->nccl_progress_st = UCC_OK;
}

static ucc_mpool_ops_t ucc_tl_nccl_req_mapped_mpool_ops = {
    .chunk_alloc   = ucc_tl_nccl_req_mapped_mpool_chunk_malloc,
    .chunk_release = ucc_tl_nccl_req_mapped_mpool_chunk_free,
    .obj_init      = ucc_tl_nccl_req_mapped_mpool_obj_init,
    .obj_cleanup   = ucc_tl_nccl_req_mpool_obj_cleanup
};

UCC_CLASS_INIT_FUNC(ucc_tl_nccl_context_t,
                    const ucc_base_context_params_t *params,
                    const ucc_base_config_t *config)
{
    ucc_tl_nccl_context_config_t *tl_nccl_config =
        ucc_derived_of(config, ucc_tl_nccl_context_config_t);
    int mem_ops_attr = 0;
    ucc_status_t status;

    UCC_CLASS_CALL_SUPER_INIT(ucc_tl_context_t, &tl_nccl_config->super,
                              params->context);
    memcpy(&self->cfg, tl_nccl_config, sizeof(*tl_nccl_config));
    if (self->cfg.sync_type != UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT) {
#if CUDA_VERSION < 12000
        CUresult cu_st;
        CUdevice cu_dev;
        cu_st = cuCtxGetDevice(&cu_dev);
        if (cu_st == CUDA_SUCCESS) {
            cu_st = cuDeviceGetAttribute(&mem_ops_attr,
                                        CU_DEVICE_ATTRIBUTE_CAN_USE_STREAM_MEM_OPS,
                                        cu_dev);
        } else {
            tl_debug(self->super.super.lib, "failed to get cuda device");
        }
#else
        mem_ops_attr = 1;
#endif
        if (mem_ops_attr == 0) {
            if (self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
                tl_error(self->super.super.lib, "memops not supported");
                return UCC_ERR_NOT_SUPPORTED;
            }
            tl_debug(self->super.super.lib, "fallback to event completion sync");
            self->cfg.sync_type = UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT;
        } else {
            self->cfg.sync_type = UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS;
        }
    }
    ucc_assert(self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS ||
               self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_EVENT);
    if (self->cfg.sync_type == UCC_TL_NCCL_COMPLETION_SYNC_TYPE_MEMOPS) {
        tl_debug(self->super.super.lib, "using memops completion sync");
        status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_nccl_task_t), 0,
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                                &ucc_tl_nccl_req_mapped_mpool_ops,
                                params->thread_mode, "tl_nccl_req_mp");
    } else {
        tl_debug(self->super.super.lib, "using event completion sync");
        status = ucc_mpool_init(&self->req_mp, 0, sizeof(ucc_tl_nccl_task_t), 0,
                                UCC_CACHE_LINE_SIZE, 8, UINT_MAX,
                                &ucc_tl_nccl_req_mpool_ops, params->thread_mode,
                                "tl_nccl_req_mp");
    }

    if (status != UCC_OK) {
        tl_error(self->super.super.lib,
                 "failed to initialize tl_nccl_req mpool");
        return status;
    }
    // scratch buffer for barrier
    cudaError_t cuda_st = cudaMalloc(&self->scratch_buf, sizeof(float));
    if (cuda_st != cudaSuccess) {
        return UCC_ERR_NO_MEMORY;
    }

    /* Initialize UBR support */
    self->ubr_available = 0;

#if NCCL_HAS_UBR
    /* Check if UBR should be enabled based on config and NCCL version */
    if (self->cfg.enable_ubr != UCC_NO) {
        self->ubr_available = 1;
        tl_debug(
            self->super.super.lib,
            "NCCL User Buffer Registration available (NCCL %d.%d.%d), using "
            "lazy registration",
            NCCL_MAJOR,
            NCCL_MINOR,
            NCCL_PATCH);
    } else {
        tl_debug(
            self->super.super.lib,
            "NCCL User Buffer Registration disabled by config");
    }
#else
    if (self->cfg.enable_ubr == UCC_YES) {
        tl_error(
            self->super.super.lib,
            "NCCL User Buffer Registration requested but NCCL version %d.%d.%d "
            "< 2.19.0",
            NCCL_MAJOR,
            NCCL_MINOR,
            NCCL_PATCH);
        cudaFree(self->scratch_buf);
        ucc_mpool_cleanup(&self->req_mp, 1);
        return UCC_ERR_NOT_SUPPORTED;
    }
    tl_debug(
        self->super.super.lib,
        "NCCL User Buffer Registration not available (NCCL %d.%d.%d < 2.19.0)",
        NCCL_MAJOR,
        NCCL_MINOR,
        NCCL_PATCH);
#endif

    tl_debug(self->super.super.lib, "initialized tl context: %p", self);
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_nccl_context_t)
{
    tl_debug(self->super.super.lib, "finalizing tl context: %p", self);
    ucc_mpool_cleanup(&self->req_mp, 1);
    cudaFree(self->scratch_buf);
    self->scratch_buf = NULL;
}

UCC_CLASS_DEFINE(ucc_tl_nccl_context_t, ucc_tl_context_t);

ucc_status_t
ucc_tl_nccl_get_context_attr(const ucc_base_context_t *context, /* NOLINT */
                             ucc_base_ctx_attr_t      *attr)
{
    ucc_base_ctx_attr_clear(attr);
    return UCC_OK;
}

/* Map memory for NCCL User Buffer Registration (UBR).
 * 
 * This function creates a memory handle for UBR but does not immediately register
 * the buffer with NCCL communicators. The actual registration happens lazily when
 * the buffer is first used in a collective operation.
 * 
 * Requirements:
 * - NCCL >= 2.19.0 compiled with UBR support
 * - UCC_TL_NCCL_ENABLE_UBR must be enabled (default: try)
 * - Buffer must be CUDA device memory (cudaMalloc, cudaMallocManaged, etc.)
 * - Buffer must have non-zero length
 * 
 * Lifecycle:
 * - mem_map() creates handle (no NCCL registration)
 * - team_create() initializes NCCL communicators (no registration)
 * - collective_init() triggers lazy registration via ncclCommRegister()
 * - mem_unmap() deregisters via ncclCommDeregister()
 * 
 * Best Practice: Call mem_unmap() BEFORE team_destroy() for clean deregistration.
 * However, if team_destroy() is called first, NCCL automatically cleans up all
 * registrations, so no resource leak occurs (just a debug message in mem_unmap)
 */
ucc_status_t ucc_tl_nccl_mem_map(
    const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
    ucc_mem_map_memh_t *memh, ucc_mem_map_tl_t *tl_h)
{
    ucc_tl_nccl_context_t *ctx = ucc_derived_of(context, ucc_tl_nccl_context_t);
    ucc_tl_nccl_memh_data_t *m_data;

    /* Check if UBR is available and enabled */
    if (!ctx->ubr_available) {
        tl_debug(
            ctx->super.super.lib, "NCCL UBR not available, skipping mem_map");
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* Support both EXPORT and IMPORT modes for global memh */
    if (mode != UCC_MEM_MAP_MODE_EXPORT && mode != UCC_MEM_MAP_MODE_IMPORT) {
        tl_debug(ctx->super.super.lib,
                 "NCCL UBR: unsupported mode %d", mode);
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* Reject zero-length buffers */
    if (memh->len == 0) {
        tl_debug(ctx->super.super.lib,
                 "NCCL UBR: zero-length buffer, skipping mem_map");
        return UCC_ERR_NOT_SUPPORTED;
    }

    /* Allocate TL-specific memory handle data */
    m_data = (ucc_tl_nccl_memh_data_t *)ucc_calloc(
        1, sizeof(ucc_tl_nccl_memh_data_t), "tl_nccl_memh_data");
    if (!m_data) {
        tl_error(
            ctx->super.super.lib, "failed to allocate TL memory handle data");
        return UCC_ERR_NO_MEMORY;
    }

    /* For NCCL UBR, we only store metadata (address/length) for lazy registration.
     * When ncclCommRegister is called later, it stores this metadata locally.
     * The NCCL communicator handles IPC handle exchange internally during collective
     * operations (via point-to-point proxy calls), so we don't need special IMPORT
     * handling. We can use memh->address/memh->len directly in both EXPORT and IMPORT
     * modes - the address should be valid in the current process context. */
    m_data->address          = memh->address;
    m_data->length           = memh->len;
    m_data->registered_comms = NULL;
    m_data->nccl_handles     = NULL;
    m_data->num_comms        = 0;
    m_data->max_comms        = 0;

    /* Set TL handle data */
    tl_h->tl_data = m_data;
    strncpy(tl_h->tl_name, "nccl", UCC_MEM_MAP_TL_NAME_LEN - 1);
    tl_h->tl_name[UCC_MEM_MAP_TL_NAME_LEN - 1] = '\0';

    tl_debug(ctx->super.super.lib,
             "NCCL UBR: %s memh for buffer %p, size %zu (lazy registration)",
             mode == UCC_MEM_MAP_MODE_EXPORT ? "created" : "imported",
             memh->address, memh->len);

    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_mem_unmap(
    const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
    ucc_mem_map_tl_t *tl_h)
{
    ucc_tl_nccl_context_t *ctx = ucc_derived_of(context, ucc_tl_nccl_context_t);
    ucc_tl_nccl_memh_data_t *m_data;
#if NCCL_HAS_UBR
    ncclResult_t nccl_status;
    int          i;
#endif

    if (!tl_h || !tl_h->tl_data) {
        return UCC_OK;
    }

    m_data = (ucc_tl_nccl_memh_data_t *)tl_h->tl_data;

#if NCCL_HAS_UBR
    /* Deregister from all NCCL communicators this buffer was registered with
     * 
     * LIFECYCLE NOTE: Best practice is to call ucc_mem_unmap() BEFORE 
     * ucc_team_destroy() to ensure clean deregistration. However, if a team 
     * is destroyed first, NCCL automatically cleans up all buffer registrations
     * during ncclCommDestroy() via ncclRegCleanup(). In this case:
     * - ncclCommDeregister() will return ncclInvalidArgument (comm already freed)
     * - No resource leak occurs (NCCL already cleaned up)
     * - We log a debug message and continue
     * 
     * We cannot detect dangling comm pointers before calling ncclCommDeregister()
     * because we have no notification when teams are destroyed. We rely on NCCL's
     * CommCheck() to safely detect invalid communicators.
     */
    for (i = 0; i < m_data->num_comms; i++) {
        nccl_status = ncclCommDeregister(
            m_data->registered_comms[i], m_data->nccl_handles[i]);
        if (nccl_status == ncclSuccess) {
            tl_debug(
                ctx->super.super.lib,
                "NCCL UBR: deregistered buffer %p from comm %p",
                m_data->address,
                m_data->registered_comms[i]);
        } else if (nccl_status == ncclInvalidArgument) {
            /* Comm was likely already destroyed - NCCL auto-cleaned the registration.
             * This can happen if team was destroyed before mem_unmap was called.
             * No resource leak occurs as NCCL's ncclRegCleanup() already freed everything. */
            tl_debug(
                ctx->super.super.lib,
                "NCCL UBR: comm %p already destroyed, buffer %p was auto-cleaned by NCCL",
                m_data->registered_comms[i],
                m_data->address);
        } else {
            /* Unexpected error */
            tl_warn(
                ctx->super.super.lib,
                "NCCL UBR: failed to deregister buffer %p from comm %p: %s",
                m_data->address,
                m_data->registered_comms[i],
                ncclGetErrorString(nccl_status));
        }
    }

    /* Free the arrays */
    if (m_data->registered_comms) {
        ucc_free(m_data->registered_comms);
    }
    if (m_data->nccl_handles) {
        ucc_free(m_data->nccl_handles);
    }
#endif

    /* Free the TL data */
    ucc_free(m_data);
    tl_h->tl_data = NULL;

    tl_debug(ctx->super.super.lib, "NCCL UBR: unmapped buffer");
    return UCC_OK;
}

ucc_status_t ucc_tl_nccl_memh_pack(
    const ucc_base_context_t *context, ucc_mem_map_mode_t mode,
    ucc_mem_map_tl_t *tl_h, void **pack_buffer)
{
    ucc_tl_nccl_context_t   *ctx = ucc_derived_of(context, ucc_tl_nccl_context_t);
    ucc_tl_nccl_memh_data_t *m_data;
    void                    *packed;

    /* If tl_h is NULL, return early */
    if (!tl_h) {
        *pack_buffer = NULL;
        return UCC_OK;
    }

    /* If UBR is not available/disabled or no TL data, return empty pack */
    if (!ctx->ubr_available || !tl_h->tl_data) {
        tl_h->packed_size = 0;
        *pack_buffer      = NULL;
        return UCC_OK;
    }

    m_data = (ucc_tl_nccl_memh_data_t *)tl_h->tl_data;

    /* Pack minimal data (address + length) so this TL is included in the memh.
     * The core filters out TLs with packed_size == 0, so we must pack something.
     * Actual NCCL registration happens lazily when buffer is first used. */
    tl_h->packed_size = sizeof(void *) + sizeof(size_t);
    packed = ucc_malloc(tl_h->packed_size, "nccl_memh_pack");
    if (!packed) {
        tl_error(ctx->super.super.lib,
                 "failed to allocate pack buffer");
        return UCC_ERR_NO_MEMORY;
    }

    memcpy(packed, &m_data->address, sizeof(void *));
    memcpy((char *)packed + sizeof(void *), &m_data->length, sizeof(size_t));
    *pack_buffer = packed;

    tl_debug(ctx->super.super.lib,
             "NCCL UBR: packed memh for buffer %p, size %zu (lazy registration)",
             m_data->address, m_data->length);

    return UCC_OK;
}
