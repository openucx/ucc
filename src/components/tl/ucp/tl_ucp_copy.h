/**
 * Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "tl_ucp.h"
#include "tl_ucp_coll.h"

#define COPY_TASK_TEST(_phase, _ctx, _errmsg, _ctask) do {                     \
    if (_ctask != NULL) {                                                      \
        status = _ctx->copy.test(ctx, _ctask);                                 \
        if (status > 0) {                                                      \
            task->super.status = UCC_INPROGRESS;                               \
            SAVE_STATE(_phase);                                                \
            return;                                                            \
        }                                                                      \
        _ctx->copy.finalize(_ctask);                                           \
        _ctask = NULL;                                                         \
        if (ucc_unlikely(status < 0)) {                                        \
            tl_error(UCC_TASK_LIB(task), _errmsg);                             \
            task->super.status = status;                                       \
            return;                                                            \
        }                                                                      \
    }                                                                          \
} while(0)

typedef union ucc_tl_ucp_copy_task {
    ucc_ee_executor_task_t ee_task;
    ucs_status_ptr_t       ucp_task;
} ucc_tl_ucp_copy_task_t;

/* copies based on MC */
ucc_status_t ucc_tl_ucp_mc_copy_post(void *dst, ucc_memory_type_t dst_mtype,
                                     ucp_mem_h dst_memh,
                                     void *src, ucc_memory_type_t src_mtype,
                                     ucp_mem_h src_memh,
                                     size_t size,
                                     ucc_tl_ucp_task_t *coll_task,
                                     ucc_tl_ucp_copy_task_t **copy_task);

ucc_status_t ucc_tl_ucp_mc_copy_test(ucc_tl_ucp_context_t *ctx,
                                     ucc_tl_ucp_copy_task_t *copy_task);

ucc_status_t ucc_tl_ucp_mc_copy_finalize(ucc_tl_ucp_copy_task_t *copy_task);

/* copies based on EC */
ucc_status_t ucc_tl_ucp_ec_copy_post(void *dst, ucc_memory_type_t dst_mtype,
                                     ucp_mem_h dst_memh,
                                     void *src, ucc_memory_type_t src_mtype,
                                     ucp_mem_h src_memh,
                                     size_t size,
                                     ucc_tl_ucp_task_t *coll_task,
                                     ucc_tl_ucp_copy_task_t **copy_task);

ucc_status_t ucc_tl_ucp_ec_copy_test(ucc_tl_ucp_context_t *ctx,
                                     ucc_tl_ucp_copy_task_t *copy_task);

ucc_status_t ucc_tl_ucp_ec_copy_finalize(ucc_tl_ucp_copy_task_t *copy_task);

/* copies based on UCX */
ucc_status_t ucc_tl_ucp_ucp_copy_post(void *dst, ucc_memory_type_t dst_mtype,
                                     ucp_mem_h dst_memh,
                                     void *src, ucc_memory_type_t src_mtype,
                                     ucp_mem_h src_memh,
                                     size_t size,
                                     ucc_tl_ucp_task_t *coll_task,
                                     ucc_tl_ucp_copy_task_t **copy_task);

ucc_status_t ucc_tl_ucp_ucp_copy_test(ucc_tl_ucp_context_t *ctx,
                                      ucc_tl_ucp_copy_task_t *copy_task);

ucc_status_t ucc_tl_ucp_ucp_copy_finalize(ucc_tl_ucp_copy_task_t *copy_task);
