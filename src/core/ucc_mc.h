/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_H_
#define UCC_MC_H_

#include "ucc/api/ucc.h"
#include "components/mc/base/ucc_mc_base.h"
#include "core/ucc_dt.h"
#include "utils/ucc_math.h"

ucc_status_t ucc_mc_init(const ucc_mc_params_t *mc_params);

ucc_status_t ucc_mc_available(ucc_memory_type_t mem_type);

/**
 * Query for memory attributes.
 * @param [in]        ptr       Memory pointer to query.
 * @param [in,out]    mem_attr  Memory attributes.
 */
ucc_status_t ucc_mc_get_mem_attr(const void *ptr, ucc_mem_attr_t *mem_attr);

ucc_status_t ucc_mc_alloc(ucc_mc_buffer_header_t **h_ptr, size_t len,
                          ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_free(ucc_mc_buffer_header_t *h_ptr);

ucc_status_t ucc_mc_flush(ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_finalize();

ucc_status_t ucc_mc_ee_task_post(void *ee_context, ucc_ee_type_t ee_type,
                                 void **ee_task);

ucc_status_t ucc_mc_ee_task_query(void *ee_task, ucc_ee_type_t ee_type);

ucc_status_t ucc_mc_ee_task_end(void *ee_task, ucc_ee_type_t ee_type);

ucc_status_t ucc_mc_ee_create_event(void **event, ucc_ee_type_t ee_type);

ucc_status_t ucc_mc_ee_destroy_event(void *event, ucc_ee_type_t ee_type);

ucc_status_t ucc_mc_ee_event_post(void *ee_context, void *event,
                                  ucc_ee_type_t ee_type);

ucc_status_t ucc_mc_ee_event_test(void *event, ucc_ee_type_t ee_type);


ucc_status_t ucc_mc_memcpy(void *dst, const void *src, size_t len,
                           ucc_memory_type_t dst_mem,
                           ucc_memory_type_t src_mem);

/**
 * Performs reduction of two vectors and stores result to dst
 * @param [in]  src1     First vector reduction operand
 * @param [in]  src2     Second vector reduction operand
 * @param [out] dst      dst = src1 (op) src2{0}
 * @param [in]  count    Number of elements in dst
 * @param [in]  dtype    Vectors elements datatype
 * @param [in]  op       Reduction operation
 * @param [in]  mem_type Vectors memory type
 */
ucc_status_t ucc_mc_reduce(const void *src1, const void *src2, void *dst,
                           size_t count, ucc_datatype_t dtype,
                           ucc_reduction_op_t op, ucc_memory_type_t mem_type);

/**
 * Performs reduction of multiple vectors and stores result to dst
 * @param [in]  src1      First vector reduction operand
 * @param [in]  src2      Array of vector reduction operands
 * @param [out] dst       dst = src1 (op) src2{0} (op) src2{1} (op) ...
 *                                               (op) src2{size-1}
 * @param [in]  n_vectors Number of vectors in src2
 * @param [in]  count     Number of elements in dst
 * @param [in]  stride    Offset between vectors in src2
 * @param [in]  dtype     Vectors elements datatype
 * @param [in]  op        Reduction operation
 * @param [in]  mem_type  Vectors memory type
 */
ucc_status_t ucc_mc_reduce_multi(void *src1, void *src2, void *dst,
                                 size_t n_vectors, size_t count, size_t stride,
                                 ucc_datatype_t dtype, ucc_reduction_op_t op,
                                 ucc_memory_type_t mem_type);

static inline ucc_status_t ucc_dt_reduce(const void *src1, const void *src2,
                                         void *dst, size_t count,
                                         ucc_datatype_t dt,
                                         ucc_memory_type_t mem_type,
                                         ucc_coll_args_t *args)
{
    if (!UCC_DT_IS_PREDEFINED(dt)) {
        return UCC_ERR_NOT_SUPPORTED; //TODO
    } else {
        return ucc_mc_reduce(src1, src2, dst, count,
                             dt, args->op, mem_type);
    }
}


static inline ucc_status_t ucc_mc_reduce_userdefined(void *src1, void *src2,
                                                     void *dst, size_t n_vectors,
                                                     size_t count, size_t stride,
                                                     ucc_dt_generic_t *dt)
{
    int i;

    dt->ops.reduce.cb(src1, src2, dst, count, dt->ops.reduce.ctx);
    for (i = 1; i < n_vectors; i++) {
        dt->ops.reduce.cb(PTR_OFFSET(src2, stride*i), dst, dst, count,
                           dt->ops.reduce.ctx);
    }
    return UCC_OK;
}

static inline ucc_status_t ucc_dt_reduce_multi(void *src1, void *src2,
                                               void *dst, size_t n_vectors,
                                               size_t count, size_t stride,
                                               ucc_datatype_t dt,
                                               ucc_memory_type_t mem_type,
                                               ucc_coll_args_t *args)
{
    if (!UCC_DT_IS_PREDEFINED(dt)) {
        ucc_assert(UCC_DT_HAS_REDUCE(dt));
        return ucc_mc_reduce_userdefined(src1, src2, dst, n_vectors, count,
                                         stride, ucc_dt_to_generic(dt));
    } else {
        return ucc_mc_reduce_multi(src1, src2, dst, n_vectors, count, stride,
                                   dt, args->op, mem_type);
    }
}

#endif
