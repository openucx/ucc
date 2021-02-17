/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_H_
#define UCC_MC_H_

#include "ucc/api/ucc.h"
#include "components/mc/base/ucc_mc_base.h"

ucc_status_t ucc_mc_init();

ucc_status_t ucc_mc_available(ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_type(const void *ptr, ucc_memory_type_t *mem_type);

ucc_status_t ucc_mc_query(const void *ptr, size_t length, ucc_mem_attr_t *mem_attr);

ucc_status_t ucc_mc_alloc(void **ptr, size_t len, ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_free(void *ptr, ucc_memory_type_t mem_type);

ucc_status_t ucc_mc_finalize();

ucc_status_t ucc_mc_reduce(const void *src1, const void *src2, void *dst,
                           size_t count, ucc_datatype_t dt,
                           ucc_reduction_op_t op, ucc_memory_type_t mem_type);
ucc_status_t ucc_mc_reduce_multi(void *sbuf1, void *sbuf2, void *rbuf,
                                 size_t count, size_t size, size_t stride,
                                 ucc_datatype_t dtype, ucc_reduction_op_t op,
                                 ucc_memory_type_t mem_type);

#endif
