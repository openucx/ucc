/**
 * Copyright (C) Mellanox Technologies Ltd. 2020-2021.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_MC_BASE_H_
#define UCC_MC_BASE_H_

#include "config.h"
#include "utils/ucc_component.h"
#include "utils/ucc_class.h"
#include "utils/ucc_parser.h"
#include "core/ucc_global_opts.h"

/**
 * UCC memory attributes field mask
 */
typedef enum ucc_mem_attr_field {
    UCC_MEM_ATTR_FIELD_MEM_TYPE     = UCC_BIT(0), /**< Indicate if memory type
                                                       is populated. E.g. CPU/GPU */
    UCC_MEM_ATTR_FIELD_BASE_ADDRESS = UCC_BIT(2), /**< Request base address of the
                                                       allocation */
    UCC_MEM_ATTR_FIELD_ALLOC_LENGTH = UCC_BIT(3)  /**< Request the whole length of the
                                                       allocation */
} ucc_mem_attr_field_t;


typedef struct ucc_mem_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucc_mem_attr_field_t.
     */
    uint64_t          field_mask;

    /**
     * The type of memory. E.g. CPU/GPU memory or some other valid type
     */
    ucc_memory_type_t mem_type;

    /**
     * Base address of the allocation to which the provided buffer belongs to.
     * If the mc not support base address query, then the pointer passed to
     * ucc_mc_cuda_mem_query is returned as is.
     */
    void              *base_address;

    /**
     * Length of the whole allocation to which the provided buffer belongs to.
     * If the md not support querying allocation length, then the length passed
     * to ucc_mc_cuda_mem_query is returned as is.
     */
    size_t            alloc_length;

} ucc_mem_attr_t;


/**
 * Array of string names for each memory type
 */
extern const char *ucc_memory_type_names[];

typedef struct ucc_mc_config {
    ucc_log_component_config_t log_component;
} ucc_mc_config_t;
extern ucc_config_field_t ucc_mc_config_table[];

typedef struct ucc_mc_ops {
    ucc_status_t (*mem_query)(const void *ptr, size_t length,
                              ucc_mem_attr_t *mem_attr);
    ucc_status_t (*mem_alloc)(void **ptr, size_t size);
    ucc_status_t (*mem_free)(void *ptr);
    ucc_status_t (*reduce)(const void *src1, const void *src2,
                           void *dst, size_t count, ucc_datatype_t dt,
                           ucc_reduction_op_t op);
    ucc_status_t (*memcpy)(void *dst, const void *src, size_t len,
                           ucc_memory_type_t dst_mem,
                           ucc_memory_type_t src_mem);
 } ucc_mc_ops_t;

typedef struct ucc_mc_base {
    ucc_component_iface_t           super;
    uint32_t                        ref_cnt;
    ucc_memory_type_t               type;
    ucc_mc_config_t                *config;
    ucc_config_global_list_entry_t  config_table;
    ucc_status_t                   (*init)();
    ucc_status_t                   (*finalize)();
    const ucc_mc_ops_t              ops;
} ucc_mc_base_t;

#endif
