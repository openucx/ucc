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

typedef enum ucc_memory_type {
    UCC_MEMORY_TYPE_HOST,         /**< Default system memory */
    UCC_MEMORY_TYPE_CUDA,         /**< NVIDIA CUDA memory */
    UCC_MEMORY_TYPE_CUDA_MANAGED, /**< NVIDIA CUDA managed memory */
    UCC_MEMORY_TYPE_ROCM,         /**< AMD ROCM memory */
    UCC_MEMORY_TYPE_ROCM_MANAGED, /**< AMD ROCM managed system memory */
    UCC_MEMORY_TYPE_LAST,
    UCC_MEMORY_TYPE_UNKNOWN = UCC_MEMORY_TYPE_LAST
} ucc_memory_type_t;

/**
 * Array of string names for each memory type
 */
extern const char *ucc_memory_type_names[];

typedef struct ucc_mc_config {
    ucc_log_component_config_t log_component;
} ucc_mc_config_t;
extern ucc_config_field_t ucc_mc_config_table[];

typedef struct ucc_mc_ops {
    ucc_status_t (*mem_type)(const void *ptr, ucc_memory_type_t *mem_type);
    ucc_status_t (*mem_alloc)(void **ptr, size_t size);
    ucc_status_t (*mem_free)(void *ptr);
    ucc_status_t (*reduce)(const void *src1, const void *src2, void *dst,
                           size_t count, ucc_datatype_t dt,
                           ucc_reduction_op_t op);
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
