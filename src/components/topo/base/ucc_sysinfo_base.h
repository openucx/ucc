/**
 * Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_SYSINFO_BASE_H_
#define UCC_SYSINFO_BASE_H_

#include "config.h"
#include "utils/ucc_component.h"
#include "utils/ucc_parser.h"

typedef enum ucc_sysinfo_type {
   UCC_SYSINFO_TYPE_NVLINK,
   UCC_SYSINFO_TYPE_IB,
   UCC_SYSINFO_TYPE_LAST
} ucc_sysinfo_type_t;

typedef struct ucc_sysinfo_params {
    char dummy;
} ucc_sysinfo_params_t;

typedef struct ucc_sysinfo_ops {
    ucc_status_t (*get_info)(void **info, int *n_info);
    ucc_status_t (*set_visible_devices)(
        void *info, int n_info, const ucc_config_names_array_t *devices,
        uint32_t *visible_devices);
} ucc_sysinfo_ops_t;

typedef struct ucc_sysinfo_config {
    char dummy;
} ucc_sysinfo_config_t;

extern ucc_config_field_t ucc_sysinfo_config_table[];

typedef struct ucc_sysinfo_base {
    ucc_component_iface_t          super;
    ucc_sysinfo_type_t             type;
    uint32_t                       ref_cnt;
    ucc_sysinfo_config_t          *config;
    ucc_config_global_list_entry_t config_table;
    ucc_status_t (*init)(const ucc_sysinfo_params_t *params);
    ucc_status_t (*finalize)();
    ucc_sysinfo_ops_t ops;
} ucc_sysinfo_base_t;

#endif
