/**
 * Copyright (C) Mellanox Technologies Ltd. 2022.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#ifndef UCC_EC_BASE_H_
#define UCC_EC_BASE_H_

#include "config.h"
#include "utils/ucc_component.h"
#include "utils/ucc_class.h"
#include "utils/ucc_parser.h"

typedef struct ucc_ec_config {
    ucc_log_component_config_t log_component;
} ucc_ec_config_t;
extern ucc_config_field_t ucc_ec_config_table[];

typedef struct ucc_ec_params {
    ucc_thread_mode_t thread_mode;
} ucc_ec_params_t;

/**
 * UCC execution component attributes field mask
 */
typedef enum ucc_ec_attr_field {
    UCC_EC_ATTR_FIELD_THREAD_MODE = UCC_BIT(0)
}  ucc_ec_attr_field_t;

typedef struct ucc_ec_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucc_ec_attr_field_t.
     */
    uint64_t          field_mask;
    ucc_thread_mode_t thread_mode;
} ucc_ec_attr_t;

typedef struct ucc_ec_ops {
    ucc_status_t (*task_post)(void *ee_context, void **ee_req);
    ucc_status_t (*task_query)(void *ee_req);
    ucc_status_t (*task_end)(void *ee_req);
    ucc_status_t (*create_event)(void **event);
    ucc_status_t (*destroy_event)(void *event);
    ucc_status_t (*event_post)(void *ee_context, void *event);
    ucc_status_t (*event_test)(void *event);
} ucc_ec_ops_t;

typedef struct ucc_ec_base {
    ucc_component_iface_t           super;
    uint32_t                        ref_cnt;
    ucc_ee_type_t                   type;
    ucc_ec_config_t                *config;
    ucc_config_global_list_entry_t  config_table;
    ucc_status_t                   (*init)(const ucc_ec_params_t *ec_params);
    ucc_status_t                   (*get_attr)(ucc_ec_attr_t *ec_attr);
    ucc_status_t                   (*finalize)();
    ucc_ec_ops_t                    ops;
} ucc_ec_base_t;

#endif
