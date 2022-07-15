/**
 * Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    UCC_EC_ATTR_FIELD_THREAD_MODE        = UCC_BIT(0),
    UCC_EC_ATTR_FILED_MAX_EXECUTORS_BUFS = UCC_BIT(1)
}  ucc_ec_attr_field_t;

typedef struct ucc_ec_attr {
    /**
     * Mask of valid fields in this structure, using bits from
     * @ref ucc_ec_attr_field_t.
     */
    uint64_t          field_mask;
    ucc_thread_mode_t thread_mode;
    /**
     *  Maximum number of buffers in ucc_ee_executor_task_args,
     *  includes src buffer
     */
    int               max_ee_bufs;
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

typedef struct ucc_ee_executor {
    ucc_ee_type_t  ee_type;
    void          *ee_context;
} ucc_ee_executor_t;

enum ucc_ee_executor_params_field {
    UCC_EE_EXECUTOR_PARAM_FIELD_TYPE    = UCC_BIT(0),
};

typedef enum ucc_ee_executor_task_type {
    UCC_EE_EXECUTOR_TASK_TYPE_COPY               = UCC_BIT(0),
    UCC_EE_EXECUTOR_TASK_TYPE_REDUCE             = UCC_BIT(1),
    UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI       = UCC_BIT(2),
    UCC_EE_EXECUTOR_TASK_TYPE_REDUCE_MULTI_ALPHA = UCC_BIT(3),
    UCC_EE_EXECUTOR_TASK_TYPE_COPY_MULTI         = UCC_BIT(4),
} ucc_ee_executor_task_type_t;

typedef struct ucc_ee_executor_params {
    uint64_t        mask;
    ucc_ee_type_t   ee_type;
} ucc_ee_executor_params_t;

#define UCC_EE_EXECUTOR_NUM_BUFS 9
#define UCC_EE_EXECUTOR_NUM_COPY_BUFS 6
/*
 *  buffers[0] - destination
 *  buffers[1] .. buffers[UCC_EE_EXECUTOR_NUM_BUFS - 1] - source
 *  count - number of elements in destination
 *  size - number of operands or step between source buffers in bytes
 *  dt - datatype
 *  op - reduction operation
 */

typedef struct ucc_ee_executor_task_args_copy_multi{
    void   *src[UCC_EE_EXECUTOR_NUM_COPY_BUFS];
    void   *dst[UCC_EE_EXECUTOR_NUM_COPY_BUFS];
    size_t  counts[UCC_EE_EXECUTOR_NUM_COPY_BUFS];
    size_t  num_vectors;
} ucc_ee_executor_task_args_copy_multi_t;

typedef struct ucc_ee_executor_task_args {
    ucc_ee_executor_task_type_t             task_type;
    void                                   *bufs[UCC_EE_EXECUTOR_NUM_BUFS];
    double                                  alpha;
    ucc_count_t                             count;
    size_t                                  stride;
    uint32_t                                size;
    ucc_datatype_t                          dt;
    ucc_reduction_op_t                      op;
    ucc_ee_executor_task_args_copy_multi_t  copy_multi;
} ucc_ee_executor_task_args_t;

typedef struct ucc_ee_executor_task {
    ucc_ee_executor_t           *eee;
    ucc_ee_executor_task_args_t  args;
    ucc_status_t                 status;
} ucc_ee_executor_task_t;

typedef struct ucc_ee_executor_ops {
    ucc_status_t (*init)(const ucc_ee_executor_params_t *params,
                         ucc_ee_executor_t **executor);
    ucc_status_t (*status)(const ucc_ee_executor_t *executor);
    ucc_status_t (*start)(ucc_ee_executor_t *executor,
                          void *ee_context);
    ucc_status_t (*stop)(ucc_ee_executor_t *executor);
    ucc_status_t (*finalize)(ucc_ee_executor_t *executor);
    ucc_status_t (*task_post)(ucc_ee_executor_t *executor,
                              const ucc_ee_executor_task_args_t *task_args,
                              ucc_ee_executor_task_t **task);
    ucc_status_t (*task_test)(const ucc_ee_executor_task_t *task);
    ucc_status_t (*task_finalize)(ucc_ee_executor_task_t *task);
} ucc_ee_executor_ops_t;

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
    ucc_ee_executor_ops_t           executor_ops;
} ucc_ec_base_t;

#endif
