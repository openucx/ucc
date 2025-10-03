/**
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    UCC_EE_EXECUTOR_PARAM_FIELD_TYPE       = UCC_BIT(0),
    UCC_EE_EXECUTOR_PARAM_FIELD_TASK_TYPES = UCC_BIT(1),
};

typedef enum ucc_ee_executor_task_type {
    UCC_EE_EXECUTOR_TASK_REDUCE           = UCC_BIT(0),
    UCC_EE_EXECUTOR_TASK_REDUCE_STRIDED   = UCC_BIT(1),
    UCC_EE_EXECUTOR_TASK_REDUCE_MULTI_DST = UCC_BIT(2),
    UCC_EE_EXECUTOR_TASK_COPY             = UCC_BIT(3),
    UCC_EE_EXECUTOR_TASK_COPY_MULTI       = UCC_BIT(4),
    UCC_EE_EXECUTOR_TASK_LAST
} ucc_ee_executor_task_type_t;

typedef struct ucc_ee_executor_params {
    uint64_t        mask;
    ucc_ee_type_t   ee_type;
    uint64_t        task_types;
} ucc_ee_executor_params_t;

#define UCC_EE_EXECUTOR_NUM_BUFS 9

/* Maximum number of buffers for UCC_EE_EXECUTOR_TASK_REDUCE_MULTI_DST and
   UCC_EE_EXECUTOR_TASK_COPY_MULTI operations */
#define UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS 7

/* Reduces "n_srcs" buffers (each contains "count" elements of type "dt")
   into "dst" buffer.

   If UCC_EEE_TASK_FLAG_REDUCE_SRCS_EXT flag is not set on task_args
   then the sources are taken from "srcs" and n_srcs can not exceed
   UCC_EE_EXECUTOR_NUM_BUFS.

   If UCC_EEE_TASK_FLAG_REDUCE_SRCS_EXT flag IS set on task_args
   then the sources are taken from "src_ext". In that case it is caller
   responsibility to make sure srcs_ext pointer is valid until task
   is complete.

   If UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA flag is set on task_args
   each element of the result of reduction is multiplied by "alpha" */
typedef struct ucc_eee_task_reduce {
    void *             dst;
    union {
        void * srcs[UCC_EE_EXECUTOR_NUM_BUFS];
        void **srcs_ext;
    };
    size_t   count;
    double   alpha;
    ucc_datatype_t     dt;
    ucc_reduction_op_t op;
    uint16_t n_srcs;
} ucc_eee_task_reduce_t;

/* Performs "n_bufs" independent reductions for corresponding buffers in src1 and
   src2 and saving result into "dst" buffer */
typedef struct ucc_eee_task_reduce_multi_dst {
    void               *dst[UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS];
    void               *src1[UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS];
    void               *src2[UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS];
    size_t              counts[UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS];
    ucc_datatype_t      dt;
    ucc_reduction_op_t  op;
    uint16_t            n_bufs;
}  ucc_eee_task_reduce_multi_dst_t;

/* Reduces "n_srcs2+1" buffers (each contains "count" elements of type "dt")
   into "dst" buffer. The first source buffer is "src1". Other n_src2 source
   buffers are defined as SRC[i] = src2 + stride * i, where stride is defined
   in bytes.

   If UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA flag is set on task_args
   each element of the result of reduction is multiplied by "alpha" */

typedef struct ucc_eee_task_reduce_strided {
    void *             dst;
    void *             src1;
    void *             src2;
    size_t             stride;
    size_t             count;
    double             alpha;
    ucc_datatype_t     dt;
    ucc_reduction_op_t op;
    uint16_t           n_src2;
} ucc_eee_task_reduce_strided_t;

/* Copies len bytes from "src" into "dst" */
typedef struct ucc_eee_task_copy {
    const void *src;
    void       *dst;
    size_t      len;
} ucc_eee_task_copy_t;

enum ucc_eee_task_flags {
    UCC_EEE_TASK_FLAG_REDUCE_WITH_ALPHA = UCC_BIT(0),
    UCC_EEE_TASK_FLAG_REDUCE_SRCS_EXT   = UCC_BIT(1)
};

/* Performs "num_vectors" copies from SRC[i] to DST[i] */
typedef struct ucc_eee_task_copy_multi{
    void   *src[UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS];
    void   *dst[UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS];
    size_t  counts[UCC_EE_EXECUTOR_MULTI_OP_NUM_BUFS];
    size_t  num_vectors;
} ucc_eee_task_copy_multi_t;

typedef struct ucc_ee_executor_task_args {
    uint16_t                     task_type;
    uint16_t                     flags;
    union {
        ucc_eee_task_reduce_t           reduce;
        ucc_eee_task_reduce_strided_t   reduce_strided;
        ucc_eee_task_reduce_multi_dst_t reduce_multi_dst;
        ucc_eee_task_copy_t             copy;
        ucc_eee_task_copy_multi_t       copy_multi;
    };
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
