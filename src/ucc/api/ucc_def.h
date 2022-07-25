/**
 * @file ucc_def.h
 * @date 2020
 * @copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * @copyright Copyright (C) Huawei Technologies Co., Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_DEF_H_
#define UCC_DEF_H_

#include <ucc/api/ucc_status.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @ingroup UCC_LIB_INIT_DT
 * @brief UCC library handle
 *
 * The ucc library handle is an opaque handle created by the library. It
 * abstracts the collective library. It holds the global information and
 * resources associated  with the library. The library handle cannot be passed
 * from one library instance to another.
 */
typedef struct ucc_lib_info*       ucc_lib_h;

/**
 * @ingroup UCC_CONTEXT_DT
 * @brief UCC context
 *
 * The UCC context is an opaque handle to abstract the network resources for
 * collective operations. The network resources could be either software or
 * hardware. Based on the type of the context, the resources can be shared or
 * either be exclusively used. The UCC context is required but not sufficient to
 * execute a collective operation.
 */

typedef struct ucc_context*         ucc_context_h;

/**
 * @ingroup UCC_TEAM_DT
 * @brief UCC team handle
 *
 * The UCC team handle is an opaque handle created by the library. It abstracts
 * the group resources required for the collective operations and participants
 * of the collective operation. The participants of the collective operation can
 * be an OS process or thread.
 */

typedef struct ucc_team*            ucc_team_h;

/**
 * @ingroup UCC_COLLECTIVES_DT
 * @brief UCC collective request handle
 *
 * The UCC request handle is an opaque handle created by the library during the
 * invocation of the collective operation. The request may be used to learn the
 * status of the collective operation, progress, or complete the collective
 * operation.
 */
typedef struct ucc_coll_req* ucc_coll_req_h;
typedef struct ucc_coll_req {
    ucc_status_t status;
} ucc_coll_req_t;

/**
 * @ingroup UCC_COLLECTIVES_DT
 * @brief UCC collective completion callback
 *
 * The callback is invoked whenever the collective operation is completed.
 * It is not allowed to call UCC APIs from the completion callback except
 * for @ref ucc_collective_finalize.
 */
typedef struct ucc_coll_callback {
    void (*cb)(void *data, ucc_status_t status);
    void  *data;
} ucc_coll_callback_t;
/**
 * @ingroup UCC_COLLECTIVES
 * @brief UCC memory handle
 *
 * The UCC memory handle is an opaque handle created by the library representing
 * the buffer and address.
 */
typedef struct ucc_mem_handle*      ucc_mem_h;

/**
 * @ingroup UCC_LIB_INIT_DT
 *
 * @brief UCC library configuration handle
 */
typedef struct ucc_lib_config*           ucc_lib_config_h;

/**
 * @ingroup UCC_CONTEXT_DT
 *
 * @brief UCC context configuration handle
 */
typedef struct ucc_context_config*       ucc_context_config_h;


/**
 * @ingroup UCC_COLLECTIVES_DT
 * @brief Count datatype to support both small (32 bit) and large counts (64 bit)
 */
typedef uint64_t ucc_count_t;

/**
 * @ingroup UCC_COLLECTIVES_DT
 * @brief Datatype to support both small (32 bit) and large address offsets (64 bit)
 */
typedef uint64_t ucc_aint_t;

/* Reflects the definition in UCS - The i-th bit */
#define UCC_BIT(i)               (1ul << (i))

#define UCC_MASK(i)              (UCC_BIT(i) - 1)

/* Reflects the definition in UCS */
/**
 * @ingroup UCC_UTILS
 * @brief Print configurations
 */
typedef enum {
    UCC_CONFIG_PRINT_CONFIG        = UCC_BIT(0),
    UCC_CONFIG_PRINT_HEADER        = UCC_BIT(1),
    UCC_CONFIG_PRINT_DOC           = UCC_BIT(2),
    UCC_CONFIG_PRINT_HIDDEN        = UCC_BIT(3)
} ucc_config_print_flags_t;

/**
 * @ingroup UCC_COLLECTIVES_DT
 * @brief Datatype for collective tags
 */
typedef uint16_t ucc_coll_id_t ;

/**
 * @ingroup UCC_TEAM_DT
 */
typedef void *ucc_p2p_conn_t;

/**
 * @ingroup UCC_TEAM_DT
 */
typedef void* ucc_context_addr_h;

/**
 * @ingroup UCC_TEAM_DT
 */
typedef size_t ucc_context_addr_len_t;


/**
 * @ingroup UCC_EVENTS_DT
 * @brief UCC execution engine handle
 *
 * The UCC execution engine handle is an opaque handle created by the library representing
 * the execution context and related queues.
 */
typedef struct ucc_ee*      ucc_ee_h;

typedef struct ucc_dt_generic ucc_dt_generic_t;
/**
 * @ingroup UCC_DATATYPES
 * @brief Helper enum for generic/predefined datatype representation
 *
 */
typedef enum {
    UCC_DATATYPE_PREDEFINED = 0,
    UCC_DATATYPE_GENERIC    = UCC_BIT(0),
    UCC_DATATYPE_SHIFT      = 3,
    UCC_DATATYPE_CLASS_MASK = UCC_MASK(UCC_DATATYPE_SHIFT)
} ucc_dt_type_t;

#define UCC_PREDEFINED_DT(_id) \
    (ucc_datatype_t)((((uint64_t)(_id)) << UCC_DATATYPE_SHIFT) | \
                     (UCC_DATATYPE_PREDEFINED))


#endif
