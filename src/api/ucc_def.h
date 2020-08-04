/**
 * @file ucc_def.h
 * @date 2020
 * @copyright Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */
#ifndef UCC_DEF_H_
#define UCC_DEF_H_

#include <api/ucc_status.h>
#include <stddef.h>
#include <stdint.h>

/** 
 * @ingroup UCC_LIBRARY
 * @brief UCC library handle
 * 
 * The ucc library handle is an opaque handle created by the library. It
 * abstracts the collective library. It holds the global information and
 * resources associated  with the library. The library handle cannot be passed
 * from one library instance to another.
 */
 
typedef struct ucc_lib_info_t       ucc_lib_h; 

/**
 * @ingroup UCC_CONTEXT
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
 * @ingroup UCC_TEAM
 * @brief UCC team handle
 * 
 * The UCC team handle is an opaque handle created by the library. It abstracts
 * the group resources required for the collective operations and participants
 * of the collective operation. The participants of the collective operation can
 * be an OS process or thread.
 */

typedef struct ucc_team*            ucc_team_h;

/**
 * @ingroup UCC_COLLECTIVE
 * @brief UCC collective request handle
 * 
 * The UCC request handle is an opaque handle created by the library during the
 * invocation of the collective operation. The request may be used to learn the
 * status of the collective operation, progress, or complete the collective
 * operation.
 */
 
typedef struct ucc_coll_req*        ucc_coll_req_h;

/**
 * @ingroup UCC_COLLECTIVES
 * @brief UCC memory handle
 * 
 * The UCC memory handle is an opaque handle created by the library representing
 * the buffer and address. 
 */
typedef struct ucc_mem_handle*      ucc_mem_h;

typedef struct ucc_lib_config           ucc_lib_config_t;

typedef struct ucc_context_config       ucc_context_config_t;

typedef struct ucc_config_print_flags   ucc_config_print_flags_t;

typedef struct ucc_context_oob_coll {
    int             (*allgather)(void *src_buf, void *recv_buf, size_t size,
                                 void *allgather_info,  void **request);
    ucc_status_t    (*req_test)(void *request);
    ucc_status_t    (*req_free)(void *request);
    uint32_t 	     participants;
    void             *coll_info;
}  ucc_context_oob_coll_t;

typedef uint64_t ucc_count_t;

typedef uint64_t ucc_aint_t;

/* The i-th bit */
#define UCC_BIT(i)               (1ul << (i))

#endif
























