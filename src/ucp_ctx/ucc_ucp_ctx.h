/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#ifndef UCC_UCP_CTX_H_
#define UCC_UCP_CTX_H_

#include "config.h"
#include "utils/ucc_parser.h"
#include "utils/ucc_compiler_def.h"
#include "utils/ucc_component.h"
#include "ucc_ucp_tag.h"

typedef enum {
    UCC_UCP_CTX_NOT_REQUIRED = 0,
    UCC_UCP_CTX_REQUIRED,
} ucc_ucp_ctx_requirement_t;

typedef struct ucc_ucp_req {
    ucc_status_t status;
} ucc_ucp_req_t;

typedef struct ucc_ucp_ctx ucc_ucp_ctx_t;

enum ucc_ucp_ctx_param_field {
    UCC_UCP_CTX_PARAM_FIELD_DEVICES           = UCC_BIT(0),
    UCC_UCP_CTX_PARAM_FIELD_THREAD_MODE       = UCC_BIT(1),
    UCC_UCP_CTX_PARAM_FIELD_ESTIMATED_NUM_PPN = UCC_BIT(2),
    UCC_UCP_CTX_PARAM_FIELD_ESTIMATED_NUM_EPS = UCC_BIT(3),
    UCC_UCP_CTX_PARAM_FIELD_PREFIX            = UCC_BIT(4),
};

typedef struct ucc_ucp_ctx_create_params {
    uint64_t          mask;
    char             *devices;
    char             *prefix;
    ucc_thread_mode_t thread_mode;
    int               estimated_num_ppn;
    int               estimated_num_eps;
} ucc_ucp_ctx_create_params_t;

typedef struct ucc_ucp_ctx {
    void *ucp_context;
    void *ucp_worker;
} ucc_ucp_ctx_t;

typedef struct ucc_ucp_ctx_handle ucc_ucp_ctx_handle_t;

/* This is the interface for the dynamically loadable component
   that holds the shared UCP context/worker. It is compiled into
   stand alone component in order to avoid linking libucc against
   libucp.

   The component has just 2 APIs used to create a ucc_ucp_context
   handle. There is 1-1 relation between ucc_context_t and
   ucc_ucp_ctx_handle_t (ucc_context_t might not have the
   ucc_ucp_ctx at all if it is not required by any other component).
   Each independent component can than obtain an instance of ucc_ucp_ctx_t
   (which is a pair of ucp_context/worker) via ucc_ucp_ctx_get/put.

   Note, each component that wants to use Shared UCP CTX has to register
   itself into the corresponding enum in ucc_ucp_tag.h in order to
   guarantee the tag space sharing. */
typedef struct ucc_ucp_ctx_iface {
    ucc_component_iface_t super;
    ucc_status_t (*create)(const ucc_ucp_ctx_create_params_t *params,
                           ucc_ucp_ctx_handle_t **ctx_handle);
    ucc_status_t (*destroy)(ucc_ucp_ctx_handle_t *ctx_handle);
} ucc_ucp_ctx_iface_t;

ucc_status_t ucc_ucp_ctx_get(ucc_ucp_ctx_handle_t *ctx_handle,
                             ucc_ucp_ctx_t **ctx);
ucc_status_t ucc_ucp_ctx_put(ucc_ucp_ctx_handle_t *ctx_handle);

#endif
