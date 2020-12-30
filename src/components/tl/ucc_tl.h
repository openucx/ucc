/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_TL_H_
#define UCC_TL_H_

#include "components/base/ucc_base_iface.h"

/** TL (transport layer) is an internal interface that provides a basic
    implementation of collectives and p2p primitives. It differs from CL in that
    it focuses on the abstraction over hardware transport rather than the
    programming model. These collectives can leverage either p2p communication
    transport or collective transport. The primary example of p2p communication
    transport is UCX, and the primary example of collective transport is SHARP,
    MCAST, and shared memory.
 */

typedef struct ucc_tl_lib     ucc_tl_lib_t;
typedef struct ucc_tl_iface   ucc_tl_iface_t;
typedef struct ucc_tl_context ucc_tl_context_t;

typedef enum ucc_tl_type {
    UCC_TL_UCP = UCC_BIT(0),
} ucc_tl_type_t;

typedef struct ucc_tl_lib_config {
    ucc_base_config_t  super;
    ucc_tl_iface_t    *iface;
} ucc_tl_lib_config_t;
extern ucc_config_field_t ucc_tl_lib_config_table[];

typedef struct ucc_tl_context_config {
    ucc_base_config_t super;
    ucc_tl_lib_t     *tl_lib;
} ucc_tl_context_config_t;
extern ucc_config_field_t ucc_tl_context_config_table[];

ucc_status_t ucc_tl_context_config_read(ucc_tl_lib_t *tl_lib,
                                        const ucc_context_config_t *config,
                                        ucc_tl_context_config_t **cl_config);

ucc_status_t ucc_tl_lib_config_read(ucc_tl_iface_t *iface, const char *full_prefix,
                                    const ucc_lib_config_t *config,
                                    ucc_tl_lib_config_t **cl_config);

typedef struct ucc_tl_iface {
    ucc_component_iface_t          super;
    ucc_tl_type_t                  type;
    ucs_config_global_list_entry_t tl_lib_config;
    ucs_config_global_list_entry_t tl_context_config;
    ucc_base_lib_iface_t           lib;
    ucc_base_context_iface_t       context;
} ucc_tl_iface_t;

typedef struct ucc_tl_lib {
    ucc_base_lib_t              super;
    ucc_tl_iface_t             *iface;
} ucc_tl_lib_t;
UCC_CLASS_DECLARE(ucc_tl_lib_t, ucc_tl_iface_t *, const ucc_tl_lib_config_t *);

typedef struct ucc_tl_context {
    ucc_base_context_t super;
    int                ref_count;
} ucc_tl_context_t;
UCC_CLASS_DECLARE(ucc_tl_context_t, ucc_tl_lib_t *);

#define UCC_TL_IFACE_DECLARE(_name, _NAME)                                     \
    UCC_BASE_IFACE_DECLARE(TL_, tl_, _name, _NAME)

ucc_status_t ucc_tl_context_get(ucc_context_t *ctx, ucc_tl_type_t type,
                                ucc_tl_context_t **tl_context);
ucc_status_t ucc_tl_context_put(ucc_tl_context_t *tl_context);
#endif
