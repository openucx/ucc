/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"

#ifndef UCC_CL_H_
#define UCC_CL_H_

#include "components/base/ucc_base_iface.h"
#include "ucc_cl_type.h"

/** CL (collective layer) is an internal collective interface reflecting the
    public UCC API and extensions to support modularity, the composition of
    multiple collective implementations, and functionality that bridges the
    gap between hardware implementation of communication primitives and the
    programming models.

    The CL layer will build upon TL for the communication transport requirements.
    The CL can include a basic implementation, which provides minimal
    functionality over the TL, or can provide more optimized implementation such
    as hierarchical implementation that leverages multiple TL components.

    The different implementations of CL are realized as different CL components.
    The CL components are loaded dynamically, and their names should match the
    predefined pattern “ucc_cl_.so”. The CL that is used for a given application
    invocation can be selected with the UCC_CLS lib parameter.
*/

typedef struct ucc_cl_lib     ucc_cl_lib_t;
typedef struct ucc_cl_iface   ucc_cl_iface_t;
typedef struct ucc_cl_context ucc_cl_context_t;
typedef struct ucc_cl_team    ucc_cl_team_t;

typedef struct ucc_cl_lib_config {
    ucc_base_config_t  super;
    ucc_cl_iface_t    *iface;
    int                priority;
} ucc_cl_lib_config_t;
extern ucc_config_field_t ucc_cl_lib_config_table[];


typedef struct ucc_cl_context_config {
    ucc_base_config_t super;
    ucc_cl_lib_t   *cl_lib;
} ucc_cl_context_config_t;
extern ucc_config_field_t ucc_cl_context_config_table[];

ucc_status_t ucc_cl_context_config_read(ucc_cl_lib_t *cl_lib,
                                        const ucc_context_config_t *config,
                                        ucc_cl_context_config_t **cl_config);

ucc_status_t ucc_cl_lib_config_read(ucc_cl_iface_t *iface,
                                    const char *full_prefix,
                                    ucc_cl_lib_config_t **cl_config);

typedef struct ucc_cl_iface {
    ucc_component_iface_t          super;
    ucc_cl_type_t                  type;
    ucc_lib_attr_t                 attr;
    ucc_config_global_list_entry_t cl_lib_config;
    ucc_config_global_list_entry_t cl_context_config;
    ucc_base_lib_iface_t           lib;
    ucc_base_context_iface_t       context;
    ucc_base_team_iface_t          team;
    ucc_base_coll_iface_t          coll;
} ucc_cl_iface_t;

typedef struct ucc_cl_lib {
    ucc_base_lib_t              super;
    ucc_cl_iface_t             *iface;
    int                         priority;
} ucc_cl_lib_t;
UCC_CLASS_DECLARE(ucc_cl_lib_t, ucc_cl_iface_t *, const ucc_cl_lib_config_t *,
                  int);

typedef struct ucc_cl_context {
    ucc_base_context_t super;
} ucc_cl_context_t;
UCC_CLASS_DECLARE(ucc_cl_context_t, ucc_cl_lib_t *, ucc_context_t *);

typedef struct ucc_cl_team {
    ucc_base_team_t super;
} ucc_cl_team_t;
UCC_CLASS_DECLARE(ucc_cl_team_t, ucc_cl_context_t *);

typedef struct ucc_cl_lib_attr {
    ucc_base_attr_t super;
    uint64_t tls;
} ucc_cl_lib_attr_t;

#define UCC_CL_IFACE_DECLARE(_name, _NAME)                                     \
    UCC_BASE_IFACE_DECLARE(CL_, cl_, _name, _NAME)

#define UCC_CL_CTX_IFACE(_cl_ctx)                                              \
    (ucc_derived_of((_cl_ctx)->super.lib, ucc_cl_lib_t))->iface

#define UCC_CL_TEAM_IFACE(_cl_team)                                            \
    (ucc_derived_of((_cl_team)->super.context->lib, ucc_cl_lib_t))->iface

#endif
