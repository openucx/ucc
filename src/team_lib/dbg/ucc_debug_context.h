/*
 * Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */
#ifndef UCC_UCX_CONTEXT_H_
#define UCC_UCX_CONTEXT_H_
#include "ucc_tl_debug.h"

typedef struct ucc_tl_debug_context {
    ucc_tl_context_t     super;
} ucc_tl_debug_context_t;

ucc_status_t ucc_debug_context_create(ucc_team_lib_t *tl_lib,
                                      const ucc_context_params_t *params,
                                      const ucc_tl_context_config_t *config,
                                      ucc_tl_context_t **tl_context);
void ucc_debug_context_destroy(ucc_tl_context_t *context);
#endif
