/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "ucc_tl_debug.h"
#include "ucc_debug_context.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

static ucs_config_field_t ucc_tl_debug_lib_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(ucc_tl_debug_lib_config_t, super),
        UCS_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)
    },

    {NULL}
};

static ucs_config_field_t ucc_tl_debug_context_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(ucc_tl_debug_context_config_t, super),
        UCS_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)
    },

    {NULL}
};


static ucc_status_t ucc_debug_lib_init(const ucc_lib_params_t *params,
                                       const ucc_lib_config_t *config,
                                       const ucc_tl_lib_config_t *tl_config,
                                       ucc_team_lib_t **tl_lib) {
    ucc_tl_debug_t *tl_debug;
    ucc_status_t status;
    tl_debug = (ucc_tl_debug_t*)malloc(sizeof(*tl_debug));
    if (!tl_debug) {
        status = UCC_ERR_NO_MESSAGE;
        goto error;
    }
    tl_debug->super.iface = &ucc_team_lib_debug.super;
    *tl_lib = &tl_debug->super;
    return UCC_OK;

error:
    if (tl_debug) free(tl_debug);
    return status;
}

static void ucc_debug_lib_cleanup(ucc_team_lib_t *tl_lib) {
    ucc_tl_debug_t *tl_debug = ucs_derived_of(tl_lib, ucc_tl_debug_t);
    free(tl_lib);
}

ucc_tl_debug_iface_t ucc_team_lib_debug = {
    .super.name                  = "debug",
    .super.priority              = 1,
    .super.tl_lib_config       = {
        .name                    = "DEBUG team library",
        .prefix                  = "TL_DEBUG_",
        .table                   = ucc_tl_debug_lib_config_table,
        .size                    = sizeof(ucc_tl_debug_lib_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "DEBUG tl context",
        .prefix                  = "TL_DEBUG_",
        .table                   = ucc_tl_debug_context_config_table,
        .size                    = sizeof(ucc_tl_debug_context_config_t),
    },
    .super.init                  = ucc_debug_lib_init,
    .super.cleanup               = ucc_debug_lib_cleanup,
    .super.context_create        = ucc_debug_context_create,
    .super.context_destroy       = ucc_debug_context_destroy,
};

