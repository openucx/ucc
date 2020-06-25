/**
* Copyright (C) Mellanox Technologies Ltd. 2001-2020.  ALL RIGHTS RESERVED.
*
* See file LICENSE for terms.
*/
#include "config.h"
#include "ucc_tl_basic.h"
#include "ucc_basic_context.h"
#include "ucc_basic_team.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <sys/types.h>
#include <unistd.h>

static ucs_config_field_t ucc_tl_basic_lib_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(ucc_tl_basic_lib_config_t, super),
        UCS_CONFIG_TYPE_TABLE(ucc_tl_lib_config_table)
    },

    {NULL}
};

static ucs_config_field_t ucc_tl_basic_context_config_table[] = {
    {"", "", NULL,
        ucs_offsetof(ucc_tl_basic_context_config_t, super),
        UCS_CONFIG_TYPE_TABLE(ucc_tl_context_config_table)
    },

    {NULL}
};


static ucc_status_t ucc_basic_lib_init(const ucc_lib_params_t *params,
                                       const ucc_lib_config_t *config,
                                       const ucc_tl_lib_config_t *tl_config,
                                       ucc_team_lib_t **tl_lib) {
    ucc_tl_basic_t *tl_basic;
    ucc_status_t status;
    ucc_tl_basic_lib_config_t *cfg = ucs_derived_of(tl_config,
                                                    ucc_tl_basic_lib_config_t);
    tl_basic = (ucc_tl_basic_t*)malloc(sizeof(*tl_basic));
    if (!tl_basic) {
        status = UCC_ERR_NO_MESSAGE;
        goto error;
    }
    tl_basic->super.iface = &ucc_team_lib_basic.super;
    *tl_lib = &tl_basic->super;
    return UCC_OK;

error:
    if (tl_basic) free(tl_basic);
    return status;
}

static void ucc_basic_lib_cleanup(ucc_team_lib_t *tl_lib) {
    ucc_tl_basic_t *tl_basic = ucs_derived_of(tl_lib, ucc_tl_basic_t);
    free(tl_lib);
}

ucc_tl_basic_iface_t ucc_team_lib_basic = {
    .super.name                  = "basic",
    .super.priority              = 10,
    .super.tl_lib_config       = {
        .name                    = "BASIC team library",
        .prefix                  = "TL_BASIC_",
        .table                   = ucc_tl_basic_lib_config_table,
        .size                    = sizeof(ucc_tl_basic_lib_config_t),
    },
    .super.tl_context_config     = {
        .name                    = "BASIC tl context",
        .prefix                  = "TL_BASIC_",
        .table                   = ucc_tl_basic_context_config_table,
        .size                    = sizeof(ucc_tl_basic_context_config_t),
    },
    .super.init                  = ucc_basic_lib_init,
    .super.cleanup               = ucc_basic_lib_cleanup,
    .super.context_create        = ucc_basic_context_create,
    .super.context_destroy       = ucc_basic_context_destroy,
    .super.team_create_post      = ucc_basic_team_create_post,
    .super.team_create_test      = ucc_basic_team_create_test,
    .super.team_destroy          = ucc_basic_team_destroy,
};

