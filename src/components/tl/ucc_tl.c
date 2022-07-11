/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_tl.h"
#include "utils/ucc_log.h"

ucc_config_field_t ucc_tl_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_base_lib_config_table)},

    {NULL}
};

ucc_config_field_t ucc_tl_context_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_base_ctx_config_table)},

    {NULL}
};

UCC_CLASS_INIT_FUNC(ucc_tl_lib_t, ucc_tl_iface_t *tl_iface,
                    const ucc_tl_lib_config_t *tl_config)
{
    UCC_CLASS_CALL_BASE_INIT();
    self->iface         = tl_iface;
    self->super.log_component = tl_config->super.log_component;
    ucc_strncpy_safe(self->super.log_component.name,
                     tl_iface->tl_lib_config.name,
                     sizeof(self->super.log_component.name));
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_lib_t)
{
}

UCC_CLASS_DEFINE(ucc_tl_lib_t, void);

UCC_CLASS_INIT_FUNC(ucc_tl_context_t, const ucc_tl_context_config_t *tl_config,
                    ucc_context_t *ucc_context)
{
    UCC_CLASS_CALL_BASE_INIT();
    self->super.lib         = &tl_config->tl_lib->super;
    self->super.ucc_context = ucc_context;
    self->ref_count = 0;
    if (0 == strcmp(tl_config->super.score_str, "0")) {
        return UCC_ERR_LAST;
    }
    self->super.score_str = strdup(tl_config->super.score_str);

    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_context_t)
{
    ucc_free(self->super.score_str);
}

UCC_CLASS_DEFINE(ucc_tl_context_t, void);

ucc_status_t ucc_tl_context_config_read(ucc_tl_lib_t *tl_lib,
                                        const ucc_context_config_t *config,
                                        ucc_tl_context_config_t **tl_config)
{
    ucc_status_t status;
    status = ucc_base_config_read(config->lib->full_prefix,
                                  &tl_lib->iface->tl_context_config,
                                  (ucc_base_config_t **)tl_config);
    if (UCC_OK == status) {
        (*tl_config)->tl_lib = tl_lib;
    }
    return status;
}

ucc_status_t ucc_tl_lib_config_read(ucc_tl_iface_t *iface,
                                    const char *full_prefix,
                                    ucc_tl_lib_config_t **tl_config)
{
    return ucc_base_config_read(full_prefix, &iface->tl_lib_config,
                                (ucc_base_config_t **)tl_config);
}

ucc_status_t ucc_tl_context_get(ucc_context_t *ctx, const char* name,
                                ucc_tl_context_t **tl_context)
{
    int i;
    ucc_tl_lib_t *tl_lib;
    for (i = 0; i < ctx->n_tl_ctx; i++) {
        tl_lib = ucc_derived_of(ctx->tl_ctx[i]->super.lib, ucc_tl_lib_t);
        if (0 == strcmp(tl_lib->iface->super.name, name)) {
            ctx->tl_ctx[i]->ref_count++;
            *tl_context = ctx->tl_ctx[i];
            return UCC_OK;
        }
    }
    return UCC_ERR_NOT_FOUND;
}

ucc_status_t ucc_tl_context_put(ucc_tl_context_t *tl_context)
{
    tl_context->ref_count--;
    return UCC_OK;
}

ucc_status_t
ucc_team_multiple_req_alloc(ucc_team_multiple_req_t **req,
                                   int n_teams)
{
    ucc_team_multiple_req_t *r;

    r = ucc_malloc(sizeof(ucc_team_multiple_req_t) +
                   (n_teams - 1) * sizeof(struct ucc_team_team_desc),
                   "team create multiple request");
    if (!r) {
        goto exit_err;
    }
    r->last    = -1;
    r->n_teams = n_teams;
    *req       = r;
    return UCC_OK;
exit_err:
    *req = NULL;
    return UCC_ERR_NO_MEMORY;
}

void ucc_team_multiple_req_free(ucc_team_multiple_req_t *req)
{
    ucc_free(req);
}

ucc_status_t ucc_tl_team_create_multiple(ucc_team_multiple_req_t *req)
{
    int *id = &req->last;
    ucc_base_team_t *b_team;
    ucc_status_t     status;

    if (*id == req->n_teams) {
        return UCC_OK;
    }
    if ((*id == -1) || (req->descs[*id].status != UCC_INPROGRESS)) {
        /* post next team */
        *id += 1;
        if (*id == req->n_teams) {
            return UCC_OK;
        }
        status = UCC_TL_CTX_IFACE(req->descs[*id].ctx)
                     ->team.create_post(&((req->descs[*id].ctx->super)),
                                        &req->descs[*id].param, &b_team);
        if (UCC_OK != status) {
            req->descs[*id].status = status;
            req->descs[*id].team   = NULL;
        } else {
            req->descs[*id].status = UCC_INPROGRESS;
            req->descs[*id].team   = ucc_derived_of(b_team, ucc_tl_team_t);
        }
        return UCC_INPROGRESS;
    }
    req->descs[*id].status = UCC_TL_CTX_IFACE(req->descs[*id].ctx)
                               ->team.create_test(&req->descs[*id].team->super);
    return UCC_INPROGRESS;
}

ucc_status_t ucc_tl_team_destroy_multiple(ucc_team_multiple_req_t *req)
{
    int         *id = &req->last;
    ucc_status_t status;
    if (*id < 0) (*id)++;
    while (*id != req->n_teams) {
        status = UCC_TL_TEAM_IFACE(req->descs[*id].team)
                     ->team.destroy(&req->descs[*id].team->super);
        req->descs[*id].status = status;
        if (UCC_INPROGRESS != status) {
            (*id)++;
        } else {
            return UCC_INPROGRESS;
        }
    }
    return UCC_OK;
}

UCC_CLASS_INIT_FUNC(ucc_tl_team_t, ucc_tl_context_t *tl_context,
                    const ucc_base_team_params_t *params)
{
    UCC_CLASS_CALL_BASE_INIT();
    self->super.context = &tl_context->super;
    self->super.params  = *params;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_team_t)
{
}

UCC_CLASS_DEFINE(ucc_tl_team_t, void);
