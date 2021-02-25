/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucc_tl.h"
#include "utils/ucc_log.h"

ucc_config_field_t ucc_tl_lib_config_table[] = {
    {"", "", NULL, ucc_offsetof(ucc_tl_lib_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_base_config_table)},

    {NULL}
};

ucc_config_field_t ucc_tl_context_config_table[] = {
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

UCC_CLASS_INIT_FUNC(ucc_tl_context_t, ucc_tl_lib_t *tl_lib,
                    ucc_context_t *ucc_context)
{
    UCC_CLASS_CALL_BASE_INIT();
    self->super.lib         = &tl_lib->super;
    self->super.ucc_context = ucc_context;
    self->ref_count = 0;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_context_t)
{
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

ucc_status_t ucc_tl_context_get(ucc_context_t *ctx, ucc_tl_type_t type,
                                ucc_tl_context_t **tl_context)
{
    int i;
    ucc_tl_lib_t *tl_lib;
    for (i = 0; i < ctx->n_tl_ctx; i++) {
        tl_lib = ucc_derived_of(ctx->tl_ctx[i]->super.lib, ucc_tl_lib_t);
        if (tl_lib->iface->type == type) {
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

ucc_status_t ucc_team_create_multiple_req_alloc(ucc_team_create_multiple_req_t **req,
                                        int n_teams)
{
    ucc_team_create_multiple_req_t *r;

    r = ucc_malloc(sizeof(ucc_team_create_multiple_req_t),
                          "team create multiple req alloc");
    if (!r) {
        goto exit_err;
    }
    r->contexts = ucc_malloc(n_teams * sizeof(ucc_tl_context_t*),
                             "team create multiple req alloc");
    if (!r->contexts) {
        goto free_req;
    }
    r->params = ucc_malloc(n_teams * sizeof(ucc_base_team_params_t*),
                           "team create multiple req alloc");
    if (!r->params) {
        goto free_contexts;
    }
    r->teams = ucc_malloc(n_teams * sizeof(ucc_tl_team_t*),
                          "team create multiple req alloc");
    if (!r->teams) {
        goto free_params;
    }
    r->teams_status = ucc_malloc(n_teams * sizeof(ucc_status_t),
                                 "team create multiple req alloc");
    if (!r->teams_status) {
        goto free_teams;
    }
    r->last_created = -1;
    r->n_teams = n_teams;
    *req = r;
    return UCC_OK;
free_teams:
    ucc_free(r->teams);
free_params:
    ucc_free(r->params);
free_contexts:
    ucc_free(r->contexts);
free_req:
    ucc_free(r);
exit_err:
    *req = NULL;
    return UCC_ERR_NO_MEMORY;
}

void ucc_team_create_multiple_req_free(ucc_team_create_multiple_req_t *req)
{
    ucc_free(req->contexts);
    ucc_free(req->params);
    ucc_free(req->teams);
    ucc_free(req->teams_status);
    ucc_free(req);
}

ucc_status_t ucc_tl_team_create_multiple(ucc_team_create_multiple_req_t *req)
{
    ucc_base_team_t *b_team;
    ucc_status_t     status;
    int *id = &req->last_created;

    if (*id == req->n_teams) {
        return UCC_OK;
    }
    if ((*id == -1) || (req->teams_status[*id] != UCC_INPROGRESS)) {
        /* post next team */
        *id += 1;
        if (*id == req->n_teams) {
            return UCC_OK;
        }
        status = UCC_TL_CTX_IFACE(req->contexts[*id])->team.create_post(
                    &((req->contexts[*id])->super),
                    req->params[*id],
                    &b_team);
        if (UCC_OK != status) {
            req->teams_status[*id] = status;
            req->teams[*id] = NULL;
        } else {
            req->teams_status[*id] = UCC_INPROGRESS;
            req->teams[*id] = ucc_derived_of(b_team, ucc_tl_team_t);
        }
        return UCC_INPROGRESS;
    }
    req->teams_status[*id] = UCC_TL_CTX_IFACE(req->contexts[*id])->team.create_test(&req->teams[*id]->super);
    return UCC_INPROGRESS;
}

UCC_CLASS_INIT_FUNC(ucc_tl_team_t, ucc_tl_context_t *tl_context)
{
    UCC_CLASS_CALL_BASE_INIT();
    self->super.context = &tl_context->super;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_team_t)
{
}

UCC_CLASS_DEFINE(ucc_tl_team_t, void);
