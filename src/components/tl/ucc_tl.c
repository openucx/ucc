/**
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_tl.h"
#include "utils/ucc_log.h"
#include "core/ucc_team.h"
#include "ucc_tl_log.h"

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
    ucc_base_lib_properties_t prop;
    ucc_status_t status;

    status = tl_iface->lib.get_properties(&prop);
    if (status != UCC_OK) {
        return status;
    }

    self->iface               = tl_iface;
    self->super.use_tuning    = tl_config->super.use_tuning;
    self->super.log_component = tl_config->super.log_component;
    self->super.min_team_size = prop.default_team_size;
    ucc_strncpy_safe(self->super.log_component.name,
                     tl_iface->tl_lib_config.name,
                     sizeof(self->super.log_component.name));

    if (tl_config->super.min_team_size != UCC_ULUNITS_AUTO) {
        if (tl_config->super.min_team_size < prop.min_team_size) {
            tl_warn(self, "min supported team size is %d, requested %d",
                    prop.min_team_size,
                    (ucc_rank_t)tl_config->super.min_team_size);
        } else if (tl_config->super.min_team_size > prop.max_team_size) {
            tl_warn(self, "max supported team size is %d, requested %d",
                    prop.max_team_size,
                    (ucc_rank_t)tl_config->super.min_team_size);
        } else {
            self->super.min_team_size = tl_config->super.min_team_size;
        }
    }

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
    self->ref_count         = 0;
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
ucc_team_multiple_req_alloc(ucc_team_multiple_req_t **req, int n_teams)
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

static ucc_status_t ucc_tl_is_reachable(const ucc_base_team_params_t *params,
                                        unsigned long tl_id)
{
    ucc_team_t                *core_team    = params->team;
    ucc_context_t             *core_context = core_team->contexts[0];
    ucc_addr_storage_t        *addr_storage;
    ucc_context_addr_header_t *addr_header;
    ucc_rank_t                 i, rank;
    int                        j, use_ctx;

    ucc_assert(core_team->num_contexts == 1);

    if (params->size == 1) {
        return UCC_OK;
    }

    if (core_context->addr_storage.storage) {
        addr_storage = &core_context->addr_storage;
        use_ctx = 1;
    } else {
        addr_storage = &core_team->addr_storage;
        use_ctx = 0;
    }

    if (addr_storage->flags & UCC_ADDR_STORAGE_FLAG_TLS_SYMMETRIC) {
        return UCC_OK;
    }

    for (i = 0; i < params->size; i++) {
        rank = ucc_ep_map_eval(params->map, i);
        if (use_ctx) {
            rank = ucc_ep_map_eval(core_team->ctx_map, rank);
        }
        addr_header = UCC_ADDR_STORAGE_RANK_HEADER(addr_storage, rank);
        for (j = 0; j < addr_header->n_components; j++) {
            if (addr_header->components[j].id == tl_id) {
                break;
            }
        }
        if (j == addr_header->n_components) {
            return UCC_ERR_NOT_FOUND;
        }
    }

    return UCC_OK;
}

ucc_status_t ucc_tl_team_create_multiple(ucc_team_multiple_req_t *req)
{
    int             *id = &req->last;
    ucc_base_team_t *b_team;
    ucc_status_t     status;
    ucc_tl_lib_t    *lib;

    if (*id == req->n_teams) {
        return UCC_OK;
    }
    if ((*id == -1) || (req->descs[*id].status != UCC_INPROGRESS)) {
        /* post next team */
        *id += 1;
        if (*id == req->n_teams) {
            return UCC_OK;
        }
        lib = ucc_derived_of(req->descs[*id].ctx->super.lib, ucc_tl_lib_t);
        status = ucc_tl_is_reachable(&req->descs[*id].param,
                                     lib->iface->super.id);
        if (UCC_OK != status) {
            ucc_debug("TL %s is not reachable, skipping\n",
                      lib->iface->super.name);
        } else {
            status = UCC_TL_CTX_IFACE(req->descs[*id].ctx)
                        ->team.create_post(&((req->descs[*id].ctx->super)),
                                            &req->descs[*id].param, &b_team);
        }
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
    ucc_base_lib_t *lib = tl_context->super.lib;
    ucc_base_lib_attr_t attr;
    ucc_tl_iface_t *tl_iface;
    ucc_status_t status;

    self->super.context = &tl_context->super;
    self->super.params  = *params;

    tl_iface = UCC_TL_CTX_IFACE(tl_context);
    attr.mask = UCC_BASE_LIB_ATTR_FIELD_MIN_TEAM_SIZE |
                UCC_BASE_LIB_ATTR_FIELD_MAX_TEAM_SIZE;
    status = tl_iface->lib.get_attr(lib, &attr);
    if (status != UCC_OK) {
        return status;
    }

    if (attr.min_team_size > params->size) {
        tl_debug(lib, "team size %d is too small, min supported %d",
                 params->size, attr.min_team_size);
        return UCC_ERR_NOT_SUPPORTED;
    }

    if (attr.max_team_size < params->size) {
        tl_debug(lib, "team size %d is too big, max supported %d",
                 params->size, attr.max_team_size);
        return UCC_ERR_NOT_SUPPORTED;
    }

    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_tl_team_t)
{
}

UCC_CLASS_DEFINE(ucc_tl_team_t, void);
