/**
 * Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See file LICENSE for terms.
 */

#include "ucc_cl.h"
#include "utils/ucc_log.h"
#include "utils/ucc_malloc.h"
#include "core/ucc_global_opts.h"

static char * ucc_cl_tls_doc_str = "List of TLs used by a given CL component.\n"
    "Allowed values: either \"all\" or comma-separated list of: ";
#define TLS_CONFIG_ENTRY 1
ucc_config_field_t ucc_cl_lib_config_table[] = {
    [0] = {"", "", NULL, ucc_offsetof(ucc_cl_lib_config_t, super),
           UCC_CONFIG_TYPE_TABLE(ucc_base_lib_config_table)},

    [TLS_CONFIG_ENTRY] = {"TLS", "all", NULL,
                          ucc_offsetof(ucc_cl_lib_config_t, tls),
                          UCC_CONFIG_TYPE_ALLOW_LIST},

    {NULL}};

ucc_config_field_t ucc_cl_context_config_table[] = {
    [0] = {"", "", NULL, ucc_offsetof(ucc_cl_context_config_t, super),
     UCC_CONFIG_TYPE_TABLE(ucc_base_ctx_config_table)},

    {NULL}
};

const char *ucc_cl_names[] = {
    [UCC_CL_BASIC] = "basic",
    [UCC_CL_HIER]  = "hier",
    [UCC_CL_ALL]   = "all",
    [UCC_CL_LAST]  = NULL
};

UCC_CLASS_INIT_FUNC(ucc_cl_lib_t, ucc_cl_iface_t *cl_iface,
                    const ucc_cl_lib_config_t *cl_config)
{
    ucc_status_t status;

    UCC_CLASS_CALL_BASE_INIT();
    self->iface         = cl_iface;
    self->super.log_component = cl_config->super.log_component;
    ucc_strncpy_safe(self->super.log_component.name,
                     cl_iface->cl_lib_config.name,
                     sizeof(self->super.log_component.name));

    status = ucc_config_allow_list_process(
        &cl_config->tls, &ucc_global_config.tl_framework.names, &self->tls);
    if (status != UCC_OK) {
        return status;
    }
    if (self->tls.array.count == 0) {
        ucc_error("no TLs are selected for %s", cl_iface->cl_lib_config.name);
        ucc_free(self->tls.array.names);
        return UCC_ERR_NOT_FOUND;
    }
    self->tls_forced.count = 0;
    self->tls_forced.names = NULL;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_lib_t)
{
    ucc_config_names_array_free(&self->tls.array);
    ucc_config_names_array_free(&self->tls_forced);
}

UCC_CLASS_DEFINE(ucc_cl_lib_t, void);

ucc_status_t ucc_parse_cls_string(const char *cls_str,
                                  ucc_cl_type_t **cls_array, int *n_cls)
{
    int            cls_selected[UCC_CL_LAST] = {0};
    int            n_cls_selected            = 0;
    char          *cls_copy, *saveptr, *cl;
    ucc_cl_type_t *cls;
    ucc_cl_type_t  cl_type;
    cls_copy = strdup(cls_str);
    if (!cls_copy) {
        ucc_error("failed to create a copy cls string");
        return UCC_ERR_NO_MEMORY;
    }
    cl = strtok_r(cls_copy, ",", &saveptr);
    while (NULL != cl) {
        cl_type = ucc_cl_name_to_type(cl);
        if (cl_type == UCC_CL_LAST) {
            ucc_error("incorrect value is passed as part of UCC_CLS list: %s",
                      cl);
            ucc_free(cls_copy);
            return UCC_ERR_INVALID_PARAM;
        }
        n_cls_selected++;
        cls_selected[cl_type] = 1;
        cl                    = strtok_r(NULL, ",", &saveptr);
    }
    ucc_free(cls_copy);
    if (n_cls_selected == 0) {
        ucc_error("incorrect value is passed as part of UCC_CLS list: %s",
                  cls_str);
        return UCC_ERR_INVALID_PARAM;
    }
    cls = ucc_malloc(n_cls_selected * sizeof(ucc_cl_type_t), "cls_array");
    if (!cls) {
        ucc_error("failed to allocate %zd bytes for cls_array",
                  n_cls_selected * sizeof(int));
        return UCC_ERR_NO_MEMORY;
    }
    n_cls_selected = 0;
    for (cl_type = (ucc_cl_type_t)0; cl_type < UCC_CL_LAST; cl_type++) {
        if (cls_selected[cl_type]) {
            cls[n_cls_selected++] = cl_type;
        }
    }
    *cls_array = cls;
    *n_cls     = n_cls_selected;
    return UCC_OK;
}

UCC_CLASS_INIT_FUNC(ucc_cl_context_t, const ucc_cl_context_config_t *cl_config,
                    ucc_context_t *ucc_context)
{
    UCC_CLASS_CALL_BASE_INIT();
    self->super.lib         = &cl_config->cl_lib->super;
    self->super.ucc_context = ucc_context;

    if (0 == strcmp(cl_config->super.score_str, "0")) {
        return UCC_ERR_LAST;
    }
    self->super.score_str = strdup(cl_config->super.score_str);

    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_context_t)
{
    ucc_free(self->super.score_str);
}

UCC_CLASS_DEFINE(ucc_cl_context_t, void);

ucc_status_t ucc_cl_context_config_read(ucc_cl_lib_t *cl_lib,
                                        const ucc_context_config_t *config,
                                        ucc_cl_context_config_t **cl_config)
{
    ucc_status_t status;
    status = ucc_base_config_read(config->lib->full_prefix,
                                  &cl_lib->iface->cl_context_config,
                                  (ucc_base_config_t **)cl_config);
    if (UCC_OK == status) {
        (*cl_config)->cl_lib = cl_lib;
    }
    return status;
}

ucc_status_t ucc_cl_lib_config_read(ucc_cl_iface_t *iface,
                                    const char *full_prefix,
                                    ucc_cl_lib_config_t **cl_config)
{
    char  *tls_list;
    char  *doc_str;
    size_t doc_len;
    if (!ucc_cl_lib_config_table[TLS_CONFIG_ENTRY].doc) {
        tls_list = ucc_get_framework_components_list(
            &ucc_global_config.tl_framework, ",");
        if (tls_list) {
            doc_len = strlen(ucc_cl_tls_doc_str) + strlen(tls_list) + 1;
            doc_str = ucc_malloc(doc_len, "cl_tls_doc");
            if (!doc_str) {
                ucc_error("failed to allocate %zd bytes for cl_tls_doc",
                          doc_len);
            } else {
                ucc_snprintf_safe(doc_str, doc_len, "%s%s", ucc_cl_tls_doc_str,
                                  tls_list);
                ucc_cl_lib_config_table[TLS_CONFIG_ENTRY].doc = doc_str;
            }
            ucc_free(tls_list);
        }
    }
    return ucc_base_config_read(full_prefix, &iface->cl_lib_config,
                                (ucc_base_config_t **)cl_config);
}

UCC_CLASS_INIT_FUNC(ucc_cl_team_t, ucc_cl_context_t *cl_context,
                    const ucc_base_team_params_t *params)
{
    UCC_CLASS_CALL_BASE_INIT();
    self->super.context = &cl_context->super;
    self->super.params  = *params;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_team_t)
{
}

UCC_CLASS_DEFINE(ucc_cl_team_t, void);
