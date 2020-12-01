/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "ucc_cl.h"
#include "utils/ucc_log.h"
#include "utils/ucc_malloc.h"

ucc_config_field_t ucc_cl_lib_config_table[] = {
    {"LOG_LEVEL", "warn",
     "UCC CL logging level. Messages with a level higher or equal to the "
     "selected will be printed.\n"
     "Possible values are: fatal, error, warn, info, debug, trace, data, func, "
     "poll.",
     ucc_offsetof(ucc_cl_lib_config_t, log_component),
     UCC_CONFIG_TYPE_LOG_COMP},

    {"PRIORITY", "-1",
     "UCC CL priority.\n"
     "Possible values are: [1,inf]",
     ucc_offsetof(ucc_cl_lib_config_t, priority), UCC_CONFIG_TYPE_INT},

    {NULL}
};

ucc_config_field_t ucc_cl_context_config_table[] = {
    {NULL}
};

const char *ucc_cl_names[] = {
    [UCC_CL_BASIC] = "basic",
    [UCC_CL_ALL]   = "all",
    [UCC_CL_LAST]  = NULL
};

UCC_CLASS_INIT_FUNC(ucc_cl_lib_t, ucc_cl_iface_t *cl_iface,
                    const ucc_lib_config_t *config,
                    const ucc_cl_lib_config_t *cl_config)
{
    self->iface         = cl_iface;
    self->log_component = cl_config->log_component;
    ucc_strncpy_safe(self->log_component.name, cl_iface->cl_lib_config.name,
                     sizeof(self->log_component.name));
    self->priority =
        (-1 == cl_config->priority) ? cl_iface->priority : cl_config->priority;
    return UCC_OK;
}

UCC_CLASS_CLEANUP_FUNC(ucc_cl_lib_t)
{
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
            free(cls_copy);
            return UCC_ERR_INVALID_PARAM;
        }
        n_cls_selected++;
        cls_selected[cl_type] = 1;
        cl                    = strtok_r(NULL, ",", &saveptr);
    }
    free(cls_copy);

    cls = ucc_malloc(n_cls_selected * sizeof(ucc_cl_type_t), "cls_array");
    if (!cls) {
        ucc_error("failed to allocate %zd bytes for cls_array",
                  n_cls_selected * sizeof(int));
        return UCC_ERR_NO_MEMORY;
    }
    n_cls_selected = 0;
    for (cl_type = 0; cl_type < UCC_CL_LAST; cl_type++) {
        if (cls_selected[cl_type]) {
            cls[n_cls_selected++] = cl_type;
        }
    }
    *cls_array = cls;
    *n_cls     = n_cls_selected;
    return UCC_OK;
}

