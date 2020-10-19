/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_global_opts.h"
#include "utils/ucc_malloc.h"
#include "utils/ucc_component.h"
#include "utils/ucc_log.h"

#include <link.h>
#include <dlfcn.h>

#define UCC_LIB_SO_NAME "libucc.so"
#define UCC_COMPONENT_LIBDIR "ucc"
#define UCC_COMPONENT_LIBDIR_LEN strlen("ucc")

static int callback(struct dl_phdr_info *info, size_t size, void *data)
{
    char *str;
    char *component_path;
    if (NULL != (str = strstr(info->dlpi_name, UCC_LIB_SO_NAME))) {
        int pos        = (int)(str - info->dlpi_name);
        component_path = (char *)ucc_malloc(pos + UCC_COMPONENT_LIBDIR_LEN + 1,
                                            "component_path");
        if (!component_path) {
            ucc_error("failed to allocate %zd bytes for component_path",
                      pos + UCC_COMPONENT_LIBDIR_LEN + 1);
            return -1;
        }
        /* copying up to pos+1 due to ucs_strncpy_safe implementation specifics:
           it'll always write '\0' to the end position of the dest string. */
        ucc_strncpy_safe(component_path, info->dlpi_name, pos + 1);
        component_path[pos] = '\0';
        strcat(component_path, UCC_COMPONENT_LIBDIR);
        ucc_global_config.component_path =
            ucc_global_config.component_path_default = component_path;
    }
    return 0;
}

static void get_default_lib_path()
{
    dl_iterate_phdr(callback, NULL);
}

ucc_status_t ucc_constructor(void)
{
    ucc_status_t status;
    if (!ucc_global_config.initialized) {
        ucc_global_config.initialized            = 1;
        ucc_global_config.component_path_default = NULL;

        status = ucc_config_parser_fill_opts(
            &ucc_global_config, ucc_global_config_table, "UCC_", NULL, 1);
        if (UCC_OK != status) {
            ucc_error("failed to parse global options");
            return status;
        }
        if (strlen(ucc_global_config.component_path) == 0) {
            get_default_lib_path();
        }
    }

    if (!ucc_global_config.component_path) {
        ucc_error("failed to get ucc components path");
        return UCC_ERR_NOT_FOUND;
    }
    status = ucc_components_load("cl", &ucc_global_config.cl_framework);
    if (UCC_OK != status) {
        ucc_error("no CL components were found in the UCC_COMPONENT_PATH: %s",
                  ucc_global_config.component_path);
        return status;
    }
    return UCC_OK;
}

__attribute__((destructor)) static void ucc_destructor(void)
{
    if (ucc_global_config.initialized) {
        ucc_config_parser_release_opts(&ucc_global_config,
                                       ucc_global_config_table);
    }
}
