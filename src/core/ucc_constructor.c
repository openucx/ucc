/**
 * Copyright (C) Mellanox Technologies Ltd. 2020.  ALL RIGHTS RESERVED.
 *
 * See file LICENSE for terms.
 */

#include "config.h"
#include "ucc_global_opts.h"
#include "utils/ucc_malloc.h"
#include <link.h>
#include <dlfcn.h>
#include <string.h>

static int callback(struct dl_phdr_info *info, size_t size, void *data)
{
    char *str;
    char *component_path;
    if (NULL != (str = strstr(info->dlpi_name, "libucc.so"))) {
        int pos        = (int)(str - info->dlpi_name);
        component_path = (char *)ucc_malloc(pos + 8);
        if (!component_path) {
            //TODO add ucc_error(msg) when logging as added
            return -1;
        }
        ucc_strncpy_safe(component_path, info->dlpi_name, pos);
        component_path[pos] = '\0';
        strcat(component_path, "ucc");
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
            //TODO add ucc_error(msg) when logging is added
            return status;
        }
        if (strlen(ucc_global_config.component_path) == 0) {
            get_default_lib_path();
        }
    }
    return UCC_OK;
}

__attribute__((destructor)) static void ucc_destructor(void)
{
    if (ucc_global_config.component_path_default) {
        ucc_free(ucc_global_config.component_path_default);
    }
}
